// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "bilstm.h"
#include <math.h>

namespace ncnn {

DEFINE_LAYER_CREATOR(BiLSTM)

BiLSTM::BiLSTM()
{
    one_blob_only = false;
    support_inplace = false;
}

int BiLSTM::load_param(const ParamDict& pd)
{
    num_lstm_layer = pd.get(0, 0);
    isbilstm = pd.get(1, 0);
    num_output = pd.get(2, 0);
    weight_xc_data_size = pd.get(3, 0);
    weight_hc_data_size = pd.get(4, 0);
    return 0;
}

int BiLSTM::load_model(const ModelBin& mb)
{
    int numdir = isbilstm ? 2 : 1;

  //  int xc_size = weight_xc_data_size / num_output / 4;
  //  int hc_size = weight_hc_data_size / num_output / 4;

    weight_xc_data.resize(num_lstm_layer*numdir);
    bias_xc_data.resize(num_lstm_layer*numdir);
    weight_hc_data.resize(num_lstm_layer*numdir);
    bias_hc_data.resize(num_lstm_layer*numdir);
    
    // raw weight data
    for(int l=0; l<num_lstm_layer; l++)
    {
        for(int i=0; i<numdir; i++)
        {
            if(l == 0)
            {   //    fprintf(stderr,"________%d", l);
                weight_xc_data[i*num_lstm_layer+l] = mb.load(weight_xc_data_size, 0);
                if (weight_xc_data.empty())
                return -100;
                //fprintf(stderr,"weightxxx________%f", ((const float*)weight_xc_data[i*num_lstm_layer+l])[1]);
            }
            else
            {
                weight_xc_data[i*num_lstm_layer+l] = mb.load(weight_hc_data_size, 0);
                if (weight_xc_data.empty())
                return -100;
            }
    	
            bias_xc_data[i*num_lstm_layer+l] = mb.load(num_output*4, 1);
    	    if (bias_xc_data.empty())
                return -100;

    	    weight_hc_data[i*num_lstm_layer+l] = mb.load(weight_hc_data_size, 0);
    	    if (weight_hc_data.empty())
                return -100;
   
            bias_hc_data[i*num_lstm_layer+l] = mb.load(num_output*4, 1);
            if (bias_hc_data.empty())
                return -100;
        }
        if(l == 0)
        {
            layer1_out_weight = mb.load(num_output*num_output*2, 0);
            layer1_out_bias = mb.load(num_output, 1);

         //   fprintf(stderr, "%f ",*((float*)layer1_out_weight.data+1));
        }
      // fprintf(stderr,"________%d", hc_size);
        
    }


    return 0;
}

int BiLSTM::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs) const
{

    // size x 1 x T
    const Mat& input_blob = bottom_blobs[0];

    // T, 0 or 1 each
    const Mat& cont_blob = bottom_blobs[1];

    int T = input_blob.c;
    //int size = input_blob.w;
    int numdir = isbilstm ? 2 : 1;

    // initial hidden state
    Mat hidden(num_output, numdir);
    if (hidden.empty())
        return -100;
    //hidden.fill(0.f);

    // internal cell state
    Mat cell(num_output, numdir);
    if (cell.empty())
        return -100;
   // cell.fill(0.f);

    // 4 x num_output
    Mat gates(4, num_output);
    if (gates.empty())
        return -100;

    Mat& top_blob = top_blobs[0];
    top_blob.create(num_output*numdir, T);
    if (top_blob.empty())
        return -100;
    
    const float* x_data;
    int size;
   
    Mat layer1_out(num_output*numdir, T);
    Mat layer1Out_liner(num_output, T);
    float* cell_data = cell;
    float* hidden_data = hidden; 

    // unroll
    for (int l=0; l<num_lstm_layer; l++)
    {

        for(int t=0; t<T; t++)
        {
            for (int d=0; d<numdir; d++)
            {     
                 
                const  float cont = cont_blob[T*d+t];
        //        fprintf(stderr, "%f  %f  d:%d  t:%d\n",cell[num_output*d+6], hidden_data[num_output*d+6], d, t);

                if (l == 0 && d == 0)
                {
                    x_data = input_blob.channel(t);
                    size = input_blob.w;
                }
                else if(l == 0 && d == 1)
                {
               // fprintf(stderr, "%d", T-t-1);
                    x_data = input_blob.channel(T-t-1);
                    size = input_blob.w;             
                }
                else if(l == 1  && d == 0)
                { 
                    x_data = layer1Out_liner.row(t);
                    //x.elemsize = 
                    //x.data = hidden_data+num_output*(l-1);
                    size = num_output;
                                                          // fprintf(stderr,"\n%d",size);
                }
                else 
                {
                    x_data = layer1Out_liner.row(T-t-1);
                    size = num_output;
                }
             
                int num_layer = d ? (l+2) : l;
          // fprintf(stderr,"num_layer:%d\n",num_layer);
                for (int q=0; q<num_output; q++)
                {
            //   float h_cont = cont ? hidden_data[num_output*l+q] : 0.f;
            /*   if(d == 0 && t == 1)
               {
                   if(l == 0)
                   fprintf(stderr, "l0_h_cont_q: %d-%f\n", q, h_cont);
                   if(l == 1)
                   fprintf(stderr, "l1_h_cont_q: %d-%f\n", q, h_cont);
               }*/
                  //  const float* x_data = x;
           
                    const float* bias_xc_data_ptr = (const float*)bias_xc_data[num_layer];
                    const float* bias_hc_data_ptr = (const float*)bias_hc_data[num_layer];
            
                    float* gates_data = (float*)gates + 4 * q;

            // gate I F O G
                    const float* weight_hc_data_I = (const float*)weight_hc_data[num_layer] + num_output * q;
                    const float* weight_xc_data_I = (const float*)weight_xc_data[num_layer] + size * q;
                    const float* weight_hc_data_F = (const float*)weight_hc_data[num_layer] + num_output * q + weight_hc_data[num_layer].w /4;
                    const float* weight_xc_data_F = (const float*)weight_xc_data[num_layer] + size * q + weight_xc_data[num_layer].w /4 ;
                    const float* weight_hc_data_O = (const float*)weight_hc_data[num_layer] + num_output * q + weight_hc_data[num_layer].w * 3 /4;
                    const float* weight_xc_data_O = (const float*)weight_xc_data[num_layer] + size * q + weight_xc_data[num_layer].w * 3 / 4 ;
                    const float* weight_hc_data_G = (const float*)weight_hc_data[num_layer] + num_output * q + weight_hc_data[num_layer].w/2;
                    const float* weight_xc_data_G = (const float*)weight_xc_data[num_layer] + size * q + weight_xc_data[num_layer].w/2;

                    float I = bias_hc_data_ptr[q] + bias_xc_data_ptr[q];
                    float F = bias_hc_data_ptr[q+num_output*2] + bias_xc_data_ptr[q+num_output];
                    float O = bias_hc_data_ptr[q+num_output*3] + bias_xc_data_ptr[q+num_output*3];
                    float G = bias_hc_data_ptr[q+num_output] + bias_xc_data_ptr[q+num_output*2];

            /* if(d==0 && t == 1)
                  {
                  if(l == 0)
                  {fprintf(stderr,"l0_weight_xc_data_I:%f.....\n", weight_xc_data_I[i]);
                  fprintf(stderr,"l0_weight_xc_data_F:%f.....\n", weight_xc_data_F[i]);
                  fprintf(stderr,"l0_weight_xc_data_O:%f.....\n", weight_xc_data_O[i]);}
                  if(l == 1)
                   {fprintf(stderr,"l1_weight_xc_data_I:%f.....\n", weight_xc_data_I[i]);
                  fprintf(stderr,"l1_weight_xc_data_F:%f.....\n", weight_xc_data_F[i]);
                  fprintf(stderr,"l1_weight_xc_data_O:%f.....\n", weight_xc_data_O[i]);}
                 }*/

                    for (int i=0; i<num_output; i++)
                    {
                       float h_cont = cont ? hidden_data[num_output*d+i] : 0.f;
            /*      if(d == 1 && t == 0)
               {
                   if(l == 0)
                   fprintf(stderr, "l0_h_cont_q: %d-%f\n", q, h_cont);
                   if(l == 1)
                   fprintf(stderr, "l1_h_cont_q: %d-%f\n", q, h_cont);
               }*/
                   //     float h_cont = hidden_data[num_output*d+i];
                        I += weight_hc_data_I[i] * h_cont; 
                        F += weight_hc_data_F[i] * h_cont; 
                        O += weight_hc_data_O[i] * h_cont; 
                        G += weight_hc_data_G[i] * h_cont; 
                    }

                    for(int i=0; i<size; i++)
                    {
                                                       //      fprintf(stderr,"%d.....",size);
                        G += weight_xc_data_G[i] * x_data[i];
                        O += weight_xc_data_O[i] * x_data[i];
                        F += weight_xc_data_F[i] * x_data[i];
                        I += weight_xc_data_I[i] * x_data[i];
                    } 
                                                  //    fprintf(stderr,"---%d",q);
                    gates_data[0] = I;
                    gates_data[1] = F;
                    gates_data[2] = O;
                    gates_data[3] = G;
                }

        // lstm unit
        // sigmoid(I)
        // sigmoid(F)
        // sigmoid(O)
        // tanh(G)
        // c_t := f_t .* c_{t-1} + i_t .* g_t
        // h_t := o_t .* tanh[c_t]
               
                int tt;
                if(d == 1)
                { tt = T - t -1; }
                else
                tt = t;
                
                
                float* output_data = top_blob.row(tt); 

                for (int q=0; q<num_output; q++)
                {
                    float* gates_data = (float*)gates + 4 * q;

                    float I = gates_data[0];
                    float F = gates_data[1];
                    float O = gates_data[2];
                    float G = gates_data[3];
             /*  if(d == 1 && t == 34)
               {
                   if(l == 0)
                   fprintf(stderr, "layer0_I_F_O_G: %f, %f, %f, %f，%f\n", I, F, O, G, 1.f / (1.f + exp(-I))*tanh(G));
                   if(l == 1)
                   fprintf(stderr, "layer1_I_F_O_G: %f, %f, %f, %f\n", I, F, O, G);
               }  */
              // if(d == 1 && t == 0)
              // cont = 0;
                    I = 1.f / (1.f + exp(-I));
                    F = cont ? 1.f / (1.f + exp(-F)) : 0.f;
                //    F = 1.f / (1.f + exp(-F));
                    O = 1.f / (1.f + exp(-O));
                    G = tanh(G);
               
                    float cell = F * cell_data[num_output*d+q] + I * G;
                    float H = O * tanh(cell);

                    cell_data[num_output*d+q] = cell;
                    hidden_data[num_output*d+q] = H;
               
               //  if(d == 1 && l == 1 && t == 34)
               // {fprintf(stderr,"cell:%f..F:%f..l1_output: %f\n", cell, cont, H);}
                    if(l == 0)
                    {
                        float* layer1_output_data = layer1_out.row(tt);
                        layer1_output_data[num_output*d+q] = H;
                        if(t == 0)
                        fprintf(stderr,"l0_output(t:%d): %f\n", t, H);
                    }

                    if(l == num_lstm_layer-1)
                    {
                        output_data[num_output*d+q] = H;
                       // if(d == 0)
                       // fprintf(stderr,"forward_output(t:%d): %f\n", t, H);
                       // if(d == 1)
                       // fprintf(stderr,"backward_output(t:%d): %f\n",t, H);

                    }
  
            /*    if(d == 0 && l == num_lstm_layer-1)
                {output_data[q] = H;
                 if(t == 34)
                 fprintf(stderr,"output_forward: %f\n",H);}
                else if(d == 1 && l == num_lstm_layer-1)
                { output_data[num_output*numdir-q-1] = H;
                  if(t == 0)
                  fprintf(stderr,"output_rever: %f\n",H); }*/
                }
            }  

        // no cell output here
        }

        if(l == 0)
        {
            for (int t=0; t<T; t++)
            {
                float sum = 0.f;
              //  fprintf(stderr, "\n");
                for (int n_out=0; n_out<num_output; n_out++)
                {
                //    if (bias_term)
                    sum = layer1_out_bias[n_out];

                    for (int q=0; q<layer1_out.w; q++)
                    {
                        const float* w = (const float*)layer1_out_weight + q + n_out * layer1_out.w;
                        const float* m = layer1_out.row(t) + q;
                        sum += (*m) * (*w);
                    }
                    *(layer1Out_liner.row(t)+n_out) = sum;

              //      if(t == 0)
             //       fprintf(stderr,"output_l1:%d__ %f \n",t, sum);

                }
            }
        }
    }
    return 0;
}

} // namespace ncnn

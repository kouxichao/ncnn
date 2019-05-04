#include "lstmcell.h"
#include <math.h>

namespace ncnn {

DEFINE_LAYER_CREATOR(LSTMCell)

LSTMCell::LSTMCell()
{
    one_blob_only = false;
    support_inplace = false;
}

int LSTMCell::load_param(const ParamDict& pd)
{
    input_size = pd.get(0, 0);
    hidden_size = pd.get(1, 0);
    return 0;
}

int LSTMCell::load_model(const ModelBin& mb)
{
    long out_len = hidden_size * 4;
    long weight_ih_size = input_size * out_len;
    long weight_hh_size = hidden_size * out_len;
 
    weight_ih_data = mb.load(weight_ih_size, 0);
    if(weight_ih_data.empty())
        return -100;

    weight_hh_data = mb.load(weight_hh_size, 0);
    if(weight_hh_data.empty())
        return -100;

    bias_ih_data = mb.load(out_len, 0);
    if(bias_ih_data.empty())
        return -100;

    bias_hh_data = mb.load(out_len, 0);
    if(bias_hh_data.empty())
        return -100;


    return 0;
}

int LSTMCell::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs) const
{
    const Mat& input_blob = bottom_blobs[0];
    const Mat& hidden_blob = bottom_blobs[1];

    Mat& top_blob = top_blobs[0];
    top_blob.create(input_blob.h, hidden_size);
    if (top_blob.empty())
        return -100;



    const float *ih_I = (const float*)weight_ih_data;
    const float *ih_F = (const float*)weight_ih_data + input_size * hidden_size;
    const float *ih_G = (const float*)weight_ih_data + input_size * hidden_size * 2;
    const float *ih_O = (const float*)weight_ih_data + input_size * hidden_size * 3;

    const float *hh_I = (const float*)weight_hh_data;
    const float *hh_F = (const float*)weight_hh_data + hidden_size * hidden_size;
    const float *hh_G = (const float*)weight_hh_data + hidden_size * hidden_size * 2;
    const float *hh_O = (const float*)weight_hh_data + hidden_size * hidden_size * 3;
//    Mat gates(input_blob.h, hidden_size, 4);
//    if (gates.empty())
//        return -100;
    
    long sample_out_len = 4 * hidden_size;
    for (int out_ele=0; out_ele<hidden_size; out_ele++)
    {
        const float* weight_ih_data_I = ih_I + input_size * out_ele;
        const float* weight_hh_data_I = hh_I + hidden_size * out_ele; 
        const float* weight_ih_data_F = ih_F + input_size * out_ele;
        const float* weight_hh_data_F = hh_F + hidden_size * out_ele; 
        const float* weight_ih_data_O = ih_O + input_size * out_ele; 
        const float* weight_hh_data_O = hh_O + hidden_size * out_ele; 
        const float* weight_ih_data_G = ih_G + input_size * out_ele; 
        const float* weight_hh_data_G = hh_G + hidden_size * out_ele; 

        float I = bias_hh_data[out_ele] + bias_ih_data[out_ele];
        float F = bias_hh_data[out_ele + hidden_size] + bias_ih_data[out_ele + hidden_size];
        float O = bias_hh_data[out_ele + hidden_size * 3] + bias_ih_data[out_ele + hidden_size * 3];
        float G = bias_hh_data[out_ele + hidden_size] + bias_ih_data[out_ele + hidden_size];

        if(input_blob.h != hidden_blob.h)
        {
            return -1;
        }
        for(size_t i = 0; i < input_blob.h; i++)        
        {
//            float *gates_ptr = (float*)gates.channel(i) + 4 * out_ele;
            for(size_t j = 0; j < hidden_size; j++)
            {
                I += *((const float*)hidden_blob.row(i) + j) * weight_hh_data_I[i]; 
                F += *((const float*)hidden_blob.row(i) + j) * weight_hh_data_F[i]; 
                O += *((const float*)hidden_blob.row(i) + j) * weight_hh_data_O[i]; 
                G += *((const float*)hidden_blob.row(i) + j) * weight_hh_data_G[i]; 
            }

            for(size_t j = 0; j < input_size; j++)
            {
                I += *((const float*)input_blob.row(i) + j) * weight_ih_data_I[i];
                F += *((const float*)input_blob.row(i) + j) * weight_ih_data_F[i];
                O += *((const float*)input_blob.row(i) + j) * weight_ih_data_O[i];
                G += *((const float*)input_blob.row(i) + j) * weight_ih_data_G[i];
            }
//            gates_ptr[0] = I;
//            gates_ptr[1] = F;
//            gates_ptr[2] = O;
//            gates_ptr[3] = G;
            I = 1.f / (1.f + exp(-I));
            F = 1.f / (1.f + exp(-F));
            O = 1.f / (1.f + exp(-O));
            G = tanh(G);
            float cell = F * cell_data[hidden_size*i + out_ele] + I * G;
            float H = O * tanh(cel
            cell_data[num_output*l+q] = cell;
            hidden_data[num_output*l
            if(l == num_lstm_layer-1)
                output_data[num_output*d+q] = H;
            }
        }
    }

    // lstm unit
    // sigmoid(I)
    // sigmoid(F)
    // sigmoid(O)
    // tanh(G)
    // c_t := f_t .* c_{t-1} + i_t .* g_t
    // h_t := o_t .* tanh[c_t]
    
    float* output_data = top_blob.
    
    for(size_t i = 0; i < input_blob.h; i++)
    {

    }
    return 0;
}

} // namespace ncnn
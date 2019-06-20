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
 
    weight_ih_data = mb.load(weight_ih_size, 1);
    if(weight_ih_data.empty())
        return -100;

    weight_hh_data = mb.load(weight_hh_size, 1);
    if(weight_hh_data.empty())
        return -100;

    bias_ih_data = mb.load(out_len, 1);
    if(bias_ih_data.empty())
        return -100;


    bias_hh_data = mb.load(out_len, 1);
    if(bias_hh_data.empty())
        return -100;


    return 0;
}

int LSTMCell::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& input_blob = bottom_blobs[0];
    const Mat& hidden_blob = bottom_blobs[1];
    const Mat& cell_blob = bottom_blobs[2];
    size_t elemsize = bottom_blobs[0].elemsize;

    /**
     * this variable is just used for several
     * input samples but only one h and one c,
     * this case usually happens when lstm begin.
     * use this tech for save input size. 
     */
    int index_indicator = hidden_blob.h > 1 ? 1 : 0; 

    Mat& top_blob_h = top_blobs[0];
    Mat& top_blob_c = top_blobs[1];
    top_blob_h.create(hidden_size, input_blob.h, elemsize, opt.blob_allocator);
    top_blob_c.create(hidden_size, input_blob.h, elemsize, opt.blob_allocator);

    if (top_blob_h.empty() || top_blob_c.empty())
        return -100;

    //gates weights_blob of inputs(weights order I,F,G,O, same as pytorch's weight order)
    const float *ih_I = (const float*)weight_ih_data;
    const float *ih_F = (const float*)weight_ih_data + input_size * hidden_size;
    const float *ih_G = (const float*)weight_ih_data + input_size * hidden_size * 2;
    const float *ih_O = (const float*)weight_ih_data + input_size * hidden_size * 3;

    //gates weights_blob of pre_hidden(weights order I,F,G,O, same as pytorch's weight order)
    const float *hh_I = (const float*)weight_hh_data;
    const float *hh_F = (const float*)weight_hh_data + hidden_size * hidden_size;
    const float *hh_G = (const float*)weight_hh_data + hidden_size * hidden_size * 2;
    const float *hh_O = (const float*)weight_hh_data + hidden_size * hidden_size * 3;
    
    /**lstm unit 
     * sigmoid(I)
     * sigmoid(F)
     * sigmoid(O)
     * tanh(G)
     * c_t := f_t .* c_{t-1} + i_t .* g_t
     * h_t := o_t .* tanh[c_t]
     */
   // fprintf(stderr, "num_threads:%d\n", opt.num_threads);
    #pragma omp parallel for num_threads(4)
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


        if(input_blob.h != hidden_blob.h && hidden_blob.h != 1)
        {
            fprintf(stderr, "shape don't match !\n");
            //return -101;
        }

        for(size_t i = 0; i < input_blob.h; i++)        
        {
            float I = bias_hh_data[out_ele] + bias_ih_data[out_ele];
            float F = bias_hh_data[out_ele + hidden_size] + bias_ih_data[out_ele + hidden_size];
            float O = bias_hh_data[out_ele + hidden_size * 3] + bias_ih_data[out_ele + hidden_size * 3];
            float G = bias_hh_data[out_ele + hidden_size * 2] + bias_ih_data[out_ele + hidden_size * 2];

            for(size_t j = 0; j < hidden_size; j++)
            {
                I += *((const float*)hidden_blob.row(i * index_indicator) + j) * weight_hh_data_I[j]; 
                F += *((const float*)hidden_blob.row(i * index_indicator) + j) * weight_hh_data_F[j]; 
                O += *((const float*)hidden_blob.row(i * index_indicator) + j) * weight_hh_data_O[j]; 
                G += *((const float*)hidden_blob.row(i * index_indicator) + j) * weight_hh_data_G[j]; 
            }

            for(size_t j = 0; j < input_size; j++)
            {
                I += *((const float*)input_blob.row(i) + j) * weight_ih_data_I[j];
                F += *((const float*)input_blob.row(i) + j) * weight_ih_data_F[j];
                O += *((const float*)input_blob.row(i) + j) * weight_ih_data_O[j];
                G += *((const float*)input_blob.row(i) + j) * weight_ih_data_G[j];
            }

            I = 1.f / (1.f + exp(-I));
            F = 1.f / (1.f + exp(-F));
            O = 1.f / (1.f + exp(-O));
            G = tanh(G);
            float cell = F * cell_blob[hidden_size*i*index_indicator + out_ele] + I * G;
            float H = O * tanh(cell);

            //save output h,c in top_blob 
            *((float *)top_blob_h.row(i) + out_ele) = H;
            *((float *)top_blob_c.row(i) + out_ele) = cell;
        }
    }

    return 0;
}

} // namespace ncnn
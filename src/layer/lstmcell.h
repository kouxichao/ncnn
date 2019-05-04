#ifndef LAYER_LSTMCell_H
#define LAYER_LSTMCell_H

#include "layer.h"

namespace ncnn {

class LSTMCell : public Layer
{
public:
    LSTMCell();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs) const;

public:
    // param
    int input_size;                         
    int hidden_size;
//    int num_output;
//    int weight_xc_data_size;
// /    int weight_hc_data_size;

    // model
    Mat weight_ih_data;
    Mat weight_hh_data;
    Mat bias_ih_data;
    Mat bias_hh_data;
};

} // namespace ncnn

#endif //LAYER_LSTMCell_H
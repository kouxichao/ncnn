#ifndef LAYER_DIMOP_H
#define LAYER_DIMOP_H

#include "layer.h"

namespace ncnn {

class DimOp : public Layer
{
public:
    DimOp();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

//    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;

#if NCNN_VULKAN
    virtual int create_pipeline();
    virtual int destroy_pipeline();

    virtual int forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const;

    virtual int forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const;
#endif // NCNN_VULKAN

    enum {
        Operation_ADD   = 0,
        Operation_SUB   = 1,
        Operation_MUL   = 2,
        Operation_DIV   = 3,
        Operation_MAX   = 4,
        Operation_MIN   = 5,
        Operation_POW   = 6,
        Operation_RSUB  = 7,
        Operation_RDIV  = 8
    };

public:
    // param
    int op_type;
    int dim;

#if NCNN_VULKAN
    Pipeline* pipeline_binaryop;
    Pipeline* pipeline_binaryop_pack4;
#endif // NCNN_VULKAN
};

} // namespace ncnn

#endif // LAYER_BINARYOP_H

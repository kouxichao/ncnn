#include "dimop.h"

namespace ncnn{

DEFINE_LAYER_CREATOR(DimOp)

DimOp::DimOp()
{
    one_blob_only = false;
    support_inplace = false;
    support_vulkan = false;

#if NCNN_VULKAN
    pipeline_binaryop = 0;
    pipeline_binaryop_pack4 = 0;
#endif // NCNN_VULKAN
}

int DimOp::load_param(const ParamDict &pd)
{
    op_type = pd.get(0, 0); 
    dim = pd.get(1, 0);

    return 0;
}

int DimOp::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    Mat& top_blob = top_blobs[0];

    if(op_type == Operation_ADD)
    {
        if(dim == 0)
        {
            top_blob.create(h, channels, elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;
            for(size_t d = 0; d < channels; d++)
            {
                float *outptr = top_blob.row(d);
                const float *inptr = bottom_blob.channel(d); 
                for(size_t i = 0; i < h; i++)
                {
                    int numrow = i*w;
                    for(size_t i = 0; i < w; i++)
                    {
                        outptr[i] += *(inptr + numrow + w);
                        
                    }
                }
            }
            return 0; 
        }
        else if(dim == 1)
        {
            top_blob.create(w, channels, elemsize, opt.blob_allocator);
            top_blob.fill(0.f);
            if (top_blob.empty())
                return -100;
            for(size_t d = 0; d < channels; d++)
            {
                float *outptr = top_blob.row(d);
                const float *inptr = bottom_blob.channel(d); 
                for(size_t i = 0; i < h; i++)
                {
                    int numrow = i*w;
                    for(size_t j = 0; j < w; j++)
                    {
                        float in = *(inptr + numrow + j);
                        outptr[j] += in;
                    }
                }
            }
            return 0; 
        }
        else if(dim == 2)
        {
            return 0;
        }
        else
        {
            fprintf(stderr, "param dim should be less than 3!");
            return -101;
        }
    }
    
    return 0;
}

}
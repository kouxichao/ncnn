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

#include "binaryop.h"
#include <math.h>
#include <algorithm>
#include <functional>

namespace ncnn {

DEFINE_LAYER_CREATOR(BinaryOp)

BinaryOp::BinaryOp()
{
    one_blob_only = false;
    support_inplace = false;
    support_vulkan = true;

#if NCNN_VULKAN
    pipeline_binaryop = 0;
    pipeline_binaryop_pack4 = 0;
#endif // NCNN_VULKAN
}

int BinaryOp::load_param(const ParamDict& pd)
{
    op_type = pd.get(0, 0);
    with_scalar = pd.get(1, 0);
    b = pd.get(2, 0.f);

    if (with_scalar != 0)
    {
        one_blob_only = true;
        support_inplace = true;
    }

    return 0;
}

/*
#define DIMENSION_3(blob) blob.c != 1
#define DIMENSION_2(blob) blob.h != 1 && blob.c == 1
#define DIMENSION_1(blob) blob.w != 1 &&  blob.h == 1 && blob.c == 1
#define DIMENSION_0(blob) blob.w == 1 && blob.h == 1 && blob.c == 1 
*/
#define INDEX_INDICATOR(blob,dim) blob.dim > 1 ? 1:0

template<typename Op>
static int binary_op(const Mat& a, const Mat& b, Mat& c, const Option& opt)
{
    Op op;
    int w = std::max(a.w, b.w);
    int h = std::max(a.h, b.h);
    int channels = std::max(a.c, b.c);
    size_t elemsize = a.elemsize;

    c.create(w, h, channels, elemsize, opt.blob_allocator);
    if (c.empty())
        return -100;

    if ((a.c != b.c && a.c !=1 && b.c != 1)|| (a.w != b.w && a.w != 1 && b.w != 1) || \
    (a.h != b.h && a.h != 1 && b.h != 1)) {
        fprintf(stderr, "The size of two blobs don't match !\n");
        return -101;
    }
    int a_wInd = INDEX_INDICATOR(a, w);
    int a_hInd = INDEX_INDICATOR(a, h);
    int a_cInd = INDEX_INDICATOR(a, c);

    int b_wInd = INDEX_INDICATOR(b, w);
    int b_hInd = INDEX_INDICATOR(b, h);
    int b_cInd = INDEX_INDICATOR(b, c);
    
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q=0; q<channels; q++)
    {
        float* outptr = c.channel(q);

        for (int y=0; y<h; y++)
        {
            int h_step=y*w;
            for (int x=0; x<w; x++)
            {
                outptr[h_step + x] = op(a[q*a_cInd*a.cstep + y*a_hInd*a.w + x*a_wInd], b[q*b_cInd*b.cstep + y*b_hInd*b.w + x*b_wInd]);
            }
        }
    }

    return 0;
}

template<typename Op>
static int binary_op_scalar_inplace(Mat& a, float b, const Option& opt)
{
    Op op;

    int w = a.w;
    int h = a.h;
    int channels = a.c;
    int size = w * h;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q=0; q<channels; q++)
    {
        float* ptr = a.channel(q);

        for (int i=0; i<size; i++)
        {
            ptr[i] = op(ptr[i], b);
        }
    }

    return 0;
}

template<typename T>
struct binary_op_max : std::binary_function<T,T,T> {
    T operator() (const T& x, const T& y) const { return std::max(x, y); }
};

template<typename T>
struct binary_op_min : std::binary_function<T,T,T> {
    T operator() (const T& x, const T& y) const { return std::min(x, y); }
};

template<typename T>
struct binary_op_pow : std::binary_function<T,T,T> {
    T operator() (const T& x, const T& y) const { return pow(x, y); }
};

template<typename T>
struct binary_op_rsub : std::binary_function<T,T,T> {
    T operator() (const T& x, const T& y) const { return y - x; }
};

template<typename T>
struct binary_op_rdiv : std::binary_function<T,T,T> {
    T operator() (const T& x, const T& y) const { return y / x; }
};

int BinaryOp::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& bottom_blob1 = bottom_blobs[1];

    Mat& top_blob = top_blobs[0];

    if (op_type == Operation_ADD)
        return binary_op< std::plus<float> >(bottom_blob, bottom_blob1, top_blob, opt);

    if (op_type == Operation_SUB)
        return binary_op< std::minus<float> >(bottom_blob, bottom_blob1, top_blob, opt);

    if (op_type == Operation_MUL)
        return binary_op< std::multiplies<float> >(bottom_blob, bottom_blob1, top_blob, opt);

    if (op_type == Operation_DIV)
        return binary_op< std::divides<float> >(bottom_blob, bottom_blob1, top_blob, opt);

    if (op_type == Operation_MAX)
        return binary_op< binary_op_max<float> >(bottom_blob, bottom_blob1, top_blob, opt);

    if (op_type == Operation_MIN)
        return binary_op< binary_op_min<float> >(bottom_blob, bottom_blob1, top_blob, opt);

    if (op_type == Operation_POW)
        return binary_op< binary_op_pow<float> >(bottom_blob, bottom_blob1, top_blob, opt);

    if (op_type == Operation_RSUB)
        return binary_op< binary_op_rsub<float> >(bottom_blob, bottom_blob1, top_blob, opt);

    if (op_type == Operation_RDIV)
        return binary_op< binary_op_rdiv<float> >(bottom_blob, bottom_blob1, top_blob, opt);

    return 0;
}

int BinaryOp::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    if (op_type == Operation_ADD)
        return binary_op_scalar_inplace< std::plus<float> >(bottom_top_blob, b, opt);

    if (op_type == Operation_SUB)
        return binary_op_scalar_inplace< std::minus<float> >(bottom_top_blob, b, opt);

    if (op_type == Operation_MUL)
        return binary_op_scalar_inplace< std::multiplies<float> >(bottom_top_blob, b, opt);

    if (op_type == Operation_DIV)
        return binary_op_scalar_inplace< std::divides<float> >(bottom_top_blob, b, opt);

    if (op_type == Operation_MAX)
        return binary_op_scalar_inplace< binary_op_max<float> >(bottom_top_blob, b, opt);

    if (op_type == Operation_MIN)
        return binary_op_scalar_inplace< binary_op_min<float> >(bottom_top_blob, b, opt);

    if (op_type == Operation_POW)
        return binary_op_scalar_inplace< binary_op_pow<float> >(bottom_top_blob, b, opt);

    if (op_type == Operation_RSUB)
        return binary_op_scalar_inplace< binary_op_rsub<float> >(bottom_top_blob, b, opt);

    if (op_type == Operation_RDIV)
        return binary_op_scalar_inplace< binary_op_rdiv<float> >(bottom_top_blob, b, opt);

    return 0;
}

#if NCNN_VULKAN
int BinaryOp::create_pipeline()
{
    std::vector<vk_specialization_type> specializations(3);
    specializations[0].i = op_type;
    specializations[1].i = with_scalar;
    specializations[2].f = b;

    // pack1
    {
        pipeline_binaryop = new Pipeline(vkdev);
        pipeline_binaryop->set_optimal_local_size_xyz();
        pipeline_binaryop->create("binaryop", specializations, 3, 15);
    }

    // pack4
    {
        pipeline_binaryop_pack4 = new Pipeline(vkdev);
        pipeline_binaryop_pack4->set_optimal_local_size_xyz();
        pipeline_binaryop_pack4->create("binaryop_pack4", specializations, 3, 15);
    }

    return 0;
}

int BinaryOp::destroy_pipeline()
{
    delete pipeline_binaryop;
    pipeline_binaryop = 0;

    delete pipeline_binaryop_pack4;
    pipeline_binaryop_pack4 = 0;

    return 0;
}

int BinaryOp::forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    const VkMat& bottom_blob = bottom_blobs[0];
    const VkMat& bottom_blob1 = bottom_blobs[1];

    VkMat& top_blob = top_blobs[0];

    int packing = bottom_blob.packing;

    // TODO broadcast
    top_blob.create_like(bottom_blob, opt.blob_vkallocator, opt.staging_vkallocator);
    if (top_blob.empty())
        return -100;

//     fprintf(stderr, "BinaryOp::forward %p %p %p\n", bottom_blob.buffer(), bottom_blob1.buffer(), top_blob.buffer());

    std::vector<VkMat> bindings(3);
    bindings[0] = bottom_blob;
    bindings[1] = bottom_blob1;
    bindings[2] = top_blob;

    std::vector<vk_constant_type> constants(15);
    constants[0].i = bottom_blob.dims;
    constants[1].i = bottom_blob.w;
    constants[2].i = bottom_blob.h;
    constants[3].i = bottom_blob.c;
    constants[4].i = bottom_blob.cstep;
    constants[5].i = bottom_blob1.dims;
    constants[6].i = bottom_blob1.w;
    constants[7].i = bottom_blob1.h;
    constants[8].i = bottom_blob1.c;
    constants[9].i = bottom_blob1.cstep;
    constants[10].i = top_blob.dims;
    constants[11].i = top_blob.w;
    constants[12].i = top_blob.h;
    constants[13].i = top_blob.c;
    constants[14].i = top_blob.cstep;

    const Pipeline* pipeline = packing == 4 ? pipeline_binaryop_pack4 : pipeline_binaryop;

    // record
    cmd.record_pipeline(pipeline, bindings, constants, top_blob);

    return 0;
}

int BinaryOp::forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const
{
    int packing = bottom_top_blob.packing;
//     fprintf(stderr, "BinaryOp::forward_inplace %p\n", bottom_top_blob.buffer());

    std::vector<VkMat> bindings(3);
    bindings[0] = bottom_top_blob;
    bindings[1] = bottom_top_blob;// TODO use dummy buffer
    bindings[2] = bottom_top_blob;// TODO use dummy buffer

    std::vector<vk_constant_type> constants(15);
    constants[10].i = bottom_top_blob.dims;
    constants[11].i = bottom_top_blob.w;
    constants[12].i = bottom_top_blob.h;
    constants[13].i = bottom_top_blob.c;
    constants[14].i = bottom_top_blob.cstep;

    const Pipeline* pipeline = packing == 4 ? pipeline_binaryop_pack4 : pipeline_binaryop;

    // record
    cmd.record_pipeline(pipeline, bindings, constants, bottom_top_blob);

    return 0;
}
#endif // NCNN_VULKAN

} // namespace ncnn

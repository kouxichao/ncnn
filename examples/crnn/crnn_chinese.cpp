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
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <utility>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "net.h"

#include <wchar.h>
#include <locale.h>
#include <cwctype>

using namespace std;
static int recognition_chinese(const cv::Mat& bgr, ncnn::Mat& prop)
{
    ncnn::Net crnn_chi_net;

#ifdef ARM_HISI
    crnn_chi_net.load_param("chinese_crnn.param");
    crnn_chi_net.load_model("chinese_crnn.bin");
#else 
    crnn_chi_net.load_param("../../../models/crnn/chinese_crnn.param");
    crnn_chi_net.load_model("../../../models/crnn/chinese_crnn.bin");	
#endif
    ncnn::Mat in,data;

    in = ncnn::Mat::from_pixels(bgr.data, ncnn::Mat::PIXEL_GRAY , bgr.cols, bgr.rows);
    
//    printf("img:%d %d %d\n", in.w, in.h, in.c);

    ncnn::Mat l0_init_h = ncnn::Mat(100),
              l0_init_c = ncnn::Mat(100),
              l2_init_h = ncnn::Mat(100),
              l2_init_c = ncnn::Mat(100),
              l1_init_h = ncnn::Mat(100),
              l1_init_c = ncnn::Mat(100),
              l3_init_h = ncnn::Mat(100),
              l3_init_c = ncnn::Mat(100);
    l0_init_h.fill<float>(0.f);
    l0_init_c.fill<float>(0.f);
    l2_init_h.fill<float>(0.f);
    l2_init_c.fill<float>(0.f);
    l1_init_h.fill<float>(0.f);
    l1_init_c.fill<float>(0.f);
    l3_init_h.fill<float>(0.f);
    l3_init_c.fill<float>(0.f);

    ncnn::Extractor ex = crnn_chi_net.create_extractor();
    ex.set_light_mode(false);
    ex.input("data", in);
    ex.input("l0_init_h", l0_init_h);
    ex.input("l0_init_c", l0_init_c);
    ex.input("l2_init_h", l2_init_h);
    ex.input("l2_init_c", l2_init_c);
    ex.input("l1_init_h", l1_init_h);
    ex.input("l1_init_c", l1_init_c);
    ex.input("l3_init_h", l3_init_h);
    ex.input("l3_init_c", l3_init_c);
    ex.extract("fullyconnected0", prop);

    return 0;
}

static int recognition_chinese_lstm(const cv::Mat& bgr, ncnn::Mat& prop)
{
    ncnn::Net crnn_chi_net;

#ifdef ARM_HISI
    crnn_chi_net.load_param("chinese_crnn.param");
    crnn_chi_net.load_model("chinese_crnn.bin");
#else 
    crnn_chi_net.load_param("../../../models/crnn/chinese_crnn_lstm.param");
    crnn_chi_net.load_model("../../../models/crnn/chinese_crnn_lstm.bin");	
#endif
    ncnn::Mat in,data;

    in = ncnn::Mat::from_pixels(bgr.data, ncnn::Mat::PIXEL_GRAY , bgr.cols, bgr.rows);

    ncnn::Extractor ex = crnn_chi_net.create_extractor();
    ex.set_light_mode(false);
    ex.input("data", in);
    ex.extract("fullyconnected0", prop);
}

static int print_topk(ncnn::Mat& prop)
{
    setlocale(LC_ALL, "en_US.utf8");
    FILE* fp=fopen("../../../models/crnn/char_std_5990.txt","r");

    if(!fp) {
        perror("File opening failed");
        return EXIT_FAILURE;
    }
//    wchar_t result[35];
    int count=0,n = 0;
    wint_t wc;
    wchar_t classes[6000];
    while ((wc = fgetwc(fp)) != WEOF) {
        if(wc == 10)
           continue;
        count++;
        classes[count] = wc;
    }
//    printf("-----%d-----\n",count);
    float maxp; 
    std::vector<wchar_t> res;
    std::vector<int> p;
    p.resize(prop.h); 

    // partial sort topk with index
    for (int j=0; j<prop.h; j++)
    {
        p[j] = 0;
        maxp = *((float*)prop.row(j));

        for (int i=0; i<prop.w; i++)
        {

            if (*((float*)prop.row(j)+i) > maxp)
            {
                maxp = *((float*)prop.row(j)+i);
                p[j] = i;
            }
        }
//        fprintf(stderr, "p[j]:%d\n", p[j]);
        if(p[j] > 0 )
        {
            if(j == 0 || (j!=0 && p[j] != p[j-1]))
           {
//            fprintf(stderr, "value:%f__%dth__char:%lc\n---------------\n", maxp, p[j], classes[p[j]]);
//                result[n]=  classes[p[j]];
                res.push_back(classes[p[j]]);
                n++;
           }
        } 
    }

    res.push_back('\0');
    fprintf(stderr, "\nrecognition_result(%d): %ls\n", n, res.data());
    return 0;
}

int main(int argc, char** argv)
{
    const char* imagepath = argv[1];

    cv::Mat m = cv::imread(imagepath, 0);
    // cv::cvtColor(m,m,cv::COLOR_RGB2GRAY);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    ncnn::Mat prob_chinese, prob_english;
    recognition_chinese(m, prob_chinese);
    recognition_chinese_lstm(m, prob_english);

//    printf("out_shape(%d_%d):%d_%d %d_%d %d_%d\n",prob_chinese.dims, 
//    prob_english.dims, prob_chinese.w, prob_english.w, prob_chinese.h, prob_english.h, prob_chinese.c, prob_english.c);

    print_topk(prob_english);
    print_topk(prob_chinese);
    cv::imshow("Original image:",m);
    cv::waitKey();

    return 0;
}


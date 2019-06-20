#include "image_caption.h"
#include <stdio.h>
#include <algorithm>
#define RGB_PLANAR 1
#define CAPTION_MAX_SIZE 1000
#define RGB_PATH "image.data"
#define JPG_IMAGE 1
#define FROM_NNIE 1

#if JPG_IMAGE
#include "../print_data.h"
#include "dlib/image_io.h"
#include "dlib/image_processing.h"

int save_image_bin(const char* imagepath)
{
    /*-------------------------------------将图片转换为rgb平面二进制文件-------------------------------*/
    dlib::array2d<dlib::rgb_pixel> m, input_image(IMAGE_SIZE, IMAGE_SIZE);
    load_image(m, imagepath);
    FILE *stream = NULL; 
    stream = fopen(RGB_PATH, "wb");   

    dlib::resize_image(m, input_image);
    if(NULL == stream)
    {
        fprintf(stderr, "imgdata read error!");
        exit(1);
    } 

#ifndef RGB_META  //rgb_planar
    unsigned char* rgb_panel = new unsigned char[IMAGE_SIZE * IMAGE_SIZE *3];
    unsigned char* rgb_panel_r = rgb_panel;
    unsigned char* rgb_panel_g = rgb_panel + IMAGE_SIZE*IMAGE_SIZE;
    unsigned char* rgb_panel_b = rgb_panel + IMAGE_SIZE*IMAGE_SIZE*2;    

    for(int r = 0; r < IMAGE_SIZE; r++)
    {
	    for(int c = 0; c < IMAGE_SIZE; c++)
	    {
	        rgb_panel_r[r*IMAGE_SIZE+c] = input_image[r][c].red;
	        rgb_panel_g[r*IMAGE_SIZE+c] = input_image[r][c].green;
	        rgb_panel_b[r*IMAGE_SIZE+c] = input_image[r][c].blue;
            if(r == 0 && c < 100)
            printf("%d_%d\t", rgb_panel_r[r*IMAGE_SIZE+c], m[r][c].red);
	    }
    }
    printf("\n");
    fwrite(rgb_panel, 1, IMAGE_SIZE*IMAGE_SIZE*3, stream);
#else    //rgb_meta
    fwrite(&input_image[0][0], 1, IMAGE_SIZE*IMAGE_SIZE*3, stream);
#endif
    fclose(stream);
}
#endif

int main(int argc, char** argv)
{
    /*******************************LOAD ENCODER OUTPUT********************************/
    int encoder_outsize = IMAGE_SIZE >> 5;
    unsigned int encoder_out_elenum = encoder_outsize * encoder_outsize * ENCODER_DIM;

    #if FROM_NNIE//ENCODER                              //load encoder out from nnie output
    /**
     * 将图片转换为rgb平面二进制文件
     */
    //save_image_bin(argv[1]);
    ncnn::Mat encoder_mat(encoder_outsize, encoder_outsize, ENCODER_DIM);
    float* encoder_mat_data[1];
    encoder_mat_data[0] = encoder_mat.data;
    char* rgbPlanar_file = RGB_PATH;
    SAMPLE_SVP_NNIE_Encode(rgbPlanar_file, encoder_mat_data);
    
#ifdef DEBUG
    float max,total;
    max = *((float *)encoder_mat_data[0] + 0);
    total =  0.f;
    for(size_t i = 0; i < (encoder_outsize) * (encoder_outsize) * ENCODER_DIM; i++)
    {
        if(*((float *)encoder_mat_data[0] + i) > max)
            max = *((float *)encoder_mat_data[0] + i);
        total += *((float *)encoder_mat_data[0] + i);
    }
    printf("max: %f\t mean: %f\n", max, total/64/2048); //8.09,mean:0.33899
#endif

    for(size_t i = 0; i < (encoder_outsize) * (encoder_outsize) * ENCODER_DIM; i++)
    {
       *((float *)encoder_mat_data[0] + i) = *((float *)encoder_mat_data[0] + i) / 4096;
    }
/*
    print_data(encoder_mat, 0, 100, "nnie output\n");
    printf("nnie finished!\n");
*/
    #elif ENCODER_FROM_BIN                             //load encoder out from bin file
    ncnn::Mat encoder_mat(encoder_outsize, encoder_outsize, ENCODER_DIM);
    const char* encoder_data_file = argv[1];
    FILE *fp_encoder = fopen(encoder_data_file, "rb"); 
    if(NULL == fp_encoder)
    {
        fprintf(stderr, "encoder data open error!\n");
    }
    fread(encoder_mat, encoder_out_elenum*sizeof(float), 1, fp_encoder);
    fclose(fp_encoder);
    #else                                             //load encoder out from ncnn
    ncnn::Mat encoder_mat;
    #if OPENCV
    const char* imagepath = argv[1];
    cv::Mat m = cv::imread(imagepath);
    // cv::cvtColor(m,m,cv::COLOR_RGB2GRAY);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }
    #else
    const char* imagepath = argv[1];
    dlib::array2d<dlib::rgb_pixel> input_image;
    load_image(input_image, imagepath);
    int height = input_image.nr();
    int width = input_image.nc();

    #endif
    ncnn::Mat indata;
    indata = ncnn::Mat::from_pixels_resize((unsigned char*)(&input_image[0][0]), ncnn::Mat::PIXEL_RGB, \
                                            width, height, IMAGE_SIZE, IMAGE_SIZE);

    input_image.clear();
    printf("data_element_size:%d\n", indata.elemsize);
    print_data(indata, 0, 100, "indata_resize");

    for(size_t i = 0; i < IMAGE_SIZE * IMAGE_SIZE * 3; i++)
    {
        *((float *)indata.data + i) = *((float *)indata.data + i) / 255;
    }
    print_data(indata, 0, 100, "indata_scale");

    const float mean_vals[3] = {0.485, 0.456, 0.406};//123.675, 116.28; 103.53
    const float var_vals[3] = {1/0.229, 1/0.224, 1/0.225};//{1/0.229, 1/0.224, 1/0.225};

    indata.substract_mean_normalize(mean_vals, var_vals);
    print_data(indata, 0, 100, "indata_nor");

    ncnn::Net encoder;
    #ifdef ARM_HISI
    encoder.load_param("./imagecaption_encoder.param");
    encoder.load_model("./imagecaption_encoder.bin");
    #else
    encoder.load_param("../../../models/imagecaption/imagecaption_encoder.param");
    encoder.load_model("../../../models/imagecaption/imagecaption_encoder.bin");
    #endif    
    ncnn::Extractor ex_encoder = encoder.create_extractor();
    ex_encoder.set_light_mode(true);
    ex_encoder.input("data", indata);
    ex_encoder.extract("32_relu_bottleout", encoder_mat);
    print_data(encoder_mat, 0, 1000, "ncnn output\n");
    encoder.clear();
    printf("---------------------------------encoder_finish------------------------------------------\n");
    #endif //ENCODER 

    char* caption = (char *)calloc(CAPTION_MAX_SIZE, 1);
    //const float* encoder_out = NULL;
    imgcapdec_forward(encoder_mat, caption);
    fprintf(stderr, "final seq:\n%s\n", caption);
    free(caption);

    return 0;
}
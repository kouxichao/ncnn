#include <stdio.h>
#include "net.h"
#include "dlib/image_processing/generic_image.h"
#include "dlib/image_io.h"

int DKSceneClassification(char* rgbfilename, int iWidth, int iHeight)
{
#ifdef JPG_DEMO
    //使用图片路径进行测试（仅作测试时用,rgbfilename是jpg,png文件路径）
    dlib::array2d<dlib::rgb_pixel> img;
    load_image(img, rgbfilename);
#else
    FILE *stream = NULL; 
    stream = fopen(rgbfilename, "rb");   

    if(NULL == stream)
    {
        fprintf(stderr, "error:read imgdata!");
        exit(1);
    }

    unsigned char* rgbData = new unsigned char[iHeight*iWidth*3];
    fread(rgbData, 1, iHeight*iWidth*3, stream);
    fclose(stream); 

    dlib::array2d<dlib::rgb_pixel> img(iHeight, iWidth);
    //（unsigned char*）2（dlib::array2d<dlib::rgb_pixel>）  
    dlib::image_view<dlib::array2d<dlib::rgb_pixel>> imga(img);

// rgb_panel
    const unsigned char* channel_r = rgbData;
    const unsigned char* channel_g = rgbData + iHeight * iWidth;
    const unsigned char* channel_b = rgbData + iHeight * iWidth * 2 ;

    for(int r = 0; r < iHeight; r++) 
    {
        for(int c = 0; c < iWidth; c++)
        {
            dlib::rgb_pixel p;
            p.red = channel_r[r * iWidth + c];
            p.green = channel_g[r * iWidth + c];
            p.blue = channel_b[r * iWidth + c];
            assign_pixel( imga[r][c], p );            
        }
    }
    delete [] rgbData;
#endif  
    ncnn::Mat fc;
    ncnn::Net mobilefacenet;

#ifdef ARM_HISI
    mobilefacenet.load_param("book_scene_classify.param");//("../../../models/ncnn/book_scene_classify/book_scene_classify.param");
    mobilefacenet.load_model("book_scene_classify.bin");//("../../../models/ncnn/book_scene_classify/book_scene_classify.bin");
#else 
    mobilefacenet.load_param("../../../models/book_scene_classify/book_scene_classify.param");
    mobilefacenet.load_model("../../../models/book_scene_classify/book_scene_classify.bin");	
#endif
    ncnn::Mat in = ncnn::Mat::from_pixels_resize((unsigned char*)(&img[0][0]), ncnn::Mat::PIXEL_RGB2BGR, iWidth, iHeight, 384, 384);
//    const float mean_vals[3] = {104.f, 117.f, 123.f};//103.94,116.78,123.68
//    in.substract_mean_normalize(mean_vals, 0);
    ncnn::Extractor ex = mobilefacenet.create_extractor();
    ex.set_light_mode(false);

    ex.input("data", in);
    ex.extract("fc7_book", fc);
    printf("fc:%d,%d,%d\n%f_%f\n", fc.c, fc.h, fc.w, *(fc.row(0)), *(fc.row(0)+1));
    
    float *dp = fc.row(0);
    return *(dp+1) > *dp ? 1 : 0;
}

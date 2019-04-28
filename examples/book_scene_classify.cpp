#include <stdio.h>
#include "dlib/image_io.h"
#include "net.h"

int main(int argc, char* argv[])
{
    const char* imagepath = argv[1];
    dlib::array2d<dlib::rgb_pixel> rgbimage;
    dlib::load_image(rgbimage, imagepath);
    ncnn::Mat fc;
    ncnn::Net mobilefacenet;
    mobilefacenet.load_param("book_scene_classify.param");
    mobilefacenet.load_model("book_scene_classify.bin");

    ncnn::Mat in = ncnn::Mat::from_pixels_resize((unsigned char*)(&rgbimage[0][0]), ncnn::Mat::PIXEL_RGB2BGR, rgbimage.nc(), rgbimage.nr(), 384, 384);
//    const float mean_vals[3] = {104.f, 117.f, 123.f};//103.94,116.78,123.68
//    in.substract_mean_normalize(mean_vals, 0);
    ncnn::Extractor ex = mobilefacenet.create_extractor();
    ex.set_light_mode(false);

    ex.input("data", in);
    ex.extract("fc7_book", fc);
    printf("fc:%d,%d,%d\n%f_%f\n", fc.c, fc.h, fc.w, *(fc.row(0)), *(fc.row(0)+1));

/*
    if(*(fc.row(0)) >  *(fc.row(0)+1))
    {
	printf("text\n");
    }
    else
    	printf("others\n");
*/  
  return 0;
}



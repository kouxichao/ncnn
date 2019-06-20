#include "image_caption.h"
#include <stdio.h>
#include <algorithm>

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
    delete [] rgb_panel;
    fclose(stream);
}

int main(int argc, char** argv)
{
    save_image_bin(argv[1]);
    SAMPLE_SVP_NNIE_ImaCap_init();

    clock_t start = clock();
    for(size_t i = 0; i < 30; i++)
    {
        const char* image_path = RGB_PATH;
        char* caption = (char *)calloc(CAPTION_MAX_SIZE, 1);
        imgcap_forward(image_path, caption);
        fprintf(stderr, "final seq:\n%s\n", caption);
        free(caption);
    }
    
    clock_t finish = clock();
    printf("cost %u ms\n", (finish - start)/1000);

    SAMPLE_SVP_NNIE_ImaCap_End();
    return 0;
}
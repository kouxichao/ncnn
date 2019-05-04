#include <stdio.h>
#include "dlib/image_processing.h"
#include "dlib/image_io.h"
int DKSceneClassification(char* rgbfilename, int iWidth, int iHeight);
int main(int argc, char* argv[])
{
    /*-------------------------------------将图片转换为rgb平面二进制文件(测试使用)-----------------------------*/
    const char* imagepath = argv[1];
    dlib::array2d<dlib::rgb_pixel> m;
    load_image(m, imagepath);

    FILE *stream = NULL; 
    stream = fopen("image.rgb", "wb");   

    if(NULL == stream)
    {
        fprintf(stderr, "imgdata read error!");
        exit(1);
    } 
    
#ifndef RGB_META  //rgb_panel
    unsigned char* rgb_panel = new unsigned char[m.nc()*m.nr()*3];
    unsigned char* rgb_panel_r = rgb_panel;
    unsigned char* rgb_panel_g = rgb_panel + m.nc()*m.nr();
    unsigned char* rgb_panel_b = rgb_panel + m.nc()*m.nr()*2;    

    for(int r = 0; r < m.nr(); r++)
    {
	    for(int c = 0; c < m.nc(); c++)
	    {
	        rgb_panel_r[r*m.nc()+c] = m[r][c].red;
	        rgb_panel_g[r*m.nc()+c] = m[r][c].green;
	        rgb_panel_b[r*m.nc()+c] = m[r][c].blue;
	    }
    }
    fwrite(rgb_panel, 1, m.nc()*m.nr()*3, stream);
#else    //rgb_meta
    fwrite(&m[0][0], 1, m.nc()*m.nr()*3, stream);
#endif
    fclose(stream);
    /*------------------------------------获取分类结果：1 for book; 0 for notbook-----------------------------*/
    int book = DKSceneClassification("image.rgb", m.nc(), m.nr());
    printf("isbook:%d\n", book);

}

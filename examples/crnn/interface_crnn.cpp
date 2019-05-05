#include <stdio.h>
#include "net.h"
#include "text_recognization.h"
#include <omp.h>

ncnn::Net netCrnn;

void DKBoxTextRecognizationInit()
{
#ifdef ARM_HISI
    netCrnn.load_param("english_crnn.param");
    netCrnn.load_model("english_crnn.bin");
#else 
    netCrnn.load_param("../../../models/crnn/crnn.param");
    netCrnn.load_model("../../../models/crnn/crnn.bin");	
#endif
}

char* DKBoxTextRecognizationProcess(const char* rgbfilename, int iWidth, int iHeight, DKSBox box, DKSBoxTextRecognizationParam param)
{
    //裁剪文字区域
    FILE *stream = NULL; 
    stream = fopen(rgbfilename, "rb");   

    if(NULL == stream)
    {
        fprintf(stderr, "imgdata read error!");
        exit(1);
    } 
    
    unsigned char* rgbData = new unsigned char[iWidth*iHeight*3];
    fread(rgbData, 1, iHeight*iWidth*3, stream);
    fclose(stream); 
   
    int y_top = box.y1 > box.y2 ? box.y2 : box.y1;
    int y_bottom = box.y3 > box.y4 ? box.y3 : box.y4;
    int x_left = box.x1 > box.x4 ? box.x4 : box.x1;
    int x_right = box.x2 > box.x3 ? box.x2 : box.x3;
    y_top =  y_top > 0 ?  y_top : 0;
    x_left = x_left > 0 ? x_left : 0;
    y_bottom = y_bottom < iHeight ? y_bottom : iHeight; 
    x_right = x_right < iWidth ? x_right : iWidth;

    int cols = x_right - x_left;
    int rows = y_bottom - y_top;
    ncnn::Mat img;
    img.create(cols, rows, 3, size_t(1));
    int area = iWidth * iHeight;
#ifdef DEBUG 
    printf("%d,%d,%d,%d\n", y_top, x_left, y_bottom, x_right);
#endif    
#ifndef RGB_META
    #pragma omp parallel for     
//rgb_panel
    for(int i = y_top; i < y_bottom; i++)
    {
        for(int j = x_left; j < x_right; j++)
        {
            *((unsigned char*)(img.data)+3*(i-y_top)*cols+3*(j-x_left))   = rgbData[i*iWidth + j];
            *((unsigned char*)(img.data)+3*(i-y_top)*cols+3*(j-x_left)+1) = rgbData[area + i*iWidth + j];
            *((unsigned char*)(img.data)+3*(i-y_top)*cols+3*(j-x_left)+2) = rgbData[area*2 + i*iWidth + j];
        }
    }
#else
// rgb_metapixel

    for(int i = y_top; i < y_bottom + 1; i++)
    {
        for(int j = x_left; j < x_right + 1; j++)
        {
            *((unsigned char*)(img.data)+3*(i-y_top)*cols+3*(j-x_left))   = rgbData[i*iWidth*3 + 3 * j];
            *((unsigned char*)(img.data)+3*(i-y_top)*cols+3*(j-x_left)+1) = rgbData[i*iWidth*3 + 3*j+1];
            *((unsigned char*)(img.data)+3*(i-y_top)*cols+3*(j-x_left)+2) = rgbData[i*iWidth*3 + 3*j+2];
        }
    }
#endif
    delete  [] rgbData;
//    fclose(stream);
    //预处理并获取字符序列索引
    ncnn::Mat in,input_data;
    ncnn::Mat pred;
    in = ncnn::Mat::from_pixels((unsigned char*)img.data, ncnn::Mat::PIXEL_RGB2GRAY, cols, rows);
    ncnn::resize_bilinear(in,input_data,100,32);
    input_data.reshape(100,32,1);
    #pragma omp parallel for
    for(int i=0; i<100 * 32; i++)
    {
        *((float*)input_data.data+i) = ((*((float*)input_data.data+i))/255.f - 0.5)/0.5;
    }

    ncnn::Extractor ex = netCrnn.create_extractor();
//    ex.set_num_threads(2);
    ex.set_light_mode(true);
    ex.input("data", input_data);
    ex.extract("preds", pred);
    printf("%d\n", pred.dims);

    //对输出字符索引解码得到字符串。
    float maxprob; 
    int pre_index = 0;
    char alphabet[] = "0123456789abcdefghijklmnopqrstuvwxyz"; 
    char *result;
    if(NULL == (result = (char *)malloc(26 * sizeof(char))))
    /*请使用if来判断,这是有必要的*/
    {
        perror("error...");
        exit(1);
    }
//    static std::vector<char> result;
//    std::vector<char>::iterator iter = result.begin();
//    for(;iter!=result.end();)
//        iter = result.erase(iter);
    int ccount=0;
    for (int j=0; j<pred.h; j++)
    {
        int char_index = 0;
        maxprob = *((float*)pred.row(j));
//        printf("%d\n", maxprob);

        for (int i=0; i<pred.w; i++)
        {
            if (*((float*)pred.row(j)+i) > maxprob)
            {
                maxprob = *((float*)pred.row(j)+i);
                char_index = i;
            }
        }
        if(char_index != 0 && (j==0 || char_index != pre_index))
        {
            result[ccount] = alphabet[char_index-1];
            ccount++; 
//            result.push_back(alphabet[char_index-1]);            
        }      
        pre_index = char_index;
    }
    result[ccount] = '\0';       
//    result.push_back('\0');

    if(param.lexicon)
    {
//        return result;
        char * editre;
        if((editre = minDistanceWord(result)) != NULL)
            return editre;
        else
            return result;
    }
    else
    {
        return result;
//        return result.data();
    }

}

void DKBoxTextRecognizationEnd()
{
    netCrnn.clear();	
}

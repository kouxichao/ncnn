#include<stdio.h>
#include<unistd.h>
#include "dlib/image_io.h"
#include "dlib/image_processing/generic_image.h"
#include "face_recognization.h"
#include<cstring>

using namespace dlib;

static int convert2rgbfile(const char *imagepath, int *weight, int *height)
{
    dlib::array2d<dlib::rgb_pixel> m;
    load_image(m, imagepath);
/*    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }
*/
    FILE *stream = NULL; 
    stream = fopen("face.data", "wb");   

    if(NULL == stream)
    {
        fprintf(stderr, "imgdata read error!");
        exit(1);
    } 
    
//#ifndef RGB_META
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
//    unsigned char* rgbData = new unsigned char[m.cols*m.rows*3];
    fwrite(rgb_panel, 1, m.nc()*m.nr()*3, stream);
//#else
    
//    fwrite(&m[0][0], 1, m.nc()*m.nr()*3, stream);
//#endif
//    unsigned char* rgbData = new unsigned char[m.cols*m.rows*3];
    fclose(stream);

    *weight = m.nc();
    *height = m.nr();

    return 0;
}

int main(int argc, char* argv[])
{
    const char *root_dir = argv[2];
    
    char bbox_path[50];
    strcpy(bbox_path, root_dir);
    strcat(bbox_path, "bbox.xy"); 
    FILE *fp = fopen(bbox_path, "r+");
    if(NULL == fp)
    {
	    fprintf(stderr, "fopen bbox.xy error\n");
    }
    int  id;

    DKSSingleDetectionRes box[1];
    DKSMultiDetectionRes boxes;


    DKSFaceRegisterParam rgp;               //人脸注册的参数结构体           
    rgp.index = 0;                          //待注册的人脸框索引
    rgp.threshold = 0.4;                    //图片是否合格的阈值，越大对图片要求越严格
    rgp.flag = 0;                           //图片输入结束标志

    DKSFaceRecognizationParam rcp;          //人脸识别参数结构体
    rcp.index = 0;                          //待识别的人脸框索引
    rcp.threshold = 0.6;                    //识别阈值，越大要求越高
    rcp.k = 5;                              //knn最近邻中的k值

    char pre_name[50] = {};

    //注册
    if(*(argv[1]) == '0')
    {
        FILE* fp_idx_name = fopen("idx_name", "a+");
        if(NULL == fp_idx_name)
        {
	        fprintf(stderr, "fopen idx_name error\n");
        }
//	    std::vector<char*> idx_name;
        char name[50];
        char idx[5];
        int right,left,bottom,top;
        DKFaceRegisterInit();
        while(1)
        {

            if((fscanf(fp, "%s %s %d,%d,%d,%d", name, idx, &right, &left, &bottom, &top)) == EOF)
	        {
	            fprintf(stderr, "fscanf end(error)\n");
                break;
            }

            fprintf(stderr, "name : %s\n", name);
	        if(strstr(name, "test") == NULL)
            {
                std::string rgbfilename = std::string(root_dir) + std::string(name) + \
                 '/' + "support/" + name + "_" + idx;
                printf("PATH: %s\n", rgbfilename.data()); 
                if(access((rgbfilename + std::string(".jpg")).data(), 0) == 0)
                    rgbfilename = rgbfilename + std::string(".jpg");
                else
                    rgbfilename = rgbfilename + std::string(".png");

                box[0].box = {left,top,right,top,right,bottom,left,bottom};
//    box[0].box = {0,0,112,0,112,112,0,112};

                boxes.num = 1;
                boxes.boxes[0] = box[0];
                int w,h;
                static int picnum=0;
                int a = convert2rgbfile(rgbfilename.data(), &w, &h);
                int isqualified = -1;
                if(strcmp(pre_name, name) == 0)
                {
                    picnum++;
                    if(picnum < 6)
                    {  
                        if(picnum == 5)  
                            rgp.flag = 1; 

                        isqualified = DKFaceRegisterProcess("face.data", w, h, boxes, rgp);  //返回值，-1学习失败，0学习中，1学习成功！
                    }
                }
                else if(isqualified)
                {
                    picnum=0;
                //    fprintf(fp_idx_name, "%s\n", name);
                               //结束学习
                    printf("isqualified: %d\n", isqualified);
                  //  char * voicefile = "youyouyou";                                              //学习成功，输入相应语音文件;   
                    DKFaceRegisterEnd(name);                                                //存入数据库;
                    DKFaceRegisterInit();

                }
                
                strcpy(pre_name,  name);
            }
        }
        fclose(fp_idx_name);

    }
/*
    //识别
    if(*(argv[1]) == '1')
    {
        FILE* fp_idx_name = fopen("idx_name", "r+");
        if(NULL == fp_idx_name)
        {
	        fprintf(stderr, "fopen idx_name error\n");
        }
        char idx_name[20][30];
        int index = 0;
        while(fscanf(fp_idx_name, "%s", idx_name[index]) != EOF)
        {
//            fprintf(stderr, "temp_name : %s\n", idx_name[index]);
            index++;
        }
//        fprintf(stderr, "temp_name : %s\n", idx_name[0]);

        DKFaceRecognizationInit();
        char name[30];
        char idx[5];
        int right,left,bottom,top;
        float num = 0, acc = 0;
        while(1)
        {

            if((fscanf(fp, "%s %s %d,%d,%d,%d", name, idx, &right, &left, &bottom, &top)) == EOF)
	    {
//		fprintf(stderr, "fscanf end(error)\n");
                break;
            }

//            fprintf(stderr, "name : %s\n", name);
	    if(strstr(name, "test") != NULL)// && strcmp(name, "xiena_test") == 0)
            {
//                fprintf(stderr, "ori_pos:%d_%d_%d_%d\n", right, left, top, bottom);
//                 printf("name: %s\n", name); 
                std::size_t found = std::string(name).find_last_of("_");
                const std::string rea_name = std::string(name).substr(0, found);  
                std::string rgbfilename = std::string(root_dir) + rea_name + std::string("/") + std::string("test/") + std::string(name) + std::string("_") + std::string(idx);

std::string filename;
                if(access((rgbfilename + std::string(".jpg")).data(), 0) == 0)
                    filename = rgbfilename + std::string(".jpg");
                else
                    filename = rgbfilename + std::string(".png");
                //
                box[0].box = {left,top,right,top,right,bottom,left,bottom};
                boxes.num = 1;
                boxes.boxes[0] = box[0];

//                printf("PATH: %s\n", rgbfilename.data()); 
//                 printf("rename: %s\n", rea_name.data()); 
        	  id = DKFaceRecognizationProcess((char*)filename.data(), 100, 100, boxes, rcp);//示例中没有用到100,100两个参数。
        	    printf("image:%s \t gt_name:%s \t pre_name(ID):%s_(%d)\n", \
                (std::string(name) + "_" + idx).data(), rea_name.data(), idx_name[id], id);
                if((strcmp(rea_name.data(), idx_name[id])) == 0)
                    acc++;
                num++;
            }
	}

       	DKFaceRecognizationEnd();
        fclose(fp_idx_name);
        printf("num:%f \t acc:%f \n", num, acc);

        float accuracy = acc / num;
        fprintf(stderr, "accuracy: %.2f%%\n", accuracy * 100);
    }
*/
    fclose(fp);
    return 0;
}


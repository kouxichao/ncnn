#include<stdio.h>
#include "face_recognization.h"
#include "dlib/image_processing/frontal_face_detector.h"
#include "dlib/image_processing.h"
#include "dlib/image_io.h"

using namespace dlib;

int image2rgbbinary(const char *imagepath, rectangle &re, int *w, int *h)
{

/*-------------------------------------将图片转换为rgb平面二进制文件-------------------------------*/
    dlib::array2d<dlib::rgb_pixel> m;
    load_image(m, imagepath);
	*w=m.nc();*h=m.nr();
    FILE *stream = NULL; 
    stream = fopen("face.data", "wb");   

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

/*---------------------------------------------获取图片人脸坐标，仅供测试使用-------------------------------*/
    frontal_face_detector detector = get_frontal_face_detector();
    std::vector<rectangle> dets = detector(m);
    float index = 0, area = 0; 
    printf("dets_size:%d\n" , dets.size());
    if(!dets.size())
        return -1;
    for (unsigned long j = 0; j < dets.size();++j)    
    {
        float width = dets[j].right() - dets[j].left();
        float height = dets[j].bottom() - dets[j].top();
        if(width * height > area)
        {
            index = j;
            area = width * height;
        }
       // dlib::sleep(1000);     
	}

	re = dets[index];
	fprintf(stderr, "dets[]_%d\n",dets[index].right());
}

int main(int argc, char* argv[])
{
    DKFaceRegisterInit();                                                              //学习初始化             

    DKSFaceRegisterParam rgp;               //人脸注册的参数结构体           
    rgp.index = 0;                          //待注册的人脸框索引
    rgp.threshold = 0.4;                    //图片是否合格的阈值，越大对图片要求越严格
    rgp.flag = 0;                           //图片输入结束标志

    DKSFaceRecognizationParam rcp;          //人脸识别参数结构体
    rcp.index = 0;                          //待识别的人脸框索引
    rcp.threshold = 0.5;                    //识别阈值，越大要求越高
    rcp.k = 5;                              //knn最近邻中的k值


	rectangle re;
	int isqualified = -1;                                                              //学习合格变量
	int  wei,hei;
    //注册
    if(*(argv[1]) == '0')
    {
		for(int i=0; i< (*(argv[2]) - 48); i++)
		{
		    image2rgbbinary(argv[i+3], re, &wei, &hei);
            /*------------------------------------------初始化识别参数---------------------------------------*/
            //两点表示方框位置，方框的左上和右下坐标
		    typedef struct{
		        int xmax;
		        int xmin; 
		        int ymax;
		        int ymin; 
		    }Bbox;
		    Bbox bb={re.right(), re.left(), re.bottom() ,re.top()}; //chenweiting_test2

		    DKSSingleDetectionRes box[1];
    
		    box[0].box = {bb.xmin, bb.ymin, bb.xmax, bb.ymin, bb.xmax, bb.ymax, bb.xmin, bb.ymax}; 
    
//    box[0].box = {157, 141, 229, 141, 229, 213, 157, 213};//229,157,213,141 189,117,133,61
		    DKSMultiDetectionRes boxes;
		    boxes.num = 1;
		    boxes.boxes[0] = box[0];                //框坐标
        
            if(i == (*(argv[2]) - 49))
            rgp.flag = 1;                                                              //结束学习

		    isqualified = DKFaceRegisterProcess("face.data", wei, hei, boxes, rgp);  //返回值，-1学习失败，0学习中，1学习成功！
            printf("isqualified: %d\n", isqualified);
		}
        if(isqualified == 1)                    //学习成功
        {
            char * voicefile = argv[3];                                              //输入相应语音文件;   
            DKFaceRegisterEnd(voicefile);                                            //存入数据库;
        }
        else                                    //学习失败
        {
            char * voicefile = NULL;                                              //输入相应语音文件;   
            DKFaceRegisterEnd(voicefile);                                            //存入数据库;
            fprintf(stderr, "人脸学习失败！！！！！\n");
        }
        
    }

    //识别
    if(*(argv[1]) == '1')
    {
		image2rgbbinary(argv[2], re, &wei, &hei);
/*------------------------------------------初始化识别参数---------------------------------------*/
    //两点表示方框位置，方框的左上和右下坐标
		typedef struct{
		int xmax;
		int xmin; 
		int ymax;
		int ymin; 
		}Bbox;
		Bbox bb={re.right(), re.left(), re.bottom() ,re.top()}; //chenweiting_test2
		fprintf(stderr, "re_%d\n",re.right());

		DKSSingleDetectionRes box[1];
    
		box[0].box = {bb.xmin, bb.ymin, bb.xmax, bb.ymin, bb.xmax, bb.ymax, bb.xmin, bb.ymax}; 
    
//    box[0].box = {157, 141, 229, 141, 229, 213, 157, 213};//229,157,213,141 189,117,133,61
		DKSMultiDetectionRes boxes;
		boxes.num = 1;
		boxes.boxes[0] = box[0];                //框坐标
//        char*   rgbfilename = argv[2];
        DKFaceRecognizationInit();
//        for(int i = 0; i < 200; i++)
//        {
        char voicefilename[100];
        DKFaceRecognizationProcess(voicefilename, "face.data", wei, hei, boxes, rcp);
        printf("voicefile:%s\n", voicefilename);
//        }
        DKFaceRecognizationEnd();
    }
    return 0;
}

#include "dlib/image_processing/frontal_face_detector.h"
#include "dlib/image_processing.h"
#include "dlib/image_io.h"
#include <stdio.h>
//#include "dlib/dlib/gui_widgets.h"
#include <string>
#include <sys/stat.h>
#include <dirent.h>
using namespace dlib;


//获取path目录下的所有图片标签和特征保存在entry中
void getAllImages(const char* path)
{
    DIR* pDir;
    struct dirent* ptr;
    struct stat s;

    lstat(path, &s);
    pDir = opendir(path);
    
    frontal_face_detector detector = get_frontal_face_detector();
	array2d<rgb_pixel> img;

    if(pDir)
    {

      //  name = std::string(path).substr(found+1).data();

        while((ptr = readdir(pDir)) != 0)
        {
           if (ptr->d_type == DT_DIR)
           {
               if (strcmp(ptr -> d_name,".") != 0 && strcmp(ptr -> d_name,"..") != 0) 
               {
                   getAllImages((std::string(path) + "/" + ptr -> d_name).data());
               }
              
           }
           else if(ptr->d_type == DT_REG)
           {

                std::size_t name_end = std::string(ptr -> d_name).find_last_of("_");
                std::size_t idx_end = std::string(ptr -> d_name).find_last_of(".");                
                std::string name = std::string(ptr -> d_name).substr(0, name_end);
                std::string idx = std::string(ptr -> d_name).substr(name_end+1, idx_end - name_end - 1);

                printf("path:%s\t", name.data());
          //     entry.push_back(std::make_pair(std::string(path).substr(found+1), ptr -> d_name));
                char pa[256];
                strcpy(pa, path);
                strcat(pa, "/");    
                load_image(img, strcat(pa, ptr -> d_name));
//                image_window win(img); 
//	 		    pyramid_up(img);

         	    std::vector<rectangle> dets = detector(img);
                float index = 0, area = 0; 
                printf("dets_size:%d\n" , dets.size());
                if(!dets.size())
                    continue;

		        for (unsigned long j = 0; j < dets.size(); ++j)
		        {
                    float width = dets[j].right() - dets[j].left();
                    float height = dets[j].bottom() - dets[j].top();
                    if(width * height > area)
                    {
                        index = j;
                        area = width * height;
                    }
//                    win.add_overlay(dets[j]);
                    dlib::sleep(1000);     
		        }
                FILE *fp = fopen("../../../images/face_recognition/bbox.xy", "a+");

                if(NULL == fp)
                {
                    fprintf(stderr, "file open error");
                }
                fprintf(fp, "%s %s %ld,%ld,%ld,%ld\n", name.data(), idx.data(), dets[index].right(), dets[index].left(), dets[index].bottom(), dets[index].top());
                
                fclose(fp);
                std::cout << "image " << ptr -> d_name << " process finished" << std::endl;
           }   
        }
    }
}
int main(int argc, char const *argv[])
{
    const char *dir = argv[1];
    getAllImages(dir);
    return 0;
}

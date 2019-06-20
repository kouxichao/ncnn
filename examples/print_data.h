#include <stdio.h>

int print_data(ncnn::Mat &indata, int startrow, int num_print, char* name = NULL)
{
    printf("%s(dim:%d,c:%d,h:%d,w:%d):\n", name, indata.dims, indata.c,indata.h,indata.w);
    int rows = num_print/indata.w;
    int len;
    for(size_t i = 0; i < rows + 1; i++)
    {
        printf("row_%d:\n", i);
        if(i == rows)
            len = num_print % indata.w;
        else
            len = indata.w;
        
        for(size_t j = 0; j < len; j++)
        {
            /* code */
            printf("%d_%f\t", j, *((float*)indata.row(i+startrow) + j));
        }
        printf("\n");
    }
//    printf("\n");
}

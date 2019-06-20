#pragma once
//the size of input image
#ifndef IMAGE_SIZE
#define IMAGE_SIZE 384 
#endif

//max byte to save results
#ifndef CAPTION_MAX_SIZE
#define CAPTION_MAX_SIZE 1000
#endif

//image_data path
#ifndef RGB_PATH
#define RGB_PATH "image.data"
#endif

//encoder's output dimmention
#ifndef ENCODER_DIM
#define ENCODER_DIM 2048
#endif

//lstm h dimmention
#ifndef DECODER_DIM
#define DECODER_DIM 512
#endif

//attention dimmention
#ifndef ATTENTION_DIM
#define ATTENTION_DIM 512
#endif

//max caption length to generate(max decode times)
#ifndef MAX_CAPTION_LENGTH
#define MAX_CAPTION_LENGTH 50
#endif

//vocabulary size
#ifndef VOCABULARY_SIZE
#define VOCABULARY_SIZE 11676
#endif

//embedding dimmension
#ifndef EMBEDDING_DIM
#define EMBEDDING_DIM 512
#endif

//number sequences to consider at each decode-step
#ifndef BEAM_SIZE
#define BEAM_SIZE 5
#endif

//the size of decoder output image
#ifndef ENCODER_IMAGE_SIZE
#define ENCODER_IMAGE_SIZE 14
#endif

//number sequences to consider at each decode-step
#ifndef WEIGHT_BIN_FILEPATH
#ifdef ARM_HISI
#define WEIGHT_BIN_FILEPATH "./imagecaption_decoder.bin"
#else
#define WEIGHT_BIN_FILEPATH "../../../models/imagecaption/imagecaption_decoder.bin"
#endif
#endif

#include "net.h"

//nnie headers
#include "hi_common.h"
#include "hi_comm_sys.h"
#include "hi_comm_svp.h"
#include "sample_comm_nnie.h"

/**
 *@args:
 * tensor : The data to be processed. 
 * dim    : 0 for c, 1 for h, 2 for w
 * k      : The number of element to be selected.
 * dir    : 1 for largest k element, 0 for smallest k element.
 */
//TODO dim,dir argument
void topk(std::vector< std::pair<float, int> >& topK_re, ncnn::Mat& tensor, int k);

/**
 * function: image caption decoder forward.
 * @args:
 * caption: store final result 
 */
extern "C"
{
    void SAMPLE_SVP_NNIE_Encode(HI_CHAR *pcSrcFile, HI_FLOAT *aps32PermuteResult[]);
    void SAMPLE_SVP_NNIE_ImaCap_init();
    void SAMPLE_SVP_NNIE_ImaCap_End();
}

//int imgcap_forward(ncnn::Mat& inputData, char* caption);
int imgcap_forward(const char* image_path, char* caption);

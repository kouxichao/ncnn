#ifndef ARM_HISI
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif
#include <algorithm>
#include <stdio.h>
#include "net.h"
#include "../print_data.h"
#include "cJSON.h"

//the size of input image
#ifndef IMAGE_SIZE
#define IMAGE_SIZE 256
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
#ifndef BIN_FILEPATH
#ifdef ARM_HISI
#define BIN_FILEPATH "./imagecaption_decoder.bin"
#else
#define BIN_FILEPATH "../../../models/imagecaption/imagecaption_decoder.bin"
#endif
#endif

void topk(std::vector< std::pair<float, int> >& topK_re, ncnn::Mat& tensor, int k);
#include "layer_type.h"
#define ENCODER_FROM_BIN
int main(int argc, char** argv)
{

#ifdef ENCODER_FROM_BIN
    //get encoder output data 
    const char* encoder_data_file = argv[1];
    FILE *fp_encoder = fopen(encoder_data_file, "rb"); 
    if(NULL == fp_encoder)
    {
        fprintf(stderr, "encoder data open error!\n");
    }

    int decoder_outsize = IMAGE_SIZE/32;
    ncnn::Mat encoder_mat(decoder_outsize, decoder_outsize, ENCODER_DIM);
    unsigned int encoder_out_elenum = decoder_outsize * decoder_outsize * ENCODER_DIM;
    fread(encoder_mat, encoder_out_elenum*sizeof(float), 1, fp_encoder);

/*    
    unsigned int encoder_out_elenum = 8 * 8 * ENCODER_DIM ;
    float *encoder_out = (float *)std::calloc(encoder_out_elenum, sizeof(float));
    if(encoder_out)
    {
    for(size_t i = 0; i < 100; i++)
    {
        printf("%d_%f\t", i, *((float *)encoder_out + i));
    }
    }
    size_t encoder_out_size = fread(encoder_out, sizeof(float), encoder_out_elenum, fp_encoder);
    //encoder_out blob
    fprintf(stderr, "%d\n", encoder_out_size);
    ncnn::Mat encoder_mat(8, 8, ENCODER_DIM, encoder_out);
    fprintf(stderr, "en\n");
*/
#else
    const char* imagepath = argv[1];

    cv::Mat m = cv::imread(imagepath);
    // cv::cvtColor(m,m,cv::COLOR_RGB2GRAY);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }
    ncnn::Mat indata;

    indata = ncnn::Mat::from_pixels_resize(m.data, ncnn::Mat::PIXEL_BGR2RGB , m.cols, m.rows, IMAGE_SIZE, IMAGE_SIZE);
    //ncnn::resize_bilinear(in, indata, IMAGE_SIZE, IMAGE_SIZE);
    //print_data(in, 100);
//    print_data(indata, 100);
    printf("elemsize:%d\n", indata.elemsize);
    for(size_t i = 0; i < IMAGE_SIZE * IMAGE_SIZE * 3; i++)
    {
        *((float *)indata.data + i) = *((float *)indata.data + i) / 255;
    }
    printf("\n");
    const float mean_vals[3] = {0.485, 0.456, 0.406};
    const float var_vals[3] = {1/0.229, 1/0.224, 1/0.225};
    indata.substract_mean_normalize(mean_vals, var_vals);

    ncnn::Net encoder;
    ncnn::Mat encoder_mat;
    encoder.load_param("../../../models/imagecaption/imagecaption_encoder.param");
    encoder.load_model("../../../models/imagecaption/imagecaption_encoder.bin");

    ncnn::Extractor ex_encoder = encoder.create_extractor();
    ex_encoder.set_light_mode(false);
    ex_encoder.input("data", indata);
    ex_encoder.extract("32_relu_bottleout", encoder_mat);
    print_data(encoder_mat, 100);
#endif

    //weight data 
    const char* weight_bin_file = BIN_FILEPATH;
    FILE *fp_weight = fopen(weight_bin_file, "rb");
/*
    unsigned long weight_init_hc_elenum =  ENCODER_DIM*DECODER_DIM;
    float *weight_init_h = (float *)malloc(weight_init_hc_elenum * sizeof(float));
    float *weight_init_c = (float *)malloc(weight_init_hc_elenum * sizeof(float));
    size_t weight_init_h_size = fread(weight_init_h, sizeof(float), weight_init_hc_elenum, fp_weight);
    size_t weight_init_c_size = fread(weight_init_c, sizeof(float), weight_init_hc_elenum, fp_weight);
   
    //weight_init blob
    ncnn::Mat weight_init_h_mat(ENCODER_DIM, DECODER_DIM, weight_init_h);
    ncnn::Mat weight_init_c_mat(ENCODER_DIM, DECODER_DIM, weight_init_c);
*/

    //adaptiveAvePool 
    ncnn::Layer *adaptive_avepool = ncnn::create_layer(ncnn::LayerType::Pooling);
    ncnn::ParamDict pd;
    ncnn::Mat encoder_img,encoder_image,pm_encoder;
    pd.set(0, 2);
    pd.set(6, ENCODER_IMAGE_SIZE);
    pd.set(7, ENCODER_IMAGE_SIZE);
    adaptive_avepool->load_param(pd);
    adaptive_avepool->forward(encoder_mat, encoder_img, ncnn::get_default_option());
    delete adaptive_avepool;

    //pre_process encoder_out
    encoder_image = encoder_img.reshape(ENCODER_IMAGE_SIZE*ENCODER_IMAGE_SIZE, ENCODER_DIM);
    //get init_state input
    ncnn::Mat init_input(ENCODER_DIM);
    for(size_t h = 0; h < encoder_image.h; h++)
    {
        float sum = 0;
        int hp = h * encoder_image.w;
        for(size_t w = 0; w < encoder_image.w; w++)
        {
            sum += *((float*)encoder_image.data + hp + w);
        }
        *((float *)init_input.data + h) = sum / encoder_image.w;
    }

    ncnn::Layer *permute = ncnn::create_layer(ncnn::LayerType::Permute);
    pd.set(0,1);
    permute->load_param(pd);
    permute->forward(encoder_image, pm_encoder, ncnn::get_default_option());
    delete permute;
#if DEBUG
    print_data(pm_encoder, 0, 100, "encoder_out");
#endif
    //decoder_init_h and decoder_init_c blob
    ncnn::Layer *linear;
    ncnn::Mat init_h, init_c;
    ncnn::Mat prob, lstm_out_h, lstm_out_c;
    linear = ncnn::create_layer(ncnn::LayerType::FullyConnected);
    //set params 
    pd.set(0, DECODER_DIM);
    pd.set(1, 1);
    pd.set(2, ENCODER_DIM*DECODER_DIM);
    //init_h
    linear->load_param(pd);
    linear->load_model(ncnn::ModelBinFromStdio(fp_weight));
    linear->forward(init_input, init_h, ncnn::get_default_option());
    //init_c
    linear->load_model(ncnn::ModelBinFromStdio(fp_weight));
    linear->forward(init_input, init_c, ncnn::get_default_option());
    delete linear;
#ifdef  DEBUG   
    print_data(init_h, 0, 20, "init_h");
    print_data(init_c, 0, 20, "init_c");
#endif
    //decoder step
    ncnn::Net decoder;
    decoder.use_int8_inference = 0;

#ifdef ARM_HISI
    decoder.load_param("imagecaption_decoder.param");
    decoder.load_model(fp_weight);
#else 
    decoder.load_param("../../../models/imagecaption/imagecaption_decoder.param");
    decoder.load_model(fp_weight);	
#endif
/*  
    //load WORDMAP.json(word_index) and INDEXWORD.json(index_word)
    FILE *wordmp_fp = fopen("../../../models/imagecaption/WORDMAP.json", "r");
    if((NULL == wordmp_fp))
    {
        fprintf(stderr, "can't open WORDMAP.json!\n");
    }

    //caculate file length
    if(fseek(wordmp_fp, 0, SEEK_END) != 0)
        return -1;
    long len = ftell(wordmp_fp);
    char *wordmap = malloc(sizeof(char) * len);

    //read word_map
    fseek(wordmp_fp, 0, SEEK_SET);
    fread(wordmap, sizeof(char), len, wordmp_fp);

    //get <start> //IDEA set value directly, can simplify
    cJSON *jword = cJSON_Parse(wordmap);
    if(!jword)
    {
        fprintf(stderr, "parse json file failed!\n");
    } 
    cJSON *word = cJSON_GetObjectItem(jword, "<start>");
    int end_flag = cJSON_GetObjectItem(jword, "<end>")->valueint;
    fprintf(stderr, "%d_%d\n", word->valueint, end_flag);
    //k_prev_words 
    ncnn::Mat k_prev_words(BEAM_SIZE);
    k_prev_words.fill(word->valueint);

    cJSON_Delete(jword);
*/

    // set <start>,<end> index directly, don't look it up in WORDMAP.json.
    ncnn::Mat k_prev_words(BEAM_SIZE);
    k_prev_words.fill(11674);
    int end_flag = 11675;
    //top k sequences 
    std::vector<int> seqs[BEAM_SIZE];
    //topk word index and their scores.
    std::vector< std::pair<float, int> > topk_re(BEAM_SIZE, std::make_pair(0 ,0));
    //seqs index and word index. 
    int prev_word_inds[BEAM_SIZE], next_word_inds[BEAM_SIZE];
    std::vector<float> complete_seqs_scores;//[BEAM_SIZE]={FLT_MIN};

    //initialize for loop
    int step = 1, k = BEAM_SIZE;
    ncnn::Mat next_input_h(DECODER_DIM, k), next_input_c(DECODER_DIM, k);
    std::vector<int> complete_seqs[BEAM_SIZE];
    float top_k_scores[BEAM_SIZE];
    int complete_num = 0;

    //lstm for loop
    while(true){

        ncnn::Extractor ex = decoder.create_extractor();
        ex.set_light_mode(false);
#ifdef DEBUG
        printf("--------------------step--------------------- %d\n", step);
#endif
        if(step == 1)
        {
            ex.input("decoder_h", init_h);
            ex.input("decoder_c", init_c);
            ex.input("k_prev_words", ncnn::Mat(1, k_prev_words.data));
        }
        else if(step == 2)
        {
#ifdef DEBUG
            print_data(lstm_out_h, 0, 20, "lstm_pre_h");
            print_data(lstm_out_c, 0, 20, "lstm_pre_c");
#endif
            ex.input("decoder_h", lstm_out_h);
            ex.input("decoder_c", lstm_out_c);
            ex.input("k_prev_words", ncnn::Mat(k, k_prev_words.data));
        }
        else
        {
#ifdef DEBUG
        if(step == 3)
        {    
            print_data(lstm_out_h, 0, 20, "lstm_pre_h");
            print_data(lstm_out_c, 0, 20, "lstm_pre_c");
            print_data(lstm_out_h, 4, 20, "lstm_pre_h");
            print_data(lstm_out_c, 4, 20, "lstm_pre_c");
        }
#endif
            ex.input("decoder_h", ncnn::Mat(DECODER_DIM, k, lstm_out_h.data));
            ex.input("decoder_c", ncnn::Mat(DECODER_DIM, k, lstm_out_c.data));
            ex.input("k_prev_words", ncnn::Mat(k, k_prev_words.data));
        }
        ex.input("encoder_out", pm_encoder);

#ifdef DEBUG
        if(step == 3)
        {
        ncnn::Mat encoder_att, decoder_att, en_de_att,\
              alpha, attention_weighted_encoding, \
              awt_out, embeddings;
        ex.extract("encoder_att", encoder_att);
        ex.extract("decoder_att_reshape", decoder_att);
        ex.extract("full_att", en_de_att);
        ex.extract("alpha", alpha);
        ex.extract("encoder_out_alpha", attention_weighted_encoding);
        ex.extract("awt_out", awt_out);
        ex.extract("embeddings", embeddings);
        ex.extract("lstm_out_h", lstm_out_h);
        ex.extract("awt_gate", lstm_out_c);
        print_data(encoder_att, 0, 20, "encoder_att");
        print_data(decoder_att, 0, 20, "decoder_att");
        print_data(decoder_att, 4, 20, "decoder_att");
        print_data(en_de_att, 0, 20, "en_de_att");
        print_data(en_de_att, 784, 20, "en_de_att");
        print_data(alpha, 0, 20, "alpha");
        print_data(attention_weighted_encoding, 0, 20, "attention_weighted_encoding");
        print_data(attention_weighted_encoding, 78, 20, "attention_weighted_encoding");
        print_data(awt_out, 4, 20, "awt_out");
        print_data(embeddings, 0, 20, "embeddings");
        print_data(lstm_out_h, 4, 20, "lstm_out_h");
        print_data(lstm_out_c, 4, 20, "lstm_out_c");
        }
#endif
        //get output data lstm_out_c,lstm_out_h,prob(scores).
        //ex.extract("lstm_stepc", lstm_out_c);
        ex.extract("lstm_out_h", lstm_out_h);
        ex.extract("lstm_out_c", lstm_out_c);
        ex.extract("prob", prob);

#ifdef DEBUG
        if(step == 2)
        {ncnn::Mat ttt = ncnn::Mat(VOCABULARY_SIZE, prob.row(2));
        print_data(ttt, 1, 100, "prob");   }
#endif
        //Add pre_prob to prob
        if(step != 1)
        {
            for(size_t i = 0; i < prob.h; i++)
            {
                float *out_sam = prob.row(i);
                for(size_t j = 0; j < prob.w; j++)
                {
                    out_sam[j] = out_sam[j] + top_k_scores[i];//topk_re[i].first; 
                }
            }
        }
#ifdef DEBUG 
        if(step == 2)
        {ncnn::Mat ttt = ncnn::Mat(VOCABULARY_SIZE, prob.row(2));
        print_data(ttt, 1, 100, "add_prob");   }
#endif
        //get topk score_indexes and save them in topk_re
        topk(topk_re, prob, k);

        //get input data for next step(lstm_out_h, lstm_out_c, k_prev_words), and save predict words to seqs.
        int incomplete_num = 0; std::vector<int> temp_seq[k], incomplete_inds;
//        ncnn::Mat next_input_h(DECODER_DIM, k), next_input_c(DECODER_DIM, k);
        for(size_t i = 0; i < k; i++)
        {
            prev_word_inds[i] = topk_re[i].second / VOCABULARY_SIZE;
            next_word_inds[i] = topk_re[i].second % VOCABULARY_SIZE;
#ifdef DEBUG
            printf("pre_word_inds:%d\n", prev_word_inds[i]);
            printf("next_word_inds:%d\n", next_word_inds[i]);
#endif
            temp_seq[i] = seqs[prev_word_inds[i]];
            temp_seq[i].push_back(next_word_inds[i]);
            if (next_word_inds[i] != end_flag) {
                incomplete_inds.push_back(i);
                if(step != 1) //&& prev_word_inds[i] != incomplete_num) 
                {
                    memcpy(next_input_h.row(incomplete_num), lstm_out_h.row(prev_word_inds[i]), sizeof(float) * DECODER_DIM);
                    memcpy(next_input_c.row(incomplete_num), lstm_out_c.row(prev_word_inds[i]), sizeof(float) * DECODER_DIM);
                }
                *((int*)k_prev_words.row(0) + incomplete_num) =  next_word_inds[i];
                top_k_scores[incomplete_num] = topk_re[i].first;
                incomplete_num++; 
                //topk_re[incomplete_num].first = topk_re[i].first;
            }
            else
            {
                complete_seqs[complete_num] = temp_seq[i];
                complete_seqs_scores.push_back(topk_re[i].first);
                complete_num++;
            }
            
#ifdef DEBUG
            printf("wordIndex_value:%d_%f\n", next_word_inds[i], topk_re[i].first);
#endif
        }
        k = incomplete_num;
        
        for(size_t i = 0; i < incomplete_inds.size(); i++)
        {
            seqs[i] = temp_seq[incomplete_inds[i]];
        }
        
        if(step != 1)
        {
            lstm_out_h.release();
            lstm_out_c.release();
            lstm_out_h.create(DECODER_DIM, k);
            lstm_out_c.create(DECODER_DIM, k);
            memcpy(lstm_out_h.data, next_input_h.data, incomplete_num * sizeof(float) * DECODER_DIM);
            memcpy(lstm_out_c.data, next_input_c.data, incomplete_num * sizeof(float) * DECODER_DIM);
        }
#ifdef DEBUG
        printf("next step k:%d\n", k);  
#endif        
        if(k == 0)
        {
            break;
        }
        if(step > 50)
            break;
        step++;
    
    }

    float max_score = -__FLT_MAX__;
    int final_seq = -1;
    for(size_t i = 0; i < BEAM_SIZE; i++)
    {
        float s = complete_seqs_scores[i];
#ifdef DEBUG
        printf("complete_seqs_scores:%d_%f\n", i, complete_seqs_scores[i]);
#endif
        if(s > max_score)
        {
            max_score = s;
            final_seq = i;
        }
    }
#ifdef DEBUG
    fprintf(stderr, "final_seq(%d):\n", final_seq);

    for(int s=0; s<BEAM_SIZE; s++)
    {
        final_seq = s;
#endif
#ifdef ARM_HISI
    FILE *indexwd_fp = fopen("INDEXWORD.json", "r");
#else
    FILE *indexwd_fp = fopen("../../../models/imagecaption/INDEXWORD.json", "r");
#endif
    if((NULL == indexwd_fp))
    {
        fprintf(stderr, "can't open INDEXWORD.json!\n");
    }
    if(fseek(indexwd_fp, 0, SEEK_END) != 0)
        return -1;
    long len1 = ftell(indexwd_fp);
    char *indexword = malloc(sizeof(char) * len1);
    fseek(indexwd_fp, 0, SEEK_SET);
    fread(indexword, sizeof(char), len1, indexwd_fp);
    
    cJSON *jword = cJSON_Parse(indexword);
    if(!jword)
    {
        fprintf(stderr, "parse INDEXWORD.json file failed!\n");
    } 
    
    char ind[20];
    for(int i = 0; i < complete_seqs[final_seq].size(); i++)
    {
        int index = complete_seqs[final_seq].data()[i];
        sprintf(ind, "%d", index);
    //    printf("%d\n", index);
        char* word = cJSON_GetObjectItem(jword, ind)->valuestring;
        if(strcmp(word, "<end>") == 0)
            break;
        printf("%s ", word);
    }
    cJSON_Delete(jword);

    printf("\n");
#ifdef DEBUG
    }
#endif

    return 0;
}


/**
 *@args:
 * tensor : The data to be processed. 
 * dim    : 0 for c, 1 for h, 2 for w.
 * k      : The number of element to be selected.
 * dir    : 1 for largest k element, 0 for smallest k element.
 */
//TODO dim,dir argument
void topk(std::vector< std::pair<float, int> >& topk_re, ncnn::Mat& tensor, int k)
{
    std::vector< std::pair<float, int> > score_index;
    score_index.resize(tensor.total());
    for (int i=0; i<tensor.total(); i++)
    {
        score_index[i] = std::make_pair(tensor[i], i);
    }

    std::partial_sort(score_index.begin(), score_index.begin() + k, score_index.end(), std::greater< std::pair<float, int> >());

    for(size_t i = 0; i < k; i++)
    {
        topk_re[i] = score_index[i];
    }

}
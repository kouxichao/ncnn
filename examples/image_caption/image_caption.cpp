#include "image_caption.h"
#include <algorithm>
#include <stdio.h>
#include "net.h"
#include "cJSON.h"
#include "layer_type.h"

#if NCNN_BENCHMARK
#include "benchmark.h"
#endif // NCNN_BENCHMARK

int imgcap_forward(const char* image_path, char* caption)
{
    /*******************************LOAD ENCODER OUTPUT********************************/
    int encoder_outsize = IMAGE_SIZE >> 5;
    unsigned int encoder_out_elenum = encoder_outsize * encoder_outsize * ENCODER_DIM;

    ncnn::Mat encoder_mat(encoder_outsize, encoder_outsize, ENCODER_DIM);
    float* encoder_mat_data[1];
    encoder_mat_data[0] = encoder_mat.data;
    char* rgbPlanar_file = image_path;
    SAMPLE_SVP_NNIE_Encode(rgbPlanar_file, encoder_mat_data);
    
    for(size_t i = 0; i < (encoder_outsize) * (encoder_outsize) * ENCODER_DIM; i++)
    {
       *((float *)encoder_mat_data[0] + i) = *((float *)encoder_mat_data[0] + i) / 4096;
    }

    /********************************ENCODER OUTPUT PRE_PROCESS******************************/
    /*adaptiveAvePool*/ 
    ncnn::Layer *adaptive_avepool = ncnn::create_layer(ncnn::LayerType::Pooling);
    ncnn::ParamDict pd;
    ncnn::Mat encoder_img,encoder_image,pm_encoder;
    pd.set(0, 2);
    pd.set(6, ENCODER_IMAGE_SIZE);
    pd.set(7, ENCODER_IMAGE_SIZE);
    adaptive_avepool->load_param(pd);
    adaptive_avepool->forward(encoder_mat, encoder_img, ncnn::get_default_option());
    delete adaptive_avepool;

    /**
     * encoder_img(ENCODER_IMAGE_SIZE, ENCODER_IMAGE_SIZE, 
     * ENCODER_DIM)------->encoder_image------>pm_encoder(ENCODER_DIM,
     * ENCODER_IMAGE_SIZE*ENCODER_IMAGE_SIZE)
     */
    encoder_image = encoder_img.reshape(ENCODER_IMAGE_SIZE*ENCODER_IMAGE_SIZE, ENCODER_DIM);
    ncnn::Layer *permute = ncnn::create_layer(ncnn::LayerType::Permute);
    pd.set(0,1);
    permute->load_param(pd);
    permute->forward(encoder_image, pm_encoder, ncnn::get_default_option());
    delete permute;

    /*init_hc input data*/
    ncnn::Mat init_input(ENCODER_DIM);
    #pragma omp parallel for
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

    /*weight data file pointer*/
    const char* weight_bin_file = WEIGHT_BIN_FILEPATH;
    FILE *fp_weight = fopen(weight_bin_file, "rb");
    if(NULL == fp_weight)
    {
        fprintf(stderr, "read %s failed!", weight_bin_file);
    }
    /**
     * decoder first step input data: init_h, init_c.
     * decoder loop input data: encoder_att.
     */
    ncnn::Layer *linear;
    ncnn::Mat init_h, init_c, encoder_att;
    linear = ncnn::create_layer(ncnn::LayerType::FullyConnected);
    //set params for init_h,int_c
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
    //encoder_att
    pd.set(0, ATTENTION_DIM);
    pd.set(1, 1);
    pd.set(2, ATTENTION_DIM*ENCODER_DIM);
    linear->load_param(pd);
    linear->load_model(ncnn::ModelBinFromStdio(fp_weight));
    #if NCNN_BENCHMARK
    double start = ncnn::get_current_time();
    int ret = linear->forward(pm_encoder, encoder_att, ncnn::get_default_option());
    double end = ncnn::get_current_time();
    benchmark(linear, pm_encoder, encoder_att, start, end);
    #else
    linear->forward(pm_encoder, encoder_att, ncnn::get_default_option());
    #endif
    delete linear;

    /*******************************DECODER START**************************/
    ncnn::Net decoder;
    decoder.use_int8_inference = 0;
    
    #ifdef ARM_HISI
    decoder.load_param("imagecaption_decoder.param");
    decoder.load_model(fp_weight);
    #else 
    decoder.load_param("../../../models/imagecaption/imagecaption_decoder.param");
    decoder.load_model(fp_weight);	
    #endif

    /**
     * initialize variables for decoder loop
     */
    
    ncnn::Mat k_prev_words(BEAM_SIZE); //set <start>, <end> index directly, don't look it up in WORDMAP.json.
    k_prev_words.fill(11674);   //init k_prev_words with <start> index 11674        
    int end_flag = 11675;       //<end> index: 11675         
    std::vector< std::pair<float, int> > topk_re(BEAM_SIZE, std::make_pair(0 ,0));  //topk word index and their scores.
    int prev_word_inds[BEAM_SIZE], next_word_inds[BEAM_SIZE]; //current step seqs index of  scores and next decoder step word index.
    std::vector<int> seqs[BEAM_SIZE]; //top k sequences(captions) 
    std::vector<int> complete_seqs[BEAM_SIZE];  //completed sequences 
    std::vector<float> complete_seqs_scores;   //scores of completed sequences
    int step = 1;   //step indicator
    int k = BEAM_SIZE;   //number of remaining sequences            
    ncnn::Mat next_input_h(DECODER_DIM, k), next_input_c(DECODER_DIM, k), prob, lstm_out_h, lstm_out_c;
    float top_k_scores[BEAM_SIZE];    //top_k scores at every decode step 
    int complete_num = 0;             //number of completed sequences 

    /*decoder loop*/
    while(true)
    {
        ncnn::Extractor ex = decoder.create_extractor();
        ex.set_light_mode(false);

        #if DEBUG
        printf("--------------------step--------------------- %d\n", step);
        #endif

        if(step == 1)         //for step 1, input the init_hc, 1 pre_word <start>. because BEAM_SIZE samples' results are same.
        {
            ex.input("decoder_h", init_h);
            ex.input("decoder_c", init_c);
            ex.input("k_prev_words", ncnn::Mat(1, k_prev_words.data));
        }
        else if(step == 2)   //for step 2, directly give the step 1 output to step 2, the pre_words are the output of step 1.
        {
            ex.input("decoder_h", lstm_out_h);
            ex.input("decoder_c", lstm_out_c);
            ex.input("k_prev_words", ncnn::Mat(k, k_prev_words.data));
        }
        else                 //for stpe >= 3, k pre lstm out, k pre words
        {
            ex.input("decoder_h", ncnn::Mat(DECODER_DIM, k, lstm_out_h.data));
            ex.input("decoder_c", ncnn::Mat(DECODER_DIM, k, lstm_out_c.data));
            ex.input("k_prev_words", ncnn::Mat(k, k_prev_words.data));
        }
        ex.input("encoder_out", pm_encoder);     //input encoder's output
        ex.input("encoder_att", encoder_att);    //input encoder's attention output 

        /*get output data lstm_out_c, lstm_out_h, prob(scores)*/
        ex.extract("lstm_out_h", lstm_out_h);
        ex.extract("lstm_out_c", lstm_out_c);
        ex.extract("prob", prob);

        /*Add pre_prob to prob*/
        if(step != 1)
        {
            #pragma omp parallel for num_threads(4)
            for(size_t i = 0; i < prob.h; i++)
            {
                float *out_sam = prob.row(i);
                for(size_t j = 0; j < prob.w; j++)
                {
                    out_sam[j] = out_sam[j] + top_k_scores[i];
                }
            }
        }

        //get topk scores and indexes, and save them in topk_re
        topk(topk_re, prob, k);

        //get input data for next step(lstm_out_h, lstm_out_c, k_prev_words), and append predict words to seqs.
        int incomplete_num = 0; std::vector<int> temp_seq[k], incomplete_inds;
        for(size_t i = 0; i < k; i++)
        {
            prev_word_inds[i] = topk_re[i].second / VOCABULARY_SIZE;
            next_word_inds[i] = topk_re[i].second % VOCABULARY_SIZE;
            
            #if DEBUG
            printf("pre_word_inds:%d\n", prev_word_inds[i]);
            printf("next_word_inds:%d\n", next_word_inds[i]);
            #endif
            
            temp_seq[i] = seqs[prev_word_inds[i]];
            temp_seq[i].push_back(next_word_inds[i]);
            if (next_word_inds[i] != end_flag) {
                incomplete_inds.push_back(i);
                if(step != 1)
                {
                    memcpy(next_input_h.row(incomplete_num), lstm_out_h.row(prev_word_inds[i]), sizeof(float) * DECODER_DIM);
                    memcpy(next_input_c.row(incomplete_num), lstm_out_c.row(prev_word_inds[i]), sizeof(float) * DECODER_DIM);
                }
                *((int*)k_prev_words.row(0) + incomplete_num) =  next_word_inds[i];
                top_k_scores[incomplete_num] = topk_re[i].first;
                incomplete_num++; 
            }
            else
            {
                complete_seqs[complete_num] = temp_seq[i];
                complete_seqs_scores.push_back(topk_re[i].first);
                complete_num++;
            }
            
            #if DEBUG
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

        #if DEBUG
        printf("next step number of sequences:%d\n", k);  
        #endif   

        if(k == 0)
        {
            break;
        }
        if(step > MAX_CAPTION_LENGTH)
            break;
        step++;
    }
    fclose(fp_weight);

    /*********************************GET FINAL SEQUENCE*****************************/
    if(complete_seqs_scores.empty()) 
    {
        fprintf(stderr, "Error: Sequence length is greater than the maximum length %d\n", MAX_CAPTION_LENGTH);
        strcpy(caption, "<end>");
        return -1;
    }

    int final_seq = -1;               //final sequence index.
    float max_score = -__FLT_MAX__;   //score of final sequence
    for(size_t i = 0; i < BEAM_SIZE; i++)
    {
        #if DEBUG
        printf("complete_seqs_scores:%d_%f\n", i, complete_seqs_scores[i]);
        #endif
    
        float s = complete_seqs_scores[i];
        if(s > max_score)
        {
            max_score = s;
            final_seq = i;
        }
    }

    /*decode final sequence*/
    //open and read i2w json file.
    #if ARM_HISI
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
    char *indexword = (char*)malloc(sizeof(char) * len1);
    fseek(indexwd_fp, 0, SEEK_SET);
    fread(indexword, sizeof(char), len1, indexwd_fp);
    
    //parse json data, get final result
    cJSON *jword = cJSON_Parse(indexword);
    if(!jword)
    {
        fprintf(stderr, "parse INDEXWORD.json file failed!\n");
    } 

    int captionlen=0;
    char ind[50];
    #if DEBUG
    for(int s=0; s<BEAM_SIZE; s++)
    {
        final_seq = s;
    #endif

    for(int i = 0; i < complete_seqs[final_seq].size(); i++)
    {
        int index = complete_seqs[final_seq].data()[i];
        sprintf(ind, "%d", index); //index 2 string, because json key is string.
        char* word = cJSON_GetObjectItem(jword, ind)->valuestring;
        if(strcmp(word, "<end>") == 0)
            break;
        #if DEBUG
        printf("word:%s\t", word);
        #endif
        strcat(caption, word);
        captionlen = strlen(caption);
        caption[captionlen] = ' ';
    }
    #if DEBUG
    printf("\n");
    caption[captionlen] = '\n';
    }
    #endif

    caption[captionlen] = '\0';
    free(indexword);
    cJSON_Delete(jword);
    fclose(indexwd_fp);

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
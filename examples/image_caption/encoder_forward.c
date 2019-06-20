#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <signal.h>
#include <pthread.h>
#include <sys/prctl.h>
#include <math.h>

#include "hi_common.h"
#include "hi_comm_sys.h"
#include "hi_comm_svp.h"
#include "sample_comm.h"
#include "sample_comm_svp.h"
#include "sample_comm_nnie.h"
#include "sample_nnie_main.h"
#include "sample_svp_nnie_software.h"
#include "sample_comm_ive.h"



/*ssd para*/
static SAMPLE_SVP_NNIE_MODEL_S s_stImgCapModel = {0};
static SAMPLE_SVP_NNIE_PARAM_S s_stImgCapNnieParam = {0};
static SAMPLE_SVP_NNIE_SSD_SOFTWARE_PARAM_S s_stImgCapSoftwareParam = {0};

/******************************************************************************
* function : NNIE Forward
******************************************************************************/
static HI_S32 IMGCAP_SVP_NNIE_Forward(SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
    SAMPLE_SVP_NNIE_INPUT_DATA_INDEX_S* pstInputDataIdx,
    SAMPLE_SVP_NNIE_PROCESS_SEG_INDEX_S* pstProcSegIdx,HI_BOOL bInstant)
{
    HI_S32 s32Ret = HI_SUCCESS;
    HI_U32 i = 0, j = 0;
    HI_BOOL bFinish = HI_FALSE;
    SVP_NNIE_HANDLE hSvpNnieHandle = 0;
    HI_U32 u32TotalStepNum = 0;

    SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx].stTskBuf.u64PhyAddr,
        (HI_VOID *) pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx].stTskBuf.u64VirAddr,
        pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx].stTskBuf.u32Size);

    /*set input blob according to node name*/
    if(pstInputDataIdx->u32SegIdx != pstProcSegIdx->u32SegIdx)
    {
        for(i = 0; i < pstNnieParam->pstModel->astSeg[pstProcSegIdx->u32SegIdx].u16SrcNum; i++)
        {
            for(j = 0; j < pstNnieParam->pstModel->astSeg[pstInputDataIdx->u32SegIdx].u16DstNum; j++)
            {
                if(0 == strncmp(pstNnieParam->pstModel->astSeg[pstInputDataIdx->u32SegIdx].astDstNode[j].szName,
                    pstNnieParam->pstModel->astSeg[pstProcSegIdx->u32SegIdx].astSrcNode[i].szName,
                    SVP_NNIE_NODE_NAME_LEN))
                {
                    pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astSrc[i] =
                        pstNnieParam->astSegData[pstInputDataIdx->u32SegIdx].astDst[j];
                    break;
                }
            }
            SAMPLE_SVP_CHECK_EXPR_RET((j == pstNnieParam->pstModel->astSeg[pstInputDataIdx->u32SegIdx].u16DstNum),
                HI_FAILURE,SAMPLE_SVP_ERR_LEVEL_ERROR,"Error,can't find %d-th seg's %d-th src blob!\n",
                pstProcSegIdx->u32SegIdx,i);
        }
    }

    /*NNIE_Forward*/
    s32Ret = HI_MPI_SVP_NNIE_Forward(&hSvpNnieHandle,
        pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astSrc,
        pstNnieParam->pstModel, pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst,
        &pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx], bInstant);
    SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret,s32Ret,SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,HI_MPI_SVP_NNIE_Forward failed!\n");


    if(bInstant)
    {
        /*Wait NNIE finish*/
        while(HI_ERR_SVP_NNIE_QUERY_TIMEOUT == (s32Ret = HI_MPI_SVP_NNIE_Query(pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx].enNnieId,
            hSvpNnieHandle, &bFinish, HI_TRUE)))
        {
            usleep(100);
            SAMPLE_SVP_TRACE(SAMPLE_SVP_ERR_LEVEL_INFO,
                "HI_MPI_SVP_NNIE_Query Query timeout!\n");
        }
    } 
    bFinish = HI_FALSE;
    for(i = 0; i < pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx].u32DstNum; i++)
    {
        if(SVP_BLOB_TYPE_SEQ_S32 == pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].enType)
        {
            for(j = 0; j < pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Num; j++)
            {
                u32TotalStepNum += *((HI_U32*)(pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stSeq.u64VirAddrStep)+j);
            }
            SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64PhyAddr,
                (HI_VOID *) pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64VirAddr,
                u32TotalStepNum*pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Stride);

        }
        else
        {

            SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64PhyAddr,
                (HI_VOID *) pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64VirAddr,
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Num*
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stWhc.u32Chn*
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stWhc.u32Height*
                pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Stride);
        }
    }

    return s32Ret;
}

/******************************************************************************
* function : Fill Src Data
******************************************************************************/
static HI_S32 IMGCAP_SVP_NNIE_FillSrcData(SAMPLE_SVP_NNIE_CFG_S* pstNnieCfg,
    SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam, SAMPLE_SVP_NNIE_INPUT_DATA_INDEX_S* pstInputDataIdx)
{
    FILE* fp = NULL;
    HI_U32 i =0, j = 0, n = 0,m = 0;
	HI_U32 number = 0;
    HI_U32 u32Height = 0, u32Width = 0, u32Chn = 0, u32Stride = 0, u32Dim = 0;
    HI_U32 u32VarSize = 0;
    HI_S32 s32Ret = HI_SUCCESS;
    HI_U8*pu8PicAddr = NULL;
    HI_U32*pu32StepAddr = NULL;
    HI_U32 u32SegIdx = pstInputDataIdx->u32SegIdx;
    HI_U32 u32NodeIdx = pstInputDataIdx->u32NodeIdx;
    HI_U32 u32TotalStepNum = 0;
	HI_U32 mean_val_rgb[3] = {0.485,0.456,0.406};
	HI_U32 var_val_rgb[3] = {0.229,0.224,0.225};

    /*open file*/
    if (NULL != pstNnieCfg->pszPic)
    {
        fp = fopen(pstNnieCfg->pszPic,"rb");
        SAMPLE_SVP_CHECK_EXPR_RET(NULL == fp,HI_INVALID_VALUE,SAMPLE_SVP_ERR_LEVEL_ERROR,
            "Error, open file failed!\n");
    }

    /*get data size*/
    if(SVP_BLOB_TYPE_U8 <= pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].enType &&
        SVP_BLOB_TYPE_YVU422SP >= pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].enType)
    {
        u32VarSize = sizeof(HI_U8);
    }
    else
    {
        u32VarSize = sizeof(HI_U32);
    }

    /*fill src data*/
    if(SVP_BLOB_TYPE_SEQ_S32 == pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].enType)
    {
        u32Dim = pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].unShape.stSeq.u32Dim;
		number = u32Dim / 3;
        u32Stride = pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u32Stride;
        pu32StepAddr = (HI_U32*)(pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].unShape.stSeq.u64VirAddrStep);
        pu8PicAddr = (HI_U8*)(pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u64VirAddr);
        for(n = 0; n < pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u32Num; n++)
        {
            for(i = 0;i < *(pu32StepAddr+n); i++)
            {
                s32Ret = fread(pu8PicAddr,u32Dim*u32VarSize,1,fp);
                SAMPLE_SVP_CHECK_EXPR_GOTO(1 != s32Ret,FAIL,SAMPLE_SVP_ERR_LEVEL_ERROR,"Error,Read image file failed!\n");

				for(m = 0;m < number; m++)//预处理
				{
					pu8PicAddr[m] = (pu8PicAddr[m]/255 - mean_val_rgb[0])/var_val_rgb[0];
					pu8PicAddr[m + number] = (pu8PicAddr[m + number]/255 - mean_val_rgb[1])/var_val_rgb[1];
					pu8PicAddr[m + number * 2] = (pu8PicAddr[m + number * 2]/255 - mean_val_rgb[2])/var_val_rgb[2];
				}
                pu8PicAddr += u32Stride;
            }
            u32TotalStepNum += *(pu32StepAddr+n);
        }
        SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u64PhyAddr,
            (HI_VOID *) pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u64VirAddr,
            u32TotalStepNum*u32Stride);
    }
    else
    {
        u32Height = pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].unShape.stWhc.u32Height;
        u32Width = pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].unShape.stWhc.u32Width;
        u32Chn = pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].unShape.stWhc.u32Chn;
        u32Stride = pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u32Stride;
        pu8PicAddr = (HI_U8*)(pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u64VirAddr);
        if(SVP_BLOB_TYPE_YVU420SP== pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].enType)
        {
            for(n = 0; n < pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u32Num; n++)
            {
                for(i = 0; i < u32Chn*u32Height/2; i++)
                {
                    s32Ret = fread(pu8PicAddr,u32Width*u32VarSize,1,fp);
                    SAMPLE_SVP_CHECK_EXPR_GOTO(1 != s32Ret,FAIL,SAMPLE_SVP_ERR_LEVEL_ERROR,"Error,Read image file failed!\n");
                    pu8PicAddr += u32Stride;
                }
            }
        }
        else if(SVP_BLOB_TYPE_YVU422SP== pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].enType)
        {
            for(n = 0; n < pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u32Num; n++)
            {
                for(i = 0; i < u32Height*2; i++)
                {
                    s32Ret = fread(pu8PicAddr,u32Width*u32VarSize,1,fp);
                    SAMPLE_SVP_CHECK_EXPR_GOTO(1 != s32Ret,FAIL,SAMPLE_SVP_ERR_LEVEL_ERROR,"Error,Read image file failed!\n");
                    pu8PicAddr += u32Stride;
                }
            }
        }
        else
        {
            for(n = 0; n < pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u32Num; n++)
            {
                for(i = 0;i < u32Chn; i++)
                {
                    for(j = 0; j < u32Height; j++)
                    {
                        s32Ret = fread(pu8PicAddr,u32Width*u32VarSize,1,fp);
                        SAMPLE_SVP_CHECK_EXPR_GOTO(1 != s32Ret,FAIL,SAMPLE_SVP_ERR_LEVEL_ERROR,"Error,Read image file failed!\n");
                        pu8PicAddr += u32Stride;
                    }
                }
            }
        }
        SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u64PhyAddr,
            (HI_VOID *) pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u64VirAddr,
            pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u32Num*u32Chn*u32Height*u32Stride);
    }

    fclose(fp);
    return HI_SUCCESS;
FAIL:

    fclose(fp);
    return HI_FAILURE;
}



/******************************************************************************
* function : IMGCAP Deinit
******************************************************************************/
static HI_S32 SAMPLE_SVP_NNIE_ImaCap_Deinit(SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
    SAMPLE_SVP_NNIE_SSD_SOFTWARE_PARAM_S* pstSoftWareParam,SAMPLE_SVP_NNIE_MODEL_S *pstNnieModel)
{
    HI_S32 s32Ret = HI_SUCCESS;
    /*hardware deinit*/
    if(pstNnieParam!=NULL)
    {
        s32Ret = SAMPLE_COMM_SVP_NNIE_ParamDeinit(pstNnieParam);
        SAMPLE_SVP_CHECK_EXPR_TRACE(HI_SUCCESS != s32Ret,SAMPLE_SVP_ERR_LEVEL_ERROR,
            "Error,SAMPLE_COMM_SVP_NNIE_ParamDeinit failed!\n");
    }

    /*model deinit*/
    if(pstNnieModel!=NULL)
    {
        s32Ret = SAMPLE_COMM_SVP_NNIE_UnloadModel(pstNnieModel);
        SAMPLE_SVP_CHECK_EXPR_TRACE(HI_SUCCESS != s32Ret,SAMPLE_SVP_ERR_LEVEL_ERROR,
            "Error,SAMPLE_COMM_SVP_NNIE_UnloadModel failed!\n");
    }
    return s32Ret;
}

/******************************************************************************
* function : IMGCAP init
******************************************************************************/
static HI_S32 SAMPLE_SVP_NNIE_ImgCap_ParamInit(SAMPLE_SVP_NNIE_CFG_S* pstCfg,
    SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam, SAMPLE_SVP_NNIE_SSD_SOFTWARE_PARAM_S* pstSoftWareParam)
{
    HI_S32 s32Ret = HI_SUCCESS;
    /*init hardware para*/
    s32Ret = SAMPLE_COMM_SVP_NNIE_ParamInit(pstCfg,pstNnieParam);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret,INIT_FAIL_0,SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error(%#x),SAMPLE_COMM_SVP_NNIE_ParamInit failed!\n",s32Ret);

    return s32Ret;
INIT_FAIL_0:
    s32Ret = SAMPLE_SVP_NNIE_ImaCap_Deinit(pstNnieParam,pstSoftWareParam,NULL);
    SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret,s32Ret,SAMPLE_SVP_ERR_LEVEL_ERROR,
            "Error(%#x),SAMPLE_SVP_NNIE_ImaCap_Deinit failed!\n",s32Ret);
    return HI_FAILURE;

}

void SAMPLE_SVP_NNIE_ImaCap_init()
{
    /*Sys init*/
    //SAMPLE_COMM_SVP_CheckSysInit();    //系统初始化 system init 

    HI_S32 s32Ret = HI_SUCCESS;                                           //HI_S32就是 int s32Ret = 0 
    HI_CHAR *pcModelName = "./imagecaption_encoder.wk";    // 设置模型路径
    SAMPLE_SVP_TRACE_INFO("Image caption encoder Load model!\n");   //输出模型的信息
    s32Ret = SAMPLE_COMM_SVP_NNIE_LoadModel(pcModelName,&s_stImgCapModel);   //载入模型 
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret,ENCODER_FAIL_0,SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_COMM_SVP_NNIE_LoadModel failed!\n");    //错误报告
    return;
ENCODER_FAIL_0:
    SAMPLE_SVP_NNIE_ImaCap_Deinit(&s_stImgCapNnieParam,&s_stImgCapSoftwareParam,&s_stImgCapModel);
    //SAMPLE_COMM_SVP_CheckSysExit();

}

void SAMPLE_SVP_NNIE_ImaCap_End()
{
    SAMPLE_SVP_NNIE_ImaCap_Deinit(&s_stImgCapNnieParam,&s_stImgCapSoftwareParam,&s_stImgCapModel);
    //SAMPLE_COMM_SVP_CheckSysExit();
}

/******************************************************************************
* function : show IMGCAP sample(image 384x384 U8_C3)
******************************************************************************/
void SAMPLE_SVP_NNIE_Encode(HI_CHAR *pcSrcFile, HI_FLOAT *aps32PermuteResult[])
{
    HI_U32 u32PicNum = 1;                                                  
    HI_FLOAT f32PrintResultThresh = 0.0f;                                
    HI_S32 s32Ret = HI_SUCCESS;                                         
    SAMPLE_SVP_NNIE_CFG_S   stNnieCfg = {0};                          
    SAMPLE_SVP_NNIE_INPUT_DATA_INDEX_S stInputDataIdx = {0};
    SAMPLE_SVP_NNIE_PROCESS_SEG_INDEX_S stProcSegIdx = {0};

    /*Set configuration parameter*/                 
    stNnieCfg.pszPic= pcSrcFile;                       
    stNnieCfg.u32MaxInputNum = u32PicNum;  //max input image num in each batch 
    stNnieCfg.u32MaxRoiNum = 0;
    stNnieCfg.aenNnieCoreId[0] = SVP_NNIE_ID_0;//set NNIE core

    //clock_t finish = clock();
    //printf("time cost %d ms:", (finish - start)/1000);
    /*Ssd parameter initialization*/
    /*Ssd parameters are set in SAMPLE_SVP_NNIE_Ssd_SoftwareInit,
      if user has changed net struct, please make sure the parameter settings in
      SAMPLE_SVP_NNIE_Ssd_SoftwareInit function are correct*/
    SAMPLE_SVP_TRACE_INFO("Image caption encoder parameter initialization!\n");
    s_stImgCapNnieParam.pstModel = &s_stImgCapModel.stModel;
    s32Ret = SAMPLE_SVP_NNIE_ImgCap_ParamInit(&stNnieCfg,&s_stImgCapNnieParam,&s_stImgCapSoftwareParam);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret,SSD_FAIL_0,SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_SVP_NNIE_ImgCap_ParamInit failed!\n");

    /*Fill src data*/
    SAMPLE_SVP_TRACE_INFO("Image caption encoder start!\n");
    stInputDataIdx.u32SegIdx = 0;
    stInputDataIdx.u32NodeIdx = 0;
    s32Ret = IMGCAP_SVP_NNIE_FillSrcData(&stNnieCfg,&s_stImgCapNnieParam,&stInputDataIdx);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret,SSD_FAIL_0,SAMPLE_SVP_ERR_LEVEL_ERROR,
       "Error,IMGCAP_SVP_NNIE_FillSrcData failed!\n");

    /*NNIE process(process the 0-th segment)*/
    stProcSegIdx.u32SegIdx = 0;
    s32Ret = IMGCAP_SVP_NNIE_Forward(&s_stImgCapNnieParam,&stInputDataIdx,&stProcSegIdx,HI_TRUE);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret,SSD_FAIL_0,SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,IMGCAP_SVP_NNIE_Forward failed!\n");

    //printf("frame_num:%d(%d_%d_%d)\n", s_stImgCapNnieParam.astSegData[0].astDst[0].u32Num, \
    s_stImgCapNnieParam.astSegData[0].astDst[0].unShape.stWhc.u32Chn,\
    s_stImgCapNnieParam.astSegData[0].astDst[0].unShape.stWhc.u32Height,\
    s_stImgCapNnieParam.astSegData[0].astDst[0].unShape.stWhc.u32Width);
    unsigned long size_out = s_stImgCapNnieParam.astSegData[0].astDst[0].unShape.stWhc.u32Width * \
                            s_stImgCapNnieParam.astSegData[0].astDst[0].unShape.stWhc.u32Height * \
                            s_stImgCapNnieParam.astSegData[0].astDst[0].unShape.stWhc.u32Chn;
    for(size_t i = 0; i < size_out; i++)
    {
        *((HI_FLOAT*)s_stImgCapNnieParam.astSegData[0].astDst[0].u64VirAddr + i) = (HI_FLOAT)(*((HI_S32*)s_stImgCapNnieParam.astSegData[0].astDst[0].u64VirAddr + i));
    }
    memcpy(aps32PermuteResult[0], (HI_FLOAT*)s_stImgCapNnieParam.astSegData[0].astDst[0].u64VirAddr, sizeof(HI_FLOAT) * size_out); 

SSD_FAIL_0:
    SAMPLE_COMM_SVP_NNIE_ParamDeinit(&s_stImgCapNnieParam);
    return;
}





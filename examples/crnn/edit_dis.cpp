#include <stdio.h>
#include <string.h>
#include <iostream>

int minDistance(char* word1, char* word2) {
    int n1 = strlen(word1), n2 = strlen(word2);
    int dp[n1 + 1][n2 + 1];
    for (int i = 0; i <= n1; ++i) dp[i][0] = i;
    for (int i = 0; i <= n2; ++i) dp[0][i] = i;
    for (int i = 1; i <= n1; ++i) {
        for (int j = 1; j <= n2; ++j) {
            if (word1[i - 1] == word2[j - 1] || word1[i - 1] == word2[j - 1] + 32) {
                dp[i][j] = dp[i - 1][j - 1];
            } else {
                dp[i][j] = std::min(dp[i - 1][j - 1], std::min(dp[i - 1][j], dp[i][j - 1])) + 1;
            }
        }
    }
    return dp[n1][n2];
}

char* minDistanceWord(char* result)
{
 //   setlocale(LC_ALL, "en_US.utf8");
#ifdef ARM_HISI
    FILE* fp=fopen("lexicon.txt","r");
#else
    FILE* fp=fopen("../../../models/crnn/lexicon.txt","r");
#endif
    if(!fp) {
        perror("lexicon opening failed");
        return NULL;
    }
    char buf[30];
    char *finres;
    if(NULL == (finres = (char *)malloc(26 * sizeof(char))))
    /*请使用if来判断,这是有必要的*/
    {
        perror("error...");
        exit(1);
    }
    int len=0,mindis = __INT_MAX__;

    while (fgets(buf, 30, fp) != NULL) {
        len = strlen(buf);
        buf[len-1] = '\0';  /*去掉换行符*/
        int dis = minDistance(result, buf);
#ifdef __DEBUG__
        printf("%s_%d\n",buf,dis);
#endif
        if(dis < mindis)
        {    
            mindis = dis;
            strcpy(finres, buf);
        }
    }
    if(mindis < (strlen(result) >> 2)) 
        return finres;
    else
        return NULL;
}

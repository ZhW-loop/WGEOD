#include "cnn_t.h"
#include<stdio.h>
#include<cmath>
#include<cstdlib>
#include<ctime>
#include <algorithm>
using namespace std;
int16_t ifm[10000000] = { 0 };
int16_t weight_rg[10000000] = { 0 };
int16_t ofm[100000000] = { 0 };
int16_t bias[1024] = { 0 };
int16_t ofm_soft[100000000] = { 0 };
int16_t weight[10000000] = { 0 };
void weight_rearrangement(int16_t* weight, int16_t* weight_rg, int ofm_num, int ifm_num, int ksize)
{	
	int w_cnt = 0;
	int TN = min(ifm_num, Tn);
	int TM = min(ofm_num, Tm);
	int o_M = ceil((float)ofm_num / TM), i_N = ceil((float)ifm_num / TN), kk = ksize * ksize;
	for (int t1 = 0; t1 < o_M; t1++)
		for (int t2 = 0; t2 < i_N; t2++)
			for (int k = 0; k < ksize * ksize; k++)
				for (int tm = 0; tm < TM; tm++)
					for (int tn = 0; tn < TN; tn++)
					{
						int rg_offset = t1 * i_N * kk * TM * TN + t2 * kk * TM * TN + k * TM * TN + tm * TN + tn;
						int x1 = t1 * TM + tm, x2 = t2 * TN + tn, x3 = k;
						int source_offset = x1 * ifm_num * kk + x2 * kk + x3;
						if (x1 < ofm_num && x2 < ifm_num && x3 < kk)
						{
							weight_rg[w_cnt++] = weight[source_offset];
						}
					}
}

void conv_soft(int ifm_num, int ifm_h, int ifm_w, int ofm_num, int ofm_h, int ofm_w, int ksize, int stride,
	int16_t* ifm, int inputQ, int16_t* weight, int weightQ, int16_t* ofm, int outputQ,
	int16_t* bias, int biasQ, int pad_int)
{
	for(int i=0;i<ofm_num;i++)
		for(int j=0;j<ofm_h;j++)
			for (int k = 0; k < ofm_w; k++)
			{
				int32_t sum = 0;
				for (int c = 0; c < ifm_num; c++)
					for(int ii=0;ii<ksize;ii++)
						for (int jj = 0; jj < ksize; jj++)
						{
							int row = j * stride - pad_int + ii;
							int col = k * stride - pad_int + jj;
							if (row >= 0 && col >= 0 && row < ifm_h && col < ifm_w)
							{
								int32_t dat = ifm[c * ifm_h * ifm_w + row * ifm_w + col];
								int32_t wt = weight[i * ifm_num * ksize * ksize + c * ksize * ksize + ii * ksize + jj];
								sum += dat * wt;
							}
						}
				int32_t res = (sum >> (inputQ + weightQ - outputQ)) + (bias[i] >> (biasQ - inputQ));
				if (res > 32767) res = 32767;
				else if (res < -32768) res = -32768;
				ofm[i * ofm_h * ofm_w + j * ofm_w + k] = res;
			}
}
int main()
{
	int ksize = 3;
	int stride = 1;
	int pad_int = 1;
	int ifm_h = 27;
	int ifm_w = 27;
	int ifm_num = 3;
	int ofm_h = 27;
	int ofm_w = 27;
	int ofm_num = 65;
	int linear = 1;
	/*
	int16_t* ifm = new int16_t[ifm_w * ifm_h * ifm_num+1024];
	int16_t* weight_rg = new int16_t[ofm_num * ifm_num * ksize * ksize];
	int16_t* ofm = new int16_t[ofm_w * ofm_h * ofm_num];
	int16_t* bias = new int16_t[1024];
	int16_t* ofm_soft = new int16_t[ofm_w * ofm_h * ofm_num];
	int16_t* weight = new int16_t[ofm_num * ifm_num * ksize * ksize];
	*/
	//printf("ok\n");
	
	//printf("ok\n");
	srand((int)time(0));
	for (int i = 0; i < ifm_w * ifm_h * ifm_num; i++)
	{
		//ifm[i] = i;
		ifm[i] = rand() % 100;
		//printf("ifm\n");
		//printf("%d\t" ,ifm[i]);
	}
		
	for (int i = 0; i < ofm_num * ifm_num * ksize * ksize; i++)
	{	
		//printf("weight\n");
		//weight[i] = (i - 1) * 2;
	    weight[i] = rand() % 100;
		//printf("%d\t", weight[i]);
	}
		
	for (int i = 0; i < ofm_num; i++)
	{	
		//printf("bias\n");
		//bias[i] = 50-i ;
		bias[i] = rand() % 100;
		//printf("%d\t", bias[i]);
	}
		

	weight_rearrangement(weight, weight_rg, ofm_num, ifm_num, ksize);

	uint16_t TN = min(ifm_num, Tn);
	uint16_t TM = min(ofm_num, Tm);
	uint16_t TR = min(ofm_h, Tr);
	int mLoops = (int)ceil(((float)ofm_num) / TM);
	uint16_t TC = min(ofm_w, Tc);
	int OFM_num_bound = (mLoops + 1) * TM;
	int mLoopsxTM = mLoops * TM;
	uint16_t TRow = (TR - 1) * stride + ksize;
	uint16_t TCol = (TC - 1) * stride + ksize;
	int TRowTCol = (TRow << 16) | TCol;
	int TRTC = (TR << 16) | TC;
	int TMTN = (TM << 16) | TN;
	int ofm_w_h = (ofm_w << 16) | ofm_h;
	int ifm_w_h = (ifm_w << 16) | ifm_h;
	int iofm_num = (ifm_num << 16) | ofm_num;
	int k_s_pad_ltype = (ksize << 24) | (stride << 16) | (pad_int << 8) | 0;
	int en_bits = 0x0;
	en_bits |= (0x1 << 1);
	if (linear == 0)
		en_bits |= 0x1 << 2;

	Conv_Acc(ifm,ofm, weight_rg, bias,
		k_s_pad_ltype, iofm_num, ifm_w_h, ofm_w_h, TRTC, TMTN, OFM_num_bound, mLoopsxTM, 0,
		TRowTCol, en_bits, 8, 8, 8, 8);
	//printf("ok\n");
	conv_soft(ifm_num, ifm_h, ifm_w, ofm_num, ofm_h, ofm_w, ksize, stride,
		ifm, 8, weight, 8, ofm_soft, 8, bias, 8, pad_int);
	int flag = 1;
	for(int c = 0; c< ofm_num;c++)
		for(int i=0;i<ofm_h;i++)
			for (int j = 0; j < ofm_w; j++)
			{
				if (ofm[c * ofm_h * ofm_w + i * ofm_w + j] != ofm_soft[c * ofm_h * ofm_w + i * ofm_w + j])
				//if(c==28)
				{
					flag = 0;
					printf("Out_  [%d][%d][%d]=%x\n", c, i, j, ofm[c * ofm_h * ofm_w + i * ofm_w + j]);
					printf("Out_Soft[%d][%d][%d]=%x\n", c, i, j, ofm_soft[c * ofm_h * ofm_w + i * ofm_w + j]);
				}
			}
	if (flag == 1)
		printf("============================\n   result_match\n============================\n");
	else
		printf("============================\n   result_mismatch\n============================\n");
}



#ifndef _CNN_CPP_
#define _CNN_CPP_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#include <ap_int.h>
//#include <math.h>

#define MIN_diy(x,y) ((x) < (y) ? (x) : (y))
#define MAX_diy(x,y) ((x) > (y) ? (x) : (y))

#define FALSE 0
#define TRUE  1

#define VALID 0
#define SAME  1

#define LT_CONV  0
#define LT_MAXPOOL 1


///#DEFINE_HEADER#

#define HW_S 1
#define K 3
#define Tn 2
#define Tm 16
#define Tr 26
#define Tc 26
#define MAX_BETA_LENGTH 1024

#define OnChipIB_Width  ((Tc-1)*HW_S+K)
#define OnChipIB_Height ((Tr-1)*HW_S+K)

typedef int16_t trans_type;

void Conv_Acc(trans_type *ifm, trans_type *ofm, trans_type *weight, trans_type *bias, uint32_t k_s_pad_ltype, uint32_t iofm_num, uint32_t ifm_w_h, uint32_t ofm_w_h,
	uint32_t TRTC, uint32_t TMTN, int OFM_num_bound, int mLoopsxTM, int16_t pad_val, uint32_t TRowTCol,
	uint32_t en_bits, int WeightQ, int BetaQ, int InputQ, int OutputQ);//enable_bits[2:0]={IsReLU, LoadBias, IsNotConv}

#endif

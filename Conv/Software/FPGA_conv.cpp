
#include "cnn_t.h"

void ifm_mmcpy_row(int16_t* input, int16_t local_buf[OnChipIB_Width + 3], int CurrentOffset, int Roffset, uint32_t IHxIW, int IW_align_256b, uint16_t TCol,
	uint8_t t1, uint8_t t2, uint8_t* t1_n, uint8_t* t2_n, bool enable)
{
	//#pragma HLS INTERFACE m_axi depth=512 port=input offset=slave max_read_burst_length=128 max_write_burst_length=128
	int current_r = Roffset + t2;
	if ((!enable) || (current_r < 0))
		return;
	//printf("ok\n");
	int ifm_offset = CurrentOffset + t1 * IHxIW + t2 * IW_align_256b;

	assert((TCol > 0) && (TCol <= (OnChipIB_Width + 3)));
	memcpy(local_buf, (int16_t*)(input + ifm_offset), TCol * sizeof(int16_t));
	*t1_n = t1;
	*t2_n = t2;
}

void ifm_copy_lbuf2ibuf(int16_t input_buffer[Tn][OnChipIB_Height][OnChipIB_Width], int16_t local_buf[OnChipIB_Width + 3], uint16_t TCol, uint16_t Input_w, uint16_t Input_h, uint8_t TN_MIN,
	int16_t pad_val, int Coffset, int Roffset, uint8_t t1, uint8_t t2, bool enable)
{
	if (!enable)
		return;
	//printf("hello");
	bool TN_Enable = t1 < TN_MIN;
	int yoffset = Roffset + t2;
	bool YEnable = (yoffset >= 0) && (yoffset < Input_h);
	bool PEnable = YEnable && TN_Enable;
	int cnt = 0, ofst = 0;
	int t3 = 0, loopcnts = (int)TCol;
	if (Coffset < 0)
	{
		t3 = -1;
		loopcnts = loopcnts - 1;
	}
	for (; t3 < loopcnts; t3++)
	{
#pragma HLS LOOP_TRIPCOUNT min=1 max=4
#pragma HLS PIPELINE II=1
		int xoffset = Coffset + ofst;
		ofst = ofst + 1;
		bool XEnable = (xoffset >= 0) && (xoffset < Input_w);
		if (XEnable && PEnable)
		{
			input_buffer[t1][t2][cnt++] = local_buf[t3];
		}
		else
			input_buffer[t1][t2][cnt++] = pad_val;
	}
	//assert((cnt>=0)&&(cnt<=(OnChipIB_Width/2+3)));
}

void input_load(int16_t* input, int16_t input_buffer[Tn][OnChipIB_Height][OnChipIB_Width], uint16_t r, uint16_t c, uint16_t n, uint8_t Kstride, uint8_t Padding,
	uint16_t TRow, uint16_t TCol, uint16_t Input_w, int IW_align_256b, uint16_t Input_h, uint8_t TN_MIN, uint32_t IHxIW, int16_t pad_val, int16_t Coffset_mmcpy)
{
#pragma HLS INTERFACE m_axi depth=512 port=input offset=slave max_read_burst_length=128 max_write_burst_length=128

	uint8_t t1, t2;
	static int16_t local_buf0[OnChipIB_Width + 3];
	static int16_t local_buf1[OnChipIB_Width + 3];

	const int Coffset = c * Kstride - Padding;
	const int Roffset = r * Kstride - Padding;
	const int CurrentOffset = n * IHxIW + Roffset * IW_align_256b + Coffset_mmcpy;

	uint8_t t1_n0 = 0, t1_n1 = 0, t2_n0 = 0, t2_n1 = 0;
	bool pp = true;

	int TnxTRow = TN_MIN * TRow;
	int t = 0;
	t1 = 0; t2 = 0;
	for (t = 0; t < TnxTRow + 1; t++)
	{
#pragma HLS LOOP_TRIPCOUNT min=1 max=27
		if (pp)
		{
			ifm_mmcpy_row(input, local_buf0, CurrentOffset, Roffset, IHxIW, IW_align_256b, TCol, t1, t2, &t1_n0, &t2_n0, t != TnxTRow);
			ifm_copy_lbuf2ibuf(input_buffer, local_buf1, TCol, Input_w, Input_h, TN_MIN, pad_val, Coffset, Roffset, t1_n1, t2_n1, t != 0);
			pp = false;
		}
		else
		{
			ifm_mmcpy_row(input, local_buf1, CurrentOffset, Roffset, IHxIW, IW_align_256b, TCol, t1, t2, &t1_n1, &t2_n1, t != TnxTRow);
			ifm_copy_lbuf2ibuf(input_buffer, local_buf0, TCol, Input_w, Input_h, TN_MIN, pad_val, Coffset, Roffset, t1_n0, t2_n0, t != 0);
			pp = true;
		}

		t2++;
		if (t2 == TRow)
		{
			t2 = 0;
			t1++;
		}
	}

}

void weight_load_reorg(int16_t* Weight, int16_t weight_buffer[Tm][Tn][K][K],
	int m, int n, uint8_t Ksize, uint8_t TM_MIN, uint8_t TN_MIN, bool enable, uint16_t TN, uint16_t TM)
{
	if (!enable)
		return;

	uint8_t t1, t2, t3, t4;
	static int16_t local_buf[Tm * Tn * K * K + 3];
	static int Woffset;

	assert((TM_MIN > 0) && (TM_MIN <= Tm));
	assert((TN_MIN > 0) && (TN_MIN <= Tn));

	if (m == 0 && n == 0)
		Woffset = 0;

	uint16_t mm_offset = TM_MIN * TN_MIN * Ksize * Ksize;
	assert((mm_offset > 0) && (mm_offset <= Tm * Tn * K * K + 3));
	memcpy(local_buf, (int16_t*)(Weight + Woffset), mm_offset * sizeof(int16_t));
	//Woffset += TN * TM * KxK;
	Woffset += mm_offset;
	uint16_t cnt = 0;

	for (t3 = 0; t3 < Ksize; t3++)
#pragma HLS LOOP_TRIPCOUNT min=1 max=3
		for (t4 = 0; t4 < Ksize; t4++)
#pragma HLS LOOP_TRIPCOUNT min=1 max=3
			for (t1 = 0; t1 < Tm; t1++)
				for (t2 = 0; t2 < Tn; t2++)
				{
#pragma HLS PIPELINE II=1
					bool Enable = (t1 < TM_MIN) && (t2 < TN_MIN);
					if (Enable)
					{
						weight_buffer[t1][t2][t3][t4] = local_buf[cnt++];
					}
					else
						weight_buffer[t1][t2][t3][t4] = 0;
				}
}

void nonlinear_leaky_row(int16_t local_buf[Tc], int32_t Input[Tm][Tr][Tc], int16_t local_beta_buffer[Tm], uint8_t tm, uint8_t tr, uint8_t* tm_n, uint8_t* tr_n, uint8_t TC_MIN,
	bool IsNL, bool enable, int ltype, int WeightAddInputSubInter)
{
	if (!enable)
		return;

	uint8_t tc;
	//	float tmp_out, tmp, tmp1;
	assert((TC_MIN > 0) && (TC_MIN <= Tc));
	///assert((InterSubOutput>0)&&(InterSubOutput<32));

	uint8_t cnt = 0;

	int32_t tmp_out, tmp1, tmp0;

	int16_t tmp_int16;
	for (tc = 0; tc < TC_MIN; tc++)
	{
#pragma HLS LOOP_TRIPCOUNT min=1 max=26
#pragma HLS PIPELINE II=1
		tmp0 = Input[tm][tr][tc];
		tmp1 = (tmp0 >> WeightAddInputSubInter) + local_beta_buffer[tm];

		if (IsNL)
		{
			if (tmp1 < 0)
				tmp_out = ((int32_t)tmp1 * 0xccc) >> 15;
			else
				tmp_out = tmp1;
		}
		else
			tmp_out = tmp1;

		if (tmp_out > 32767) tmp_out = 32767;
		else if (tmp_out < -32768) tmp_out = -32768;
		tmp_int16 = tmp_out;//tmp_out*pow(2.0, OutputQ-Inter)

		local_buf[cnt++] = tmp_int16;
	}

	*tm_n = tm;
	*tr_n = tr;
}

void ofm_mmcpy_row(int16_t* Output, int16_t local_buf[Tc], int offset, uint32_t OHxOW, uint16_t Output_w, uint8_t TC_MIN, uint8_t tm, uint8_t tr, bool enable)
{
	if (!enable)
		return;

	int ofm_offset = tm * OHxOW + tr * Output_w + offset;
	int trans_offset = ofm_offset;
	uint16_t loop_cnts = TC_MIN;

	assert((loop_cnts > 0) && (loop_cnts <= Tc));

	memcpy((int16_t*)(Output + trans_offset), local_buf, loop_cnts * sizeof(int16_t));

}

void copy_local_beta(int16_t beta_buffer[MAX_BETA_LENGTH], int16_t local_beta_buffer[Tm], const int TM_MIN, int m, int InterSubBeta)
{
	assert((InterSubBeta >= 0) && (InterSubBeta < 32));

	int offset = m;
	int tm;
	for (tm = 0; tm < TM_MIN; tm++)
	{
#pragma HLS LOOP_TRIPCOUNT min=1 max=32
#pragma HLS PIPELINE II=1
		local_beta_buffer[tm] = (int16_t)beta_buffer[offset] << InterSubBeta;
		offset++;
	}
}

void write_back_output_reorg(int32_t output_buffer[Tm][Tr][Tc], int16_t* Output, int16_t beta_buffer[MAX_BETA_LENGTH], uint16_t r, uint16_t c, uint16_t m, uint16_t Output_w,
	uint8_t TM_MIN, uint8_t TR_MIN, uint8_t TC_MIN, uint32_t OHxOW, bool IsNL, bool enable, int ltype,
	int InterSubBeta, int WeightAddInputSubInter)
{
	if (!enable)
		return;
	static int16_t local_beta_buffer[Tm];
#pragma HLS ARRAY_PARTITION variable=local_beta_buffer complete dim=1
	copy_local_beta(beta_buffer, local_beta_buffer, TM_MIN, m, InterSubBeta);

	assert((TM_MIN > 0) && (TM_MIN <= Tm));
	assert((TR_MIN > 0) && (TR_MIN <= Tr));
	assert((TC_MIN > 0) && (TC_MIN <= Tc));

	const int offset = m * OHxOW + r * Output_w + c;
	static int16_t local_buf0[Tc];
	static int16_t local_buf1[Tc];
	uint8_t tm_n0 = 0, tm_n1 = 0, tr_n0 = 0, tr_n1 = 0;

	bool pp = true;
	uint8_t tr = 0, tm = 0;
	uint16_t TM_MINxTR_MIN = TM_MIN * TR_MIN;
	uint16_t t;
	for (t = 0; t < TM_MINxTR_MIN + 1; t++)
	{
#pragma HLS LOOP_TRIPCOUNT min=1 max=832
		if (pp)
		{
			nonlinear_leaky_row(local_buf0, output_buffer, local_beta_buffer, tm, tr, &tm_n0, &tr_n0, TC_MIN, IsNL, t != TM_MINxTR_MIN, ltype, WeightAddInputSubInter);
			ofm_mmcpy_row(Output, local_buf1, offset, OHxOW, Output_w, TC_MIN, tm_n1, tr_n1, t != 0);
			pp = false;
		}
		else
		{
			nonlinear_leaky_row(local_buf1, output_buffer, local_beta_buffer, tm, tr, &tm_n1, &tr_n1, TC_MIN, IsNL, t != TM_MINxTR_MIN, ltype, WeightAddInputSubInter);
			ofm_mmcpy_row(Output, local_buf0, offset, OHxOW, Output_w, TC_MIN, tm_n0, tr_n0, t != 0);
			pp = true;
		}

		tr++;
		if (tr == TR_MIN)
		{
			tr = 0;
			tm++;
		}
	}

}


void conv2d_tile(int16_t input_buffer[Tn][OnChipIB_Height][OnChipIB_Width], int32_t output_buffer[Tm][Tr][Tc],
	int16_t weight_buffer[Tm][Tn][K][K], int n_next,
	const int Ksize, const int Kstride, int m,
	const int TM_MIN, const int TR_MIN, const int TC_MIN, bool enable, uint16_t TR, uint16_t TC)
{
	/*
	#pragma HLS ARRAY_PARTITION variable=weight_buffer complete dim=2
	#pragma HLS ARRAY_PARTITION variable=weight_buffer complete dim=1
	#pragma HLS ARRAY_PARTITION variable=output_buffer complete dim=1
	#pragma HLS ARRAY_PARTITION variable=input_buffer complete dim=1
	*/
	uint16_t i, j;
	uint16_t Ksize_3b = Ksize;
	uint16_t Kstride_3b = Kstride;
	uint16_t TR_MIN_8b = TR_MIN;
	uint16_t TC_MIN_8b = TC_MIN;
	uint16_t tr, tc, tm, tn;
	const bool ne0 = (n_next == 0);
	int32_t partial_mul[Tm][Tn];
#pragma HLS ARRAY_PARTITION variable=partial_mul complete dim=1
#pragma HLS ARRAY_PARTITION variable=partial_mul complete dim=2
	int32_t partial_add[Tm];
#pragma HLS ARRAY_PARTITION variable=partial_add complete dim=1

	for (i = 0; i < Ksize_3b; i++)
#pragma HLS LOOP_TRIPCOUNT min=1 max=3
		for (j = 0; j < Ksize_3b; j++)
#pragma HLS LOOP_TRIPCOUNT min=1 max=3
			for (tr = 0; tr < TR; tr++)
#pragma HLS LOOP_TRIPCOUNT min=1 max=26
				for (tc = 0; tc < TC; tc++)
				{
#pragma HLS LOOP_TRIPCOUNT min=1 max=26
#pragma HLS PIPELINE II=1
					for (tm = 0; tm < Tm; tm++)
					{
#pragma HLS UNROLL
#pragma HLS DEPENDENCE variable=output_buffer inter false
						if (i == 0 && j == 0 && ne0)
							partial_add[tm] = 0;
						else
							partial_add[tm] = output_buffer[tm][tr][tc];

						for (tn = 0; tn < Tn; tn++)
						{
#pragma HLS UNROLL
							partial_mul[tm][tn] = weight_buffer[tm][tn][i][j] * input_buffer[tn][Kstride_3b * tr + i][Kstride_3b * tc + j];
						}

						int32_t partial_sum = 0;
						for (tn = 0; tn < Tn; tn++)
						{
#pragma HLS UNROLL
							partial_sum += partial_mul[tm][tn];
						}
						output_buffer[tm][tr][tc] = (partial_add[tm] + partial_sum);
						//printf("%d\n", output_buffer[tm][tr][tc]);
					}
				}
}

void ifm_weight_load_wrapper(int16_t* ifm, int16_t* weight, int16_t ifm_buffer[Tn][OnChipIB_Height][OnChipIB_Width], int16_t weight_buffer[Tm][Tn][K][K],
	int ifm_num, int ofm_num, int ifm_w, int IW_align_256b, int ifm_h, int pad_int,
	int TM, int TN, int tr, int tc, int tm, int tn, int tn_next[1], int ksize, int kstride, int ltype, int TM_MIN, int TRow, int TCol, int IHW,
	bool enable, int16_t pad_val, bool LoadBias, int16_t Coffset_mmcpy)
{
	if (!enable)
		return;

	tn_next[0] = tn;

	int TN_MIN, tnm, TNM_MIN;
	TN_MIN = MIN_diy(TN, ifm_num - tn);
	tnm = tn;
	TNM_MIN = TN_MIN;

	input_load(ifm, ifm_buffer, tr, tc, tnm, kstride, pad_int, TRow, TCol, ifm_w, IW_align_256b, ifm_h, TNM_MIN, IHW, pad_val, Coffset_mmcpy);
	weight_load_reorg(weight, weight_buffer, tm, tn, ksize, TM_MIN, TN_MIN, LoadBias, TN, TM);
}

void load_compute_wrapper(int16_t* ifm, int16_t* weight, int32_t ofm_buffer[Tm][Tr][Tc], int16_t bias_buffer[MAX_BETA_LENGTH], int ksize, uint8_t K_1, int kstride, int ifm_num, int ifm_w, int IW_align_256b,
	int ifm_h, int ofm_num, int pad_int, int ltype, int TRow, int TCol, int IHW, int TC_MIN, int TR_MIN, int TM_MIN, int TM, int TN, int tm, int tr, int tc,
	int TMP_X_next[1], int TX_MIN_next[1], bool pingpongx, bool input_flag, int16_t pad_val, bool LoadBias, int InterSubBeta, int WeightAddInputSubInter, int16_t Coffset_mmcpy, uint16_t TR, uint16_t TC)
{
	static int16_t ifm_buffer0[Tn][OnChipIB_Height][OnChipIB_Width];
#pragma HLS ARRAY_RESHAPE variable=ifm_buffer0 complete dim=1
	static int16_t ifm_buffer1[Tn][OnChipIB_Height][OnChipIB_Width];
#pragma HLS ARRAY_RESHAPE variable=ifm_buffer1 complete dim=1

	static int16_t weight_buffer0[Tm][Tn][K][K];
#pragma HLS ARRAY_PARTITION variable=weight_buffer0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight_buffer0 complete dim=2
	static int16_t weight_buffer1[Tm][Tn][K][K];
#pragma HLS ARRAY_PARTITION variable=weight_buffer1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=weight_buffer1 complete dim=2


	static int n0[1];
	static int n1[1];

	static int tmp_x;
	static int tmp_tx_min;

	if (!input_flag)
		return;

	TMP_X_next[0] = tm;//consider by the inner-out loop
	TX_MIN_next[0] = TM_MIN;// like above

	bool pingpong = false;
	for (int tn = 0; tn < ifm_num + TN; tn += TN)
	{
#pragma HLS LOOP_TRIPCOUNT min=1 max=32
		if (pingpong)
		{
			ifm_weight_load_wrapper(ifm, weight, ifm_buffer1, weight_buffer1, ifm_num, ofm_num, ifm_w, IW_align_256b, ifm_h,
				pad_int, TM, TN, tr, tc, tm, tn, n1, ksize, kstride, ltype, TM_MIN, TRow, TCol, IHW, tn < ifm_num, pad_val, LoadBias, Coffset_mmcpy);
			conv2d_tile(ifm_buffer0, ofm_buffer, weight_buffer0, n0[0], ksize, kstride, tm, TM_MIN, TR_MIN, TC_MIN, tn != 0, TR, TC);
			pingpong = false;
		}
		else
		{
			ifm_weight_load_wrapper(ifm, weight, ifm_buffer0, weight_buffer0, ifm_num, ofm_num, ifm_w, IW_align_256b, ifm_h,
				pad_int, TM, TN, tr, tc, tm, tn, n0, ksize, kstride, ltype, TM_MIN, TRow, TCol, IHW, tn < ifm_num, pad_val, LoadBias, Coffset_mmcpy);
			conv2d_tile(ifm_buffer1, ofm_buffer, weight_buffer1, n1[0], ksize, kstride, tm, TM_MIN, TR_MIN, TC_MIN, tn != 0, TR, TC);
			pingpong = true;
		}
	}
}

void copy_beta(int16_t beta_buffer[MAX_BETA_LENGTH], int16_t* Beta, uint16_t OFM_NUM)
{
	static int16_t local_buf[MAX_BETA_LENGTH];
	uint16_t NUM = OFM_NUM;

	memcpy(local_buf, Beta, NUM * sizeof(int16_t));

	for (int t = 0; t < OFM_NUM; t++)
	{
#pragma HLS LOOP_TRIPCOUNT min=1 max=1500
#pragma HLS PIPELINE II=1
		beta_buffer[t] = local_buf[t];
	}
}

void Conv_Acc(int16_t* ifm, int16_t* ofm, int16_t* weight, int16_t* bias, uint32_t k_s_pad_ltype, uint32_t iofm_num, uint32_t ifm_w_h, uint32_t ofm_w_h,
	uint32_t TRTC, uint32_t TMTN, int OFM_num_bound, int mLoopsxTM, int16_t pad_val, uint32_t TRowTCol, uint32_t en_bits, int WeightQ, int BetaQ, int InputQ, int OutputQ)//enable_bits[2:0]={IsReLU, LoadBias, IsNotConv}
{
#pragma HLS INTERFACE s_axilite register port=WeightQ
#pragma HLS INTERFACE s_axilite register port=k_s_pad_ltype
#pragma HLS INTERFACE s_axilite register port=TRTC
#pragma HLS INTERFACE s_axilite register port=en_bits
#pragma HLS INTERFACE s_axilite register port=pad_val
#pragma HLS INTERFACE s_axilite register port=ifm_w_h
#pragma HLS INTERFACE s_axilite register port=ofm_w_h
#pragma HLS INTERFACE s_axilite register port=InputQ
#pragma HLS INTERFACE s_axilite register port=TRowTCol
#pragma HLS INTERFACE s_axilite register port=OFM_num_bound
#pragma HLS INTERFACE s_axilite register port=OutputQ
#pragma HLS INTERFACE s_axilite register port=TMTN
#pragma HLS INTERFACE s_axilite register port=iofm_num
#pragma HLS INTERFACE s_axilite register port=mLoopsxTM
#pragma HLS INTERFACE s_axilite register port=BetaQ
#pragma HLS INTERFACE s_axilite port=return

#pragma HLS INTERFACE m_axi depth=512 port=bias offset=slave bundle=BUS1 max_read_burst_length=128 max_write_burst_length=128
#pragma HLS INTERFACE m_axi depth=512 port=weight offset=slave bundle=BUS1 max_read_burst_length=128 max_write_burst_length=128
#pragma HLS INTERFACE m_axi depth=512 port=ifm offset=slave bundle=BUS0 max_read_burst_length=128 max_write_burst_length=128
#pragma HLS INTERFACE m_axi depth=512 port=ofm offset=slave bundle=BUS0 max_read_burst_length=128 max_write_burst_length=128
	uint16_t ifm_w = (ifm_w_h >> 16) & 0xFFFF;
	uint16_t ifm_h = ifm_w_h & 0xFFFF;
	uint16_t ofm_w = (ofm_w_h >> 16) & 0xFFFF;
	uint16_t ofm_h = ofm_w_h & 0xFFFF;
	uint16_t TR = (TRTC >> 16) & 0xFFFF;
	uint16_t TC = TRTC & 0xFFFF;
	uint16_t TM = (TMTN >> 16) & 0xFFFF;
	uint16_t TN = TMTN & 0xFFFF;
	uint16_t ifm_num = (iofm_num >> 16) & 0xFFFF;
	uint16_t ofm_num = iofm_num & 0xFFFF;
	uint8_t ksize = (k_s_pad_ltype >> 24) & 0xFF;
	uint8_t kstride = (k_s_pad_ltype >> 16) & 0xFF;
	uint8_t pad_int = (k_s_pad_ltype >> 8) & 0xFF;
	uint8_t ltype = k_s_pad_ltype & 0xFFFF;
	uint16_t TRow = (TRowTCol >> 16) & 0xFFFF;
	uint16_t TCol = TRowTCol & 0xFFFF;
	bool IsReLU = (en_bits >> 2) & 0x1;
	bool LoadBias = (en_bits >> 1) & 0x1;
	bool IsNotConv = en_bits & 0x1;

	assert((ofm_num > 0) && (ofm_num <= 2048));
	assert((ifm_num > 0) && (ifm_num <= 2048));
	assert((kstride > 0) && (kstride <= HW_S));
	assert((ksize > 0) && (ksize < 8));//maybe include some pool ops
	assert((ifm_w > 0) && (ifm_w <= 2048));
	assert((ifm_h > 0) && (ifm_h <= 2048));
	assert((ofm_w > 0) && (ofm_w <= 2048));
	assert((ofm_h > 0) && (ofm_h <= 2048));
	assert((pad_int >= 0) && (pad_int <= 4));//maybe
	assert((TM > 0) && (TM <= Tm));
	assert((TN >= 0) && (TN <= Tn));
	assert((TR > 0) && (TR <= Tr));
	assert((TC > 0) && (TC <= Tc));

	///////////////////////////////////
	const int InterSubBeta = OutputQ - BetaQ;
	const int WeightAddInputSubInter = WeightQ + InputQ - OutputQ;
	///const int InterSubOutput = INTERWIDTH - OutputQ;

	assert((InterSubBeta >= 0) && (InterSubBeta < 32));
	assert((WeightAddInputSubInter >= 0) && (WeightAddInputSubInter < 32));
	///assert((InterSubOutput >= 0)&&(InterSubOutput < 32));

	static int32_t ofm_buffer0[Tm][Tr][Tc];
#pragma HLS ARRAY_RESHAPE variable=ofm_buffer0 complete dim=1
	static int32_t ofm_buffer1[Tm][Tr][Tc];
#pragma HLS ARRAY_RESHAPE variable=ofm_buffer1 complete dim=1
	static int16_t bias_buffer[MAX_BETA_LENGTH];

	uint16_t IW_align_256b = ifm_w;

	uint16_t OW_align_256b = ofm_w;

	const int OHxOW = ofm_h * OW_align_256b;
	const int IHxIW = ifm_h * IW_align_256b;

	uint8_t K_1 = ksize - 1;
	/////////////////////////////////param

	if (LoadBias)
		copy_beta(bias_buffer, bias, ofm_num);
	//memcpy(bias_buffer, bias, ofm_num*sizeof(float));

	int tr, tc, tm;
	int TR_MIN, TC_MIN, TM_MIN;

	int m0[1], m1[1];
	int TM_MIN0[1], TM_MIN1[1];
	bool pingpongm;

	for (tr = 0; tr < ofm_h; tr += TR)
	{
#pragma HLS LOOP_TRIPCOUNT min=1 max=26
		TR_MIN = MIN_diy(TR, ofm_h - tr);
		for (tc = 0; tc < ofm_w; tc += TC)
		{
#pragma HLS LOOP_TRIPCOUNT min=1 max=26
			TC_MIN = MIN_diy(TC, ofm_w - tc);
			pingpongm = false;
			for (tm = 0; tm < OFM_num_bound; tm += TM)
			{
#pragma HLS LOOP_TRIPCOUNT min=1 max=32
				TM_MIN = MIN_diy(TM, ofm_num - tm);
				bool input_flag = (tm < mLoopsxTM);
				///bool process_flag = (tm > 0)&&(tm < mLoops_a1xTM);//CONV dont use, the same as input_flag
				bool write_flag = IsNotConv ? (tm > TM) : (tm > 0);
				int16_t cofst = tc * kstride - pad_int;
				int16_t Coffset_mmcpy = (cofst) < 0 ? 0 : cofst;
				if (!pingpongm)
				{
					load_compute_wrapper(ifm, weight, ofm_buffer1, bias_buffer, ksize, K_1, kstride, ifm_num, ifm_w, IW_align_256b, ifm_h, ofm_num,
						pad_int, ltype, TRow, TCol, IHxIW, TC_MIN, TR_MIN, TM_MIN, TM, TN, tm, tr, tc,
						m1, TM_MIN1, pingpongm, input_flag, pad_val, 1, InterSubBeta, WeightAddInputSubInter, Coffset_mmcpy, TR, TC);

					write_back_output_reorg(ofm_buffer0, ofm, bias_buffer, tr, tc, m0[0], OW_align_256b, TM_MIN0[0], TR_MIN, TC_MIN, OHxOW, IsReLU,
						write_flag, ltype, InterSubBeta, WeightAddInputSubInter);
					pingpongm = true;
				}
				else
				{
					load_compute_wrapper(ifm, weight, ofm_buffer0, bias_buffer, ksize, K_1, kstride, ifm_num, ifm_w, IW_align_256b, ifm_h, ofm_num,
						pad_int, ltype, TRow, TCol, IHxIW, TC_MIN, TR_MIN, TM_MIN, TM, TN, tm, tr, tc,
						m0, TM_MIN0, pingpongm, input_flag, pad_val, 1, InterSubBeta, WeightAddInputSubInter, Coffset_mmcpy, TR, TC);

					write_back_output_reorg(ofm_buffer1, ofm, bias_buffer, tr, tc, m1[0], OW_align_256b, TM_MIN1[0], TR_MIN, TC_MIN, OHxOW, IsReLU,
						write_flag, ltype, InterSubBeta, WeightAddInputSubInter);
					pingpongm = false;
				}
			}
		}
	}
}

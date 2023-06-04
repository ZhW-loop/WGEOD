#include "pool.h"
#include <iostream>
#include<stdio.h>
#include <cstring>
#include <assert.h>
using namespace std;
void Pool_Acc(trans_type* in,trans_type* out,
		int channel,int height_in,int width_in,
		int height_out,int width_out,int Kx,int Ky,int stride)
{
#pragma HLS INTERFACE m_axi depth=90000 port=out offset=slave max_read_burst_length=256 max_write_burst_length=256
#pragma HLS INTERFACE m_axi depth=90000 port=in offset=slave max_read_burst_length=256 max_write_burst_length=256

#pragma HLS INTERFACE s_axilite port=stride
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS INTERFACE s_axilite port=Ky
#pragma HLS INTERFACE s_axilite port=width_in
#pragma HLS INTERFACE s_axilite port=Kx
#pragma HLS INTERFACE s_axilite port=height_in
#pragma HLS INTERFACE s_axilite port=height_out
#pragma HLS INTERFACE s_axilite port=width_out
#pragma HLS INTERFACE s_axilite port=channel

	#pragma HLS DATAFLOW
	hls::stream<trans_type> stream_1D;
#pragma HLS STREAM variable=stream_1D depth=8 dim=1
	hls::stream<trans_type> stream_2D;
#pragma HLS STREAM variable=stream_2D depth=8 dim=1
	hls::stream<trans_type> stream_out;
#pragma HLS STREAM variable=stream_out depth=8 dim=1
	int width_in_tp = width_in, height_in_tp = height_in;
	if(Kx!=stride)
	{
		width_in_tp++;
		height_in_tp++;
	}
	mm2stream(in, stream_1D, height_in, width_in ,channel, BURST_LEN);
	pool_1D(stream_1D,stream_2D,channel,height_in,width_in_tp,Kx,stride);
	pool_2D(stream_2D,stream_out,channel,height_in_tp,width_out,Ky,stride);
	stream2mm(stream_out, out, height_out, width_out, channel, BURST_LEN);
}

void dma_in(trans_type* in, trans_type local_buf[BURST_LEN+1], int offset, bool enable, int len, int forhls, int* forhls_n)
{
//#pragma HLS INTERFACE m_axi depth=90000 port=in offset=slave max_read_burst_length=256 max_write_burst_length=256
	if(!enable) return;
	assert(len == (BURST_LEN));
	memcpy(local_buf, in+offset, len*sizeof(trans_type));

	*forhls_n = forhls;
}

void lbuf2stream(hls::stream<trans_type>& out, trans_type local_buf[BURST_LEN+1],  bool enable, int len, int forhls_n)
{
	if(!enable) return;
	for(int i=0;i<len+forhls_n;i++)
	{
#pragma HLS LOOP_TRIPCOUNT min=256 max=256 avg=256
#pragma HLS PIPELINE II=1
		out.write(local_buf[i]);
		//cout<<local_buf[i]<<endl;
	}
}

void mm2stream(trans_type* in, hls::stream<trans_type>& out, int height_in, int width_in, int channel, int len)
{
//#pragma HLS INTERFACE m_axi depth=90000 port=in offset=slave max_read_burst_length=256 max_write_burst_length=256
	static trans_type local_buf0[BURST_LEN+1];
	static trans_type local_buf1[BURST_LEN+1];
	bool pipe = true;
	int bound = height_in*width_in*channel;
	int loopcnts = bound + BURST_LEN;
	int forhls_0 = 0, forhls_1 = 0;
	int forhls = 0;
	for(int ofst=0; ofst<loopcnts; ofst+=len)
	{
#pragma HLS LOOP_TRIPCOUNT min=676 max=676 avg=676
		if(pipe)
		{
			dma_in(in, local_buf0, ofst, ofst<bound, len, forhls, &forhls_0);
			lbuf2stream(out, local_buf1, ofst!=0, len, forhls_1);
			pipe = false;
		}
		else
		{
			dma_in(in, local_buf1, ofst, ofst<bound, len, forhls, &forhls_1);
			lbuf2stream(out, local_buf0, ofst!=0, len, forhls_0);
			pipe = true;
		}
		//cout<< forhls<<endl;
	}
}

void stream2lbuf(hls::stream<trans_type>& in, trans_type local_buf[BURST_LEN+1], bool enable, int len, int forhls_n)
{
	if(!enable) return;
	for(int i=0;i<len+forhls_n;i++)
	{
#pragma HLS LOOP_TRIPCOUNT min=256 max=256 avg=256
#pragma HLS PIPELINE II=1
		local_buf[i] = in.read();
	}
}

void dma_out(trans_type local_buf[BURST_LEN+1], trans_type* out, int offset, bool enable, int len, int forhls, int* forhls_n)
{
	if(!enable) return;
	assert(len == (BURST_LEN));
	memcpy(out+offset, local_buf, len*sizeof(trans_type));
	*forhls_n = forhls;
}

void stream2mm(hls::stream<trans_type>& in, trans_type* out, int height_out, int width_out, int channel, int len)
{
//#pragma HLS INTERFACE m_axi depth=90000 port=out offset=slave max_read_burst_length=256 max_write_burst_length=256
	static trans_type local_buf0[BURST_LEN+1];
	static trans_type local_buf1[BURST_LEN+1];
	bool pipe = true;
	int bound = height_out*width_out*channel;
	int loopcnts = bound + BURST_LEN;
	int forhls_0 = 0, forhls_1 = 0;
	int forhls = 0;
	for(int ofst = 0; ofst<loopcnts; ofst+=BURST_LEN)
	{
#pragma HLS LOOP_TRIPCOUNT min=169 max=169
		if(pipe)
		{
			stream2lbuf(in, local_buf0, ofst<bound, len,forhls_1);
			dma_out(local_buf1, out, ofst-BURST_LEN, ofst!=0, len, forhls, &forhls_0);
			pipe = false;
		}
		else
		{
			stream2lbuf(in, local_buf1, ofst<bound, len, forhls_0);
			dma_out(local_buf0, out, ofst-BURST_LEN, ofst!=0, len, forhls, &forhls_1);
			pipe = true;
		}
	}

}

void pool_1D(hls::stream<trans_type> &in,hls::stream<trans_type> &out,int channel,int height_in,int width_in,int Kx,int stride)
{
//	#pragma HLS INTERFACE axis register both port=out
//	#pragma HLS INTERFACE axis register both port=in

	#pragma HLS INTERFACE ap_stable port=width_in
	#pragma HLS INTERFACE ap_stable port=Kx
	#pragma HLS INTERFACE ap_stable port=height_in
	#pragma HLS INTERFACE ap_stable port=channel

	trans_type dff;

	for(int i=0;i<height_in*channel;i++)
	{
		#pragma HLS LOOP_TRIPCOUNT min=6656 max=6656 avg=6656
		for(int j=0;j<width_in;j++)
		{
			#pragma HLS PIPELINE II=1
			#pragma HLS LOOP_TRIPCOUNT min=26 max=26 avg=26


			if(Kx==stride)
			{
				trans_type in_block = in.read();
				trans_type tp_out;
				if(j%Kx==0)
				tp_out=in_block;
				else
					tp_out=MAX(dff, in_block);
				if((j+1)%Kx==0)//if need output
					out.write(tp_out);
				else
					dff=tp_out;
			}
			else
			{
				trans_type in_block = 0;
				if(j!=width_in-1)
					in_block = in.read();
				//cout<<in_block<<" ";
				if(j==0) dff = in_block;
				else
				{
					trans_type tp_out = MAX(dff, in_block);
					dff = in_block;
					out.write(tp_out);
					//cout<<tp_out<<" ";
				}
			}

		}
	}
}

void pool_2D(hls::stream<trans_type> &in,hls::stream<trans_type> &out,int channel,int height_in,int width_out,int Ky,int stride)
{
//	#pragma HLS INTERFACE axis register both port=out
//	#pragma HLS INTERFACE axis register both port=in

	#pragma HLS INTERFACE ap_stable port=height_in
	#pragma HLS INTERFACE ap_stable port=Ky
	#pragma HLS INTERFACE ap_stable port=width_out
	#pragma HLS INTERFACE ap_stable port=channel

	static trans_type buf[POOL_2D_BUF_DEP];
	trans_type tp_in;

	for(int c=0;c<channel;c++)
	{
		#pragma HLS LOOP_TRIPCOUNT min=256 max=256 avg=256
		for(int i=0;i<height_in;i++)
		{
			#pragma HLS LOOP_TRIPCOUNT min=26 max=26 avg=26
			for(int j=0;j<width_out;j++)
			{
				#pragma HLS DEPENDENCE variable=buf inter false
				#pragma HLS PIPELINE
				#pragma HLS LOOP_TRIPCOUNT min=13 max=13 avg=13

				tp_in=buf[j];
				if(Ky==stride)
				{
					trans_type in_block=in.read();
					trans_type tp_out;
					if((i%Ky)==0)
						tp_out=in_block;
					else
						tp_out=MAX(tp_in, in_block);
					if((i+1)%Ky==0)
						out.write(tp_out);
					else
						buf[j]=tp_out;
				}
				else
				{
					trans_type in_block = 0;
					if(i!=height_in-1)
						in_block = in.read();
					//cout<<in_block<<" ";
					if(i==0) buf[j]=in_block;
					else
					{
						trans_type tp_out = MAX(tp_in, in_block);
						buf[j] = in_block;
						out.write(tp_out);
					}
				}
			}
		}
	}

}


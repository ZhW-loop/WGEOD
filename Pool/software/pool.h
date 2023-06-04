#ifndef __POOL_H__
#define __POOL_H__

//#include <iostream>
#include <hls_stream.h>
#include <stdint.h>
//#include <ap_int.h>
#define MAX(A,B) ((A>B)?A:B)

#define POOL_2D_BUF_DEP 256
#define BURST_LEN 64

typedef int16_t   trans_type;



void Pool_Acc(trans_type* in, trans_type* out,int channel,int height_in,int width_in,int height_out,int width_out,int Kx,int Ky,int stride);

void mm2stream(trans_type* in, hls::stream<trans_type>& out, int height_in, int width_in, int channel);
void dma_in(trans_type* in, trans_type local_buf[BURST_LEN+1], int offset, bool enable);
void lbuf2stream(trans_type local_buf[BURST_LEN+1], hls::stream<trans_type>& out, bool enable);

void pool_1D(hls::stream<trans_type> &in,hls::stream<trans_type> &out,int channel,int height_in,int width_in,int Kx,int stride);
void pool_2D(hls::stream<trans_type> &in,hls::stream<trans_type> &out,int channel,int height_in,int width_out,int Ky,int stride);

void stream2mm(hls::stream<trans_type>& in, trans_type* out, int height_out, int width_out, int channel);
void stream2lbuf(hls::stream<trans_type>& in, trans_type local_buf[BURST_LEN+1], bool enable);
void dma_out(trans_type local_buf[BURST_LEN+1], trans_type* out, int offset, bool enable);

#endif

#include "pool.h"
#include <iostream>
#include<stdio.h>
using namespace std;

void pool_soft(trans_type* in,trans_type* out,
		int channel,int height_in,int width_in,
		int height_out,int width_out,int Kx,int Ky,int stride);

int main(void)
{
	trans_type in_array[90000];
	trans_type out_array[90000];
	trans_type out_soft_array[90000];

	int channel=256;
	int height_in=10;
	int width_in=10;
	int Kx=2;
	int Ky=2;
	int stride=2;
	int width_out = 0, height_out = 0;
	if(Kx == stride)
	{
		width_out = (width_in - Kx) / stride + 1;
		height_out = (height_in - Ky) / stride + 1;
	}
	else
	{
		width_out = width_in;
		height_out = height_in;
	}
	for(int c=0;c<channel;c++)
		for(int i=0;i<height_in*width_in;i++)
		{
			in_array[c*height_in*width_in+i] = i;
		}

	Pool_Acc(in_array,out_array,channel,height_in,width_in,height_out,width_out,Kx,Ky,stride);
	cout<<"ok"<<endl;
	pool_soft(in_array,out_soft_array,channel,height_in,width_in,height_out,width_out,Kx,Ky,stride);
	cout<<"ok"<<endl;
	int flag=1;
	for(int i=0;i<height_out*width_out*channel;i++)
	{
		//cout<<out_array[i]<<" "<<out_soft_array[i]<<endl;
		if(out_array[i]!=out_soft_array[i])
			flag = 0;
	}

	if(flag==1)
		printf("match\n");
	else
		printf("mis-match\n");
}

void pool_soft(trans_type* in,trans_type* out,
		int channel,int height_in,int width_in,
		int height_out,int width_out,int Kx,int Ky,int stride)
{
	for(int m=0;m<channel;m++)
	{
		for(int i=0;i<height_out;i++)
		{
			for(int j=0;j<width_out;j++)
			{
				trans_type tp;
                tp=-32768;

				for(int ii=0;ii<Ky;ii++)
					for(int jj=0;jj<Kx;jj++)
					{
						int row=i*stride+ii;
						int col=j*stride+jj;
						trans_type dat = 0;
						if(row>=0&&row<height_in&&col>=0&&col<width_in)
							 dat=in[m*width_in*height_in+row*width_in+col];
						else
							 dat= 0;
                        tp=MAX(tp, dat);
					}
				out[m*width_out*height_out + i* width_out+ j] = tp;
			}
		}
	}
}

from pynq import Overlay
import numpy as np
from pynq import Xlnk
import time
import random


channel=256
width_in=26
height_in=26
Kx=2
Ky=2
stride=2
width_out=0
height_out=0
if Kx==stride:
    width_out = (width_in - Kx) // stride + 1
    height_out = (height_in - Ky) // stride + 1
else:
    width_out = width_in
    height_out = height_in

xlnk=Xlnk()
ol=Overlay("CNNALL.bit")
ol.download();
print(ol.ip_dict.keys())

input_buffer=xlnk.cma_array(shape=(channel*height_in*width_in),cacheable=0,dtype=np.int16)
output_buffer=xlnk.cma_array(shape=(channel*height_out*width_out),cacheable=0,dtype=np.int16)
output_buffer_soft=xlnk.cma_array(shape=(channel*height_out*width_out),cacheable=0,dtype=np.int16)

print(input_buffer.nbytes);
print(output_buffer.nbytes);

for i in range(channel*width_in*height_in):
    input_buffer[i]=random.randint(-2000,2000)

# for j in range(input_buffer.shape[1]):
#     for k in range(input_buffer.shape[2]):
#         print(input_buffer[0][j][k][0],end=' ');
#     print(' ')
'''   
for i in range(channel*width_out*height_out):
    output_buffer[i]=0
    output_buffer_soft[i]=0
'''
pool=ol.Pool_Acc_0

def Run_Pool(ch,kx,ky,height_in,width_in,height_out,width_out,feature_out,feature_in,stride):
    pool.write(0x10,feature_in.physical_address)
    pool.write(0x18,feature_out.physical_address)
    pool.write(0x20,ch);
    pool.write(0x28,height_in)
    pool.write(0x30,width_in)
    pool.write(0x38,height_out)
    pool.write(0x40,width_out)
    pool.write(0x48,kx)
    pool.write(0x50,ky)
    pool.write(0x58,stride)
    print("start")
    starttime = time.time()
    pool.write(0, (pool.read(0)&0x80)|0x01 ) #start pool IP
    tp=pool.read(0)
    while not((tp>>1)&0x1):
        tp=pool.read(0)
    endtime = time.time()
    print("Hardware run time=%s s"%(endtime-starttime))
    
def Run_Pool_Soft(ch,kx,ky,feature_in,feature_out,width_out,height_out,width_in,height_in,stride):
    for i in range(ch):
        for j in range(height_out):
            for k in range(width_out):
                tp=-32768;
                for ii in range(ky):
                    for jj in range(kx):
                        row=j*stride+ii
                        col=k*stride+jj
                        if row>=0 and row<height_in and col>=0 and col<width_in:
                            dat=feature_in[i*height_in*width_in+row*width_in+col]
                        else:
                            dat = 0
                        if(dat>tp):
                            tp=dat
                feature_out[i*width_out*height_out+j*width_out+k]=tp
    

Run_Pool(channel,Kx,Ky,height_in,width_in,height_out,width_out,output_buffer,input_buffer,stride)


starttime=time.time()
Run_Pool_Soft(channel,Kx,Ky,input_buffer,output_buffer_soft,width_out,height_out,width_in,height_in,stride)
endtime=time.time()
print("software run time=%s s"%(endtime-starttime))
    
flag=1
for i in range(channel*width_out*height_out):
    if(output_buffer[i]!=output_buffer_soft[i]):
        flag=0;
        print("output_buffer    [%d] = %d"%(i,output_buffer[i]));
        print("output_buffer_soft  [%d] = %d"%(i,output_buffer_soft[i]));
                    
if(flag==1):
    print("============================\n   result_match\n============================\n");
else:
    print("============================\n   result_mismatch\n============================\n");

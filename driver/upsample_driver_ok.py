from pynq import Overlay
import numpy as np
from pynq import Xlnk
import time
import random

channel=128
width_in=13
height_in=13
stride = 2
width_out = width_in * stride
height_out = height_in * stride

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
    
for i in range(channel*width_out*height_out):
    output_buffer[i]=0
    output_buffer_soft[i]=0

dma=ol.axi_dma_1
upsample = ol.Upsample_Acc_0

def Run_Upsample(channel,height_in,width_in,feature_out,feature_in,stride):
    upsample.write(0x10,channel);
    upsample.write(0x18,height_in)
    upsample.write(0x20,width_in)
    upsample.write(0x28,stride)
    starttime = time.time()
    upsample.write(0, (upsample.read(0)&0x80)|0x01 ) #start pool IP
    dma.recvchannel.transfer(feature_out)
    dma.sendchannel.transfer(feature_in)
    dma.sendchannel.wait()
    dma.recvchannel.wait()
    tp=upsample.read(0)
    while not((tp>>1)&0x1):
        tp=upsample.read(0)
    endtime = time.time()
    print("Hardware run time=%s s"%(endtime-starttime))
    

def Run_Upsample_Soft(channel,feature_in,feature_out,width_in,height_in,stride):
    for i in range(channel):
        for j in range(height_out):
            for k in range(width_out):
                row=j//stride
                col=k//stride
                dat=feature_in[i*height_in*width_in+row*width_in+col]
                feature_out[i*width_out*height_out+j*width_out+k]=dat
                
Run_Upsample(channel,height_in,width_in,output_buffer, input_buffer, stride)

starttime=time.time()
Run_Upsample_Soft(channel,input_buffer,output_buffer_soft,width_in,height_in,stride)
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

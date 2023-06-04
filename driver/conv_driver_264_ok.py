from pynq import Overlay
import numpy as np
from pynq import Xlnk
import time
import random

np.set_printoptions(threshold=np.inf)

ksize = 3
stride = 1
pad_int = 1
ifm_h = 416
ifm_w = 416
ifm_num = 3
ofm_h = 416
ofm_w = 416
ofm_num = 1
linear = 1
# hardware
Tn = 2
Tm = 64
Tr = 26
Tc = 26

xlnk=Xlnk()
ol=Overlay("CNNALL.bit")
ol.download();
print(ol.ip_dict.keys())

conv=ol.Conv_Acc_0

#hardware
ifm=xlnk.cma_array(shape=(ifm_w * ifm_h * ifm_num),cacheable=0,dtype=np.int16)
weight_rg=xlnk.cma_array(shape=(ofm_num * ifm_num * ksize * ksize),cacheable=0,dtype=np.int16)
ofm=xlnk.cma_array(shape=(ofm_w * ofm_h * ofm_num),cacheable=0,dtype=np.int16)
bias=xlnk.cma_array(shape=(1024),cacheable=0,dtype=np.int16)
#software
ofm_soft = np.zeros((ofm_w * ofm_h * ofm_num),dtype=np.int16)
weight = np.zeros((ofm_num * ifm_num * ksize * ksize),dtype=np.int16)

for i in range(ifm_w * ifm_h * ifm_num):
    ifm[i] = random.randint(-1000,1000)
    #ifm[i] = i 
for i in range(ofm_num * ifm_num * ksize * ksize):
    weight[i] = random.randint(-1000,1000)
    #weight[i] = i
for i in range(ofm_num):
    bias[i] = random.randint(-1000,1000)
def weight_rearrangement(weight, weight_rg, ofm_num, ifm_num, ksize):
    TN = min(ifm_num, Tn)
    TM = min(ofm_num, Tm)
    o_M = int(np.ceil(ofm_num / TM))
    i_N = int(np.ceil(ifm_num / TN))
    kk = ksize * ksize
    w_cnt = 0
    for t1 in range(o_M):
        for t2 in range(i_N):
            for k in range(ksize * ksize):
                for tm in range(TM):
                    for tn in range(TN):
                        rg_offset = t1 * i_N * kk * TM * TN + t2 * kk * TM * TN + k * TM * TN + tm * TN + tn
                        x1 = t1 * TM + tm
                        x2 = t2 * TN + tn
                        x3 = k
                        source_offset = x1 * ifm_num * kk + x2 * kk + x3
                        if x1 < ofm_num and x2 < ifm_num and x3 < kk:
                            weight_rg[w_cnt] = weight[source_offset]
                            w_cnt += 1

weight_rearrangement(weight, weight_rg, ofm_num, ifm_num, ksize)

def conv_soft(ifm_num, ifm_h, ifm_w, ofm_num, ofm_h, ofm_w, ksize, stride,
              ifm, inputQ, weight, weightQ, ofm, outputQ, bias, biasQ, pad_int):
    for i in range(ofm_num):
        for j in range(ofm_h):
            for k in range(ofm_w):
                sum = np.int64(0)
                for c in range(ifm_num):
                    for ii in range(ksize):
                        for jj in range(ksize):
                            row = j*stride-pad_int+ii
                            col = k*stride-pad_int+jj
                            if not (row<0 or col<0 or row>=ifm_h or col>=ifm_w):
                                dat = ifm[c*ifm_h*ifm_w+row*ifm_w+col]
                                wt = weight[i*ifm_num*ksize*ksize+c*ksize*ksize+ii*ksize+jj]
                                sum=sum+int(dat)*int(wt)
                res=(sum>>(inputQ+weightQ-outputQ)) + (bias[i] >> (biasQ-inputQ))
                if res>32767:
                    res = 32767
                else:
                    if res<-32768:
                        res = -32768
                ofm[i*ofm_h*ofm_w+j*ofm_w+k] = res
'''
def conv_acc(ifm, ofm, weight_rg, bias, ksize, stride, pad_int,
             ifm_num, ifm_h, ifm_w, ofm_num, ofm_h, ofm_w, linear):
    TN = np.uint16(min(ifm_num, Tn))
    TM = np.uint16(min(ofm_num, Tm))
    TR = np.uint16(min(ofm_h, Tr))
    TC = np.uint16(min(ofm_w, Tc))
    mLoops = np.int(np.ceil(ofm_num / float(TM)))
    mLoopsxTM = np.int(mLoops * np.int(TM))
    OFM_num_bound = np.int((mLoops + 1) * np.int(TM))
    TRow = np.uint16((TR - 1) * stride + ksize)
    TCol = np.uint16((TC - 1) * stride + ksize)
    TRowTCol = (TRow << 16) | TCol
    TRTC = np.uint32(TR << 16 | TC)
    TMTN = np.uint32(TM << 16 | TN)
    ofm_w = np.uint16(ofm_w)
    ofm_h = np.uint16(ofm_h)
    ofm_w_h = np.uint32((ofm_w << 16) | ofm_h)
    ifm_w = np.uint16(ifm_w)
    ifm_h = np.uint16(ifm_h)
    ifm_w_h = np.uint32((ifm_w << 16) | ifm_h)
    ifm_num = np.uint16(ifm_num)
    ofm_num = np.uint16(ofm_num)
    iofm_num = np.uint32((ifm_num << 16) | ofm_num)
    ksize = np.uint16(ksize)
    stride = np.uint16(stride)
    pad_int = np.uint16(pad_int)
    k_s_pad_l = np.uint32((ksize << 24) | (stride << 16) | (pad_int << 8) | 0)
    en_bits = 0x0
    en_bits = np.uint32(en_bits)
    en_bits |= 0x1 << 1
    if linear==0: en_bits |= 0x1 << 2
    conv.write(0x10, ifm)
    conv.write(0x18, ofm)
    conv.write(0x20, weight_rg)
    conv.write(0x28, bias)
    conv.write(0x30, k_s_pad_l)
    conv.write(0x38, iofm_num)
    conv.write(0x40, ifm_w_h)
    conv.write(0x48, ofm_w_h)
    conv.write(0x50, TRTC)
    conv.write(0x58, TMTN)
    conv.write(0x60, OFM_num_bound)
    conv.write(0x68, mLoopsxTM)
    conv.write(0x70, 0)
    conv.write(0x78, TRowTCol)
    conv.write(0x80, en_bits)
    conv.write(0x88, 8)
    conv.write(0x90, 8)
    conv.write(0x98, 8)
    conv.write(0xa0, 8)
    starttime=time.time()
    conv.write(0, (conv.read(0)&0x80)|0x01 )
    tp=conv.read(0)
    while not((tp>>1)&0x1):
        tp=conv.read(0)
    endtime=time.time()
    print("Hardware run time=%s s"%(endtime-starttime))
'''
def conv_acc(ifm, ofm, weight_rg, bias, ksize, stride, pad_int,
             ifm_num, ifm_h, ifm_w, ofm_num, ofm_h, ofm_w, linear):
    TN = np.uint16(min(ifm_num, Tn))
    TM = np.uint16(min(ofm_num, Tm))
    TR = np.uint16(min(ofm_h, Tr))
    TC = np.uint16(min(ofm_w, Tc))
    mLoops = np.int(np.ceil(ofm_num / float(TM)))
    mLoopsxTM = np.int(mLoops * np.int(TM))
    OFM_num_bound = np.int((mLoops + 1) * np.int(TM))
    TRow = np.uint16((TR - 1) * stride + ksize)
    TCol = np.uint16((TC - 1) * stride + ksize)
    TRowTCol = np.int((TRow << 16) | TCol)
    TRTC = np.int(TR << 16 | TC)
    TMTN = np.int(TM << 16 | TN)
    ofm_w = np.uint16(ofm_w)
    ofm_h = np.uint16(ofm_h)
    ofm_w_h = np.int((ofm_w << 16) | ofm_h)
    ifm_w = np.uint16(ifm_w)
    ifm_h = np.uint16(ifm_h)
    ifm_w_h = np.int((ifm_w << 16) | ifm_h)
    ifm_num = np.int16(ifm_num)
    ofm_num = np.int16(ofm_num)
    iofm_num = np.int((ifm_num << 16) | ofm_num)
    ksize = np.uint16(ksize)
    stride = np.uint16(stride)
    pad_int = np.uint16(pad_int)
    k_s_pad_l = np.int((ksize << 24) | (stride << 16) | (pad_int << 8) | 0)
    en_bits = 0x0
    en_bits = np.int(en_bits)
    en_bits |= 0x1 << 1
    if linear==0: en_bits |= 0x1 << 2
    conv.write(0x10, ifm)
    conv.write(0x18, ofm)
    conv.write(0x20, weight_rg)
    conv.write(0x28, bias)
    conv.write(0x30, k_s_pad_l)
    conv.write(0x38, iofm_num)
    conv.write(0x40, ifm_w_h)
    conv.write(0x48, ofm_w_h)
    conv.write(0x50, TRTC)
    conv.write(0x58, TMTN)
    conv.write(0x60, OFM_num_bound)
    conv.write(0x68, mLoopsxTM)
    conv.write(0x70, 0)
    conv.write(0x78, TRowTCol)
    conv.write(0x80, en_bits)
    conv.write(0x88, 0)
    conv.write(0x90, 0)
    conv.write(0x98, 0)
    conv.write(0xa0, 0)
    starttime=time.time()
    conv.write(0, (conv.read(0)&0x80)|0x01 )
    tp=conv.read(0)
    while not((tp>>1)&0x1):
        tp=conv.read(0)
    endtime=time.time()
    print("Hardware run time=%s s"%(endtime-starttime))

conv_acc(ifm.physical_address, ofm.physical_address, weight_rg.physical_address,bias.physical_address,
        ksize, stride, pad_int, ifm_num, ifm_h, ifm_w, ofm_num, ofm_h, ofm_w, linear)

starttime=time.time()
conv_soft(ifm_num, ifm_h, ifm_w, ofm_num, ofm_h, ofm_w, ksize, stride,
          ifm, 0, weight, 0, ofm_soft, 0, bias, 0, pad_int)
endtime=time.time()
print("Software run time=%s s"%(endtime-starttime))
flag = 1

for c in range(ofm_num):
    for i in range(ofm_h):
        for j in range(ofm_w):
           if(ofm[c*ofm_h*ofm_w + i*ofm_w + j] != ofm_soft[c*ofm_h*ofm_w + i*ofm_w + j]):
                flag=0
                print("Out_  [%d][%d][%d]=%d"%(c,i,j,ofm[c*ofm_h*ofm_w + i*ofm_w + j]))
                print("Out_Soft[%d][%d][%d]=%d"%(c,i,j,ofm_soft[c*ofm_h*ofm_w + i*ofm_w + j]))

if(flag==1):
    print("============================\n   result_match\n============================\n")
else:
    print("============================\n   result_mismatch\n============================\n")
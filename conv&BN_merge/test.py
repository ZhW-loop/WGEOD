import numpy
import torch
import torch.nn as nn
import  numpy as np
from tool.darknet2pytorch import Darknet
WEIGHTS = Darknet("yolov3-tiny.cfg",inference=False)
WEIGHTS.load_weights("yolov3-tiny.weights")

torch.save(WEIGHTS, "./yolov3_tiny_model.pth")

model = torch.load("./yolov3_tiny_model.pth")

print(model)
'''
for i in model.named_parameters():
    print(i)
print(model)
'''

'''
#-----------------------conv&BN merge beta---------------------
layer = [0, 2, 4, 6, 8, 10, 12, 13, 14, 18, 21]
cnt = 1
for i in layer:
    weight = (model.models[i][1].weight).detach().numpy()
    bias = (model.models[i][1].bias).detach().numpy()
    mean = np.array(model.models[i][1].running_mean)
    var = np.array(model.models[i][1].running_var)
    beta = -mean*weight / pow(var, 0.5) + bias
    beta_16 = (np.int16)(np.round(beta*256.0))
    if cnt==10: cnt += 1
    filepath = "./bias_int16/conv" + str(cnt) + "_bias.txt"
    cnt += 1
    numpy.savetxt(filepath, beta_16, fmt='%d', delimiter='\n')
#-----------------------conv&BN merge beta---------------------
'''

'''
#-----------------------conv&BN merge weight---------------------
layer = [0, 2, 4, 6, 8, 10, 12, 13, 14, 18, 21]
cnt = 1
for i in layer:
    weight = (model.models[i][1].weight).detach().numpy()
    var = np.array(model.models[i][1].running_var)
    weight_conv = (model.models[i][0].weight).detach().numpy()
    to_file = np.array([])
    for i in range(weight_conv.shape[0]):
        weight_merge = np.int16(np.round(256* weight[i] * (weight_conv[i].reshape(-1)) / pow(var[i], 0.5)))
        to_file = np.append(to_file, weight_merge)
    if cnt == 10: cnt += 1
    filepath = "./weight_int16/conv" + str(cnt) + "_weight.txt"
    cnt += 1
    numpy.savetxt(filepath, to_file, fmt='%d', delimiter='\n')
#-----------------------conv&BN merge weight---------------------
'''


#-----------------------pure conv weight--------------------------
layer = [15, 22]
cnt = 10
for i in layer:
    weight_conv = (model.models[i][0].weight).detach().numpy()
    to_file = np.int16(np.round((weight_conv.reshape(-1) * 256)))
    filepath = './weight_int16/conv' + str(cnt) + '_weight.txt'
    cnt = 13
    numpy.savetxt(filepath, to_file, fmt='%d', delimiter='\n')
#-----------------------pure conv weight--------------------------

'''
#-----------------------test BN---------------------------
with torch.no_grad():
    m = nn.BatchNorm2d(3)
    m.eval()
    for i in m.parameters():
        print(i)
    #print(m.weight)
    #print(m.bias)

    input = torch.randn(1, 3, 2, 2)
    print(input)

    print(m.running_mean)
    print(m.running_var)

    output = m(input)

    print(m.running_mean)
    print(m.running_var)

    print(output)
#-----------------------test BN---------------------------
'''

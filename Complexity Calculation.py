import torch
from thop import profile
from thop import clever_format
# from test import DBCNN
from models.DEFNet_S import fusion_model_S
# import torchstat as ts
import time
import numpy as np
from PIL import Image
from torchvision import transform



if __name__ == '__main__':
    net1 = fusion_model_S().cuda()
    input11 = torch.randn(1, 3, 480, 640).cuda()
    input22 = torch.randn(1, 3, 480, 640).cuda()
    input = [input22, input11]
    flops, params = profile(net1, inputs=(input,  ))
    t_all = []

    for i in range(100):
        t1 = time.time()
        y = net1(input,  )
        t2 = time.time()
        t_all.append(t2 - t1)

    print('average time:', np.mean(t_all) / 1)
    print('average fps:', 1 / np.mean(t_all))
    # flops, params = profile(net1, inputs=(input))
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')


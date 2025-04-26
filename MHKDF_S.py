import torch
import torch.nn as nn
import os

from utils.tensor_ops import cus_sample, upsample_add
from utils.DSConv   import DSConv
from backbone.mix_transformer import (mit_b5,mit_b5_d,mit_b3,mit_b3_d,mit_b0_d,mit_b0,mit_b2_d,mit_b2)
from module.MyModules import (
    MSIM_S,
    Decoder,
    MSFHFM_S,
    SFFM,
)
import warnings
warnings.filterwarnings("ignore")


def load_pretrain_S(net):
    # dir_path = os.getcwd()
    pretrain_path = r'mit_b0.pth'
    print("Pretrain_path:", pretrain_path)
    net_dict = net.state_dict()

    pretrain_dict = torch.load(pretrain_path)

    dict = {k: v for k, v in pretrain_dict.items() if k in net_dict}
    net_dict.update(dict)
    net.load_state_dict(net_dict,strict=False)
    print('mit')
    return net

class BasicConv2d(nn.Module):
    def __init__(
        self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
    ):
        super(BasicConv2d, self).__init__()

        self.basicconv = nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.basicconv(x)


class MHKDF_S(nn.Module):
    def __init__(self, pretrained=True):
        super(MHKDF_S, self).__init__()
        self.upsample_add = upsample_add
        self.upsample = cus_sample
        seg_r = mit_b0()
        seg_r = load_pretrain_S(seg_r)
        seg_d = mit_b0_d()##
        seg_d = load_pretrain_S(seg_d)

        self.layersR = seg_r
        self.layersD = seg_d


        self.trans8 = nn.Conv2d(256, 64, 1)##
        self.trans4 = nn.Conv2d(160, 64, 1)
        self.trans2 = nn.Conv2d(64, 64, 1)
        self.trans1 = nn.Conv2d(32, 32, 1)

        self.t_trans2 = MSFHFM_S(64,32)
        # self.t_trans1 = SDFM(32,64)
        self.t_trans4 = SFFM(160, 64)
        # self.t_trans8 = SFFM(256, 64)
        self.upconv16 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv8 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv4 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv2 = BasicConv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.upconv1 = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.upconv11 = nn.Conv2d(32, 64, kernel_size=1)
        self.upconv22 = nn.Conv2d(64, 32, kernel_size=1)



        self.selfdc_8 = MSIM_S(64, 64)
        self.selfdc_4 = MSIM_S(64, 64)
        self.selfdc_2 = MSIM_S(32,32)
        self.selfdc_1 = MSIM_S(32,32)





        self.fdm = Decoder()



    def forward(self, RGBT):
        in_data = RGBT[0]
        in_depth = RGBT[1]
        in_data_d = self.layersD(in_depth)
        dataD_2 = in_data_d[1]
        dataD_4 = in_data_d[2]
        in_data_r = self.layersR(in_data,dataD_4)  #####64,128,320,512
        dataR_1= in_data_r[0]
        dataR_2 = in_data_r[1]
        dataR_4 = in_data_r[2]
        dataR_8 = in_data_r[3]

        in_data_8_r=self.trans8(dataR_8)
        in_data_4_r = self.trans4(dataR_4)

        in_data_2_r = self.trans2(dataR_2)
        in_data_1_r = self.trans1(dataR_1)

        in_data_2_aux = self.t_trans2(dataD_2,dataR_2)#+self.upconv22(in_data_2_dd)
        in_data_4_aux = self.t_trans4(dataD_4,dataR_4)
        in_data_8_aux = in_data_8_r#+in_data_8_d

        out_data_8 = in_data_8_r
        out_data_8 = self.upconv8(out_data_8)  # 1024

        out_data_4 = self.upsample_add(self.selfdc_8(out_data_8, in_data_8_aux), in_data_4_r)
        out_data_4 = self.upconv4(out_data_4)

        out_data_2 = self.upsample_add(self.selfdc_4(out_data_4, in_data_4_aux),in_data_2_r)
        out_data_2 = self.upconv2(out_data_2)# 64

        out_data_1 = self.upsample_add(self.selfdc_2(out_data_2, in_data_2_aux),in_data_1_r)
        out_data_1 = self.upconv1(out_data_1)#+ self.upconv22(in_data_1_aux)

        out_data = self.fdm(out_data_1, out_data_2, out_data_4, out_data_8)
        out_data_m = [out_data_1, out_data_2, out_data_4, out_data_8]
        return out_data,out_data_m


def fusion_model_S():
    model = MHKDF_S()
    return model

if __name__ == "__main__":
    model = MHKDF_S()
    x = torch.randn(2,3,256,256)
    depth = torch.randn(2,3,256,256)
    fuse = model([x,depth])
    print(fuse.shape)

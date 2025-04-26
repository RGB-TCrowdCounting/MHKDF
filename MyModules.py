import torch
from torch import nn
import math
# from mamba_simple import Mamba
# import causal_conv1d_cuda
from einops import rearrange, repeat
from module.BaseBlocks import BasicConv2d,BasicODConv2d,BasicDSConvConv2d
from utils.functions import PatchEmbeding
from utils.ODCONV import ODConv2d
from utils.functions import Attention_split
from pytorch_wavelets import DTCWTForward, DTCWTInverse
from utils.functions import transform
from utils.DSConv   import DSConv
from torch.nn import functional as F
from backbone.mix_transformer import Attention
from torch.cuda.amp import custom_bwd, custom_fwd
import warnings
warnings.filterwarnings("ignore")



class Enhanced_P(nn.Module):
    def __init__(self, scale_factor,inc,outc):
        super(Enhanced_P, self).__init__()
        self.scale_factor=scale_factor
        # self.inc=inc
        # self.outc=outc
        self.basicupsample = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='nearest'),
            nn.Conv2d(inc,outc,kernel_size=1),
            nn.BatchNorm2d(outc),
            nn.ReLU()
        )
    def forward(self, x,y):
        x=self.basicupsample(x)
        y=x+y
        return y

class Enhanced_S(nn.Module):
    def __init__(self, scale_factor):
        super(Enhanced_S, self).__init__()
        self.scale_factor=scale_factor
    def forward(self, x,y):
        x=F.upsample_bilinear(x, scale_factor=self.scale_factor)
        y=x+y


class DenseLayer4(nn.Module):
    def __init__(self, in_C, out_C, down_factor=4, k=4):######k原来为4
        super(DenseLayer4, self).__init__()
        self.k = k
        self.down_factor = down_factor
        mid_C = out_C // self.down_factor

        self.down = nn.Conv2d(in_C, mid_C, 1)

        self.denseblock = nn.ModuleList()
        for i in range(1, self.k + 1):
            self.denseblock.append(BasicConv2d(mid_C * i, mid_C, 3, 1, 1))

        self.fuse = BasicConv2d(in_C + mid_C, out_C, kernel_size=3, stride=1, padding=1)

    def forward(self, in_feat):
        down_feats = self.down(in_feat)
        out_feats = []
        for denseblock in self.denseblock:
            feats = denseblock(torch.cat((*out_feats, down_feats), dim=1))
            out_feats.append(feats)
        feats = torch.cat((in_feat, feats), dim=1)
        return self.fuse(feats)

class DenseLayer2(nn.Module):
    def __init__(self, in_C, out_C, down_factor=4, k=3):######k原来为4
        super(DenseLayer2, self).__init__()
        self.k = k
        self.down_factor = down_factor
        mid_C = out_C // self.down_factor

        self.down = nn.Conv2d(in_C, mid_C, 1)

        self.denseblock = nn.ModuleList()
        for i in range(1, self.k + 1):
            self.denseblock.append(BasicConv2d(mid_C*i, mid_C, 3, 1, 1))

        self.fuse = BasicConv2d(in_C + mid_C, out_C, kernel_size=3, stride=1, padding=1)

    def forward(self, in_feat):
        down_feats = self.down(in_feat)
        out_feats = []
        for denseblock in self.denseblock:
            feats = denseblock(torch.cat((out_feats, down_feats), dim=1))
            out_feats.append(feats)
        feats = torch.cat((in_feat, feats), dim=1)
        return self.fuse(feats)

class MSFHFM-T(nn.Module):
    def __init__(self, in_C, out_C):
        super(MSFHFM-T, self).__init__()
        down_factor = in_C // out_C

        self.DWT = DTCWTForward(J=1, biort='near_sym_b', qshift='qshift_b')
        self.IWT = DTCWTInverse(biort='near_sym_b', qshift='qshift_b')

        self.fuse_down_mul = BasicConv2d(in_C, in_C, 3, 1, 1)
        self.fuse_down_mul2 = BasicConv2d(2 * in_C, out_C, 3, 1, 1)
        # self.res_main = DenseLayer4(in_C, in_C, down_factor=down_factor)
        # self.fuse_main = BasicConv2d(2in_C, 2in_C, kernel_size=3, stride=1, padding=1)
        self.fuse_main = BasicConv2d(in_C, out_C, kernel_size=1)
        self.fuse_main1 = BasicConv2d(out_C, out_C, kernel_size=1)
        self.fuse_main12 = BasicConv2d(2 * out_C, out_C, kernel_size=1)
        self.fuse_main2 = BasicConv2d(out_C, out_C, kernel_size=3, stride=1, padding=1)
        self.sfkong = nn.Softmax(dim=1)
        self.sf = nn.Softmax(dim=1)
        # self.Graph = Graph_Attention_Union(out_C, out_C)

    def forward(self, rgb, depth):
        assert rgb.size() == depth.size()
        ##空域
        cat=torch.cat([rgb,depth],dim=1)
        f1=self.fuse_down_mul2(cat)*self.fuse_main(rgb)
        f2=self.fuse_main2(f1+self.fuse_main(depth))
        f3=self.fuse_down_mul2(cat)+f1+f2+self.fuse_main(rgb)
        f3=f3*f3+f3
        f3=f3*self.fuse_main(rgb)+f3+self.fuse_main(rgb)+self.fuse_main(depth)
        ##频域
        r = self.fuse_main(rgb)
        d = self.fuse_main(depth)
        rl, rh = self.DWT(r)
        dl, dh = self.DWT(d)
        fl = self.fuse_main2(rl + dl + rl * dl + self.fuse_main12(torch.cat([rl, dl], dim=1)))
        # fd = self.fuse_main2(rh[0] + dh[0] + dh[0] * dh[0] + self.fuse_main12(torch.cat([dh[0], dh[0]], dim=1)))
        # fd=[rh[0]+dh[0],rh[1]+dh[1],rh[2]+dh[2]]
        fd = [rh[0] + dh[0]]
        ff = self.IWT((fl, fd))
        rd = r + d
        feat = self.sf(rd) * f3 + self.sf(rd) * ff

        return feat

class MSFHFM-S(nn.Module):
    def __init__(self, in_C, out_C):
        super(MSFHFM-S, self).__init__()
        down_factor = in_C//out_C

        self.DWT = DTCWTForward(J=1, biort='near_sym_b', qshift='qshift_b')
        self.IWT = DTCWTInverse(biort='near_sym_b', qshift='qshift_b')

        self.fuse_down_mul = BasicConv2d(in_C, in_C, 3, 1, 1)
        self.fuse_down_mul2 = BasicConv2d(2*in_C, out_C, 3, 1, 1)
        # self.res_main = DenseLayer4(in_C, in_C, down_factor=down_factor)
        # self.fuse_main = BasicConv2d(2in_C, 2in_C, kernel_size=3, stride=1, padding=1)
        self.fuse_main = BasicConv2d(in_C, out_C, kernel_size=1)
        self.fuse_main1 = BasicConv2d(out_C,out_C,kernel_size=1)
        self.fuse_main12 = BasicConv2d(2*out_C, out_C, kernel_size=1)
        self.fuse_main2 = BasicConv2d(out_C, out_C, kernel_size=3, stride=1, padding=1)
        self.sfkong=nn.Softmax(dim=1)
        self.sf = nn.Softmax(dim=1)
        # self.Graph = Graph_Attention_Union(out_C, out_C)

    def forward(self, rgb, depth):
        assert rgb.size() == depth.size()
        cat=torch.cat([rgb,depth],dim=1)
        f1=self.fuse_down_mul2(cat)*self.fuse_main(rgb)
        f2=self.fuse_main2(f1+self.fuse_main(depth))
        f3=self.fuse_down_mul2(cat)+f1+f2+self.fuse_main(rgb)
        f3=f3*f3+f3
        f3=f3*self.fuse_main(rgb)+f3
        feat =f3

        return feat

class SFFM(nn.Module):
    def __init__(self, in_C, out_C):
        super(SFFM, self).__init__()
        down_factor = in_C//out_C
        self.dc = nn.Conv2d( out_C, 1, kernel_size=1)
        self.uc = nn.Conv2d(1,out_C, kernel_size=1)
        self.fuse_mainr = nn.Conv2d(in_C, out_C, kernel_size=1)
        self.fuse_maind = nn.Conv2d(in_C, out_C, kernel_size=1)
        self.fuse_main1 = BasicConv2d(out_C,out_C,kernel_size=1)
        self.fuse_main12 = BasicConv2d(2*out_C, out_C, kernel_size=1)
        self.fuse_main2 = BasicConv2d(out_C, out_C, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.con1 = nn.Conv2d(out_C, out_C, kernel_size=1, stride=1,bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, rgb, depth):
        assert rgb.size() == depth.size()
        r=self.fuse_mainr(rgb)
        d=self.fuse_maind(depth)

        avgoutr = torch.mean(r, dim=1, keepdim=True)
        maxoutr, _r = torch.max(d, dim=1, keepdim=True)
        avgoutd = torch.mean(r, dim=1, keepdim=True)
        maxoutd, _d = torch.max(d, dim=1, keepdim=True)
        outr=self.sigmoid(avgoutr+avgoutd)*(self.dc(d)+avgoutr+avgoutd)
        outd=self.sigmoid(maxoutr+maxoutd)*(self.dc(r)+maxoutr+maxoutd)
        xx=self.uc(outr)+self.uc(outd)+r+d
        feat1=self.softmax(r*d)
        feat2=feat1*xx+xx+r+d

        feat=feat2

        return feat

class HF(nn.Module):
    def __init__(self,in_C, out_C):
        super(HF, self).__init__()
        # self.attend_softmax = nn.Softmax(dim=-1)
        # self.maxpool=nn.MaxPool2d(3, 1, padding=1)
        # self.avgpool=nn.AvgPool2d(3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_C, out_C, kernel_size=3, padding=1)
        # self.conv1 = nn.Conv2d(in_C2, out_C, kernel_size=1)

    def forward(self, RGB, depth):
        assert RGB.size() == depth.size()
        rd=torch.cat([RGB,depth],dim=1)
        rd=self.conv3(rd)+RGB+depth

        # feature_avgpool = featurecsr_avgpool  feature_avgpool + feature_avgpool
        # feature_conv=self.conv(torch.cat([feature_maxpool,feature_avgpool],dim=1))
        feat = rd
        return feat



class Resudiual(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Resudiual, self).__init__()
        self.conv = BasicConv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv(x)
        x1 = self.sigmoid(x1)
        out = x1*x
        return out


class Tdc3x3_1(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Tdc3x3_1, self).__init__()
        self.conv1 = BasicConv2d(in_planes=in_channel, out_planes=out_channel, kernel_size=1)
        self.conv2 = BasicConv2d(in_planes=out_channel, out_planes=out_channel, kernel_size=3, dilation=1, padding=1)
        # self.conv2 = ODConv2d(in_planes=out_channel, out_planes=out_channel, kernel_size=3, dilation=1, padding=1)
        self.conv3 = BasicConv2d(in_planes=out_channel, out_planes=out_channel, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        # x1 = x
        x2 = self.conv2(x)
        x3 = x1 + x2
        x4 = self.conv3(x3)
        return x3, x4


class Tdc3x3_3(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Tdc3x3_3, self).__init__()
        self.conv1 = BasicConv2d(in_planes=in_channel, out_planes=out_channel, kernel_size=1)
        self.conv2 = BasicConv2d(in_planes=out_channel, out_planes=out_channel, kernel_size=3, dilation=2, padding=2)
        self.conv3 = BasicConv2d(in_planes=out_channel, out_planes=out_channel, kernel_size=1)
        self.residual = Resudiual(in_channel, out_channel)

    def forward(self, x, y):
        x1 = self.conv1(x)
        # x1 = x
        y = self.residual(y)
        x2 = self.conv2(x1 + y)
        x3 = x1 + x2
        x4 = self.conv3(x3)
        return x3, x4


class Tdc3x3_5(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Tdc3x3_5, self).__init__()
        self.conv1 = BasicConv2d(in_planes=in_channel, out_planes=out_channel, kernel_size=1)
        self.conv2 = BasicConv2d(in_planes=out_channel, out_planes=out_channel, kernel_size=3, dilation=4, padding=4)
        self.conv3 = BasicConv2d(in_planes=out_channel, out_planes=out_channel, kernel_size=1)
        self.residual = Resudiual(in_channel, out_channel)

    def forward(self, x, y):
        x1 = self.conv1(x)
        # x1 = x
        y = self.residual(y)
        x2 = self.conv2(x1 + y)
        x3 = x1 + x2
        x4 = self.conv3(x3)
        return x3,x4

class Tdc3x3_8(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Tdc3x3_8, self).__init__()
        self.conv1 = BasicConv2d(in_planes=in_channel, out_planes=out_channel, kernel_size=1)
        self.conv2 = BasicConv2d(in_planes=out_channel, out_planes=out_channel, kernel_size=3, dilation=8, padding=8)
        self.conv3 = BasicConv2d(in_planes=out_channel, out_planes=out_channel, kernel_size=1)
        self.residual = Resudiual(in_channel, out_channel)

    def forward(self, x, y):
        x1 = self.conv1(x)
        # x1 = x
        y = self.residual(y)
        x2 = self.conv2(x1 + y)
        x3 = x1 + x2
        x4 = self.conv3(x3)
        return x3,x4

class MSIM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MSIM, self).__init__()
        self.one = Tdc3x3_1(in_channel, out_channel)
        self.two = Tdc3x3_3(in_channel, out_channel)
        self.three = Tdc3x3_5(in_channel, out_channel)
        self.four = Tdc3x3_8(in_channel, out_channel)
        self.fusion = BasicConv2d(out_channel, out_channel, 1)
        self.fusion1 = BasicConv2d(out_channel * 7, out_channel, 1)
        self.ODC = BasicODConv2d(out_channel, out_channel, kernel_size=1)
        self.dcc = BasicConv2d(out_channel + 1, out_channel, 1)
        self.uc = BasicConv2d(1, out_channel, 1)
        self.sigmoid = nn.Sigmoid()
        self.Graph = Graph_Attention_Union(out_channel, out_channel)

    def forward(self, rgb, rgb_aux):
        x1, x2 = self.one(rgb_aux)
        x3, x4 = self.two(rgb_aux, x1)
        x5, x6 = self.three(rgb_aux, x3)
        xx7, x7 = self.four(rgb_aux, x5)
        ####
        xx = x1 + x3 + x5 + xx7
        xxod = self.ODC(xx) + xx
        x2 = xxod + x2
        x4 = xxod + x4
        x6 = xxod + x6
        x7 = xxod + x7
        ####
        x2 = x2 * rgb
        x4 = x4 * rgb
        # x6=x6*rgb
        # x7= x7*rgb
        # x5 = x5 * rgb
        x2_1 = x2 - x4
        x4_1 = x4 - x6
        x6_1 = x6 - x7
        x2_1 = self.Graph(x2, x4) + x2_1
        x4_1 = self.Graph(x4, x6) + x4_1
        x6_1 = self.Graph(x6, x7) + x6_1
        out = self.fusion1(torch.cat([x2, x4, x6, x7, x2_1, x4_1, x6_1], dim=1)) + xxod
        out = self.fusion(torch.abs(out - rgb))
        return out

class BasicUpsample(nn.Module):
    def __init__(self,scale_factor):
        super(BasicUpsample, self).__init__()

        self.basicupsample = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor,mode='nearest'),
            nn.Conv2d(32,32,kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

    def forward(self,x):
        return self.basicupsample(x)

class BasicUpsample_L(nn.Module):
    def __init__(self,scale_factor):
        super(BasicUpsample_L, self).__init__()

        self.basicupsample = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor,mode='nearest'),
            # nn.Conv2d(128,32,kernel_size=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU()
        )

    def forward(self,x):
        return self.basicupsample(x)



class Decoder(nn.Module):
    def __init__(self,):
        super(Decoder, self).__init__()
        self.basicconv1 = BasicDSConvConv2d(in_planes=64,out_planes=32,kernel_size=1)
        self.basicconv2 = BasicDSConvConv2d(in_planes=32,out_planes=32,kernel_size=1)
        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='nearest'),
            DSConv(32,32,1),
            nn.ReLU()
        )
        self.basicconv3 = BasicDSConvConv2d(in_planes=32,out_planes=32,kernel_size=3,stride=1,padding=1)
        self.basicconv4 = BasicDSConvConv2d(in_planes=64,out_planes=32,kernel_size=3,stride=1,padding=1)
        self.basicconv11 = BasicDSConvConv2d(in_planes=32, out_planes=32, kernel_size=1)
        self.basicupsample16 = BasicUpsample(scale_factor=16)
        self.basicupsample8 = BasicUpsample(scale_factor=8)
        self.basicupsample4 = BasicUpsample(scale_factor=4)
        self.basicupsample2 = BasicUpsample(scale_factor=2)
        self.basicupsample1 = BasicUpsample(scale_factor=1)
        self.sg=nn.Sigmoid()

        self.reg_layer = nn.Sequential(
            DSConv(128,64,kernel_size=3,stride=2,padding=1),
            # nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),#消融
            nn.BatchNorm2d(64),
            nn.ReLU(),
            DSConv(64,32,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            DSConv(32,16,1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            DSConv(16,1,kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            )


    def forward(self,out_data_1,out_data_2,out_data_4,out_data_8):
        out_data_8 = self.basicconv1(out_data_8)
        out_data_8 = self.basicconv3(out_data_8)+out_data_8

        out_data_4 = self.basicconv1(out_data_4)
        out_data_4 = torch.cat([out_data_4,self.upsample1(out_data_8)],dim=1)
        out_data_44 = self.basicconv4(out_data_4)
        out_data_4_1=self.basicconv11(out_data_44)
        out_data_4 =self.sg(out_data_4_1)*out_data_44+(1-self.sg(out_data_4_1))*out_data_44

        out_data_2 = self.basicconv2(out_data_2)
        out_data_2 = torch.cat([out_data_2,self.upsample1(out_data_4)],dim=1)
        out_data_22 = self.basicconv4(out_data_2)
        out_data_2_1 = self.basicconv11(out_data_22)
        out_data_2 = self.sg(out_data_2_1) * out_data_22 + (1 - self.sg(out_data_2_1)) * out_data_22


        out_data_1 = self.basicconv2(out_data_1)
        out_data_1 = torch.cat([out_data_1,self.upsample1(out_data_2)],dim=1)
        out_data_11 = self.basicconv4(out_data_1)
        out_data_1_1 = self.basicconv11(out_data_11)
        out_data_1 = self.sg(out_data_1_1) * out_data_11 + (1 - self.sg(out_data_1_1)) * out_data_11
        out_data_8 = self.basicupsample8(out_data_8)
        out_data_4 = self.basicupsample4(out_data_4)
        out_data_2 = self.basicupsample2(out_data_2)
        out_data_1 = self.basicupsample1(out_data_1)

        out_data = torch.cat([out_data_8,out_data_4,out_data_2,out_data_1],dim=1)

        out_data = self.reg_layer(out_data)


        return torch.abs(out_data)

class Graph_Attention_Union(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Graph_Attention_Union, self).__init__()

        # search region nodes linear transformation
        self.support = nn.Conv2d(in_channel, in_channel, 1, 1, bias=False)

        # target template nodes linear transformation
        self.query = nn.Conv2d(in_channel, in_channel, 1, 1, bias=False)

        # linear transformation for message passing
        self.g = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1, 1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
        )

        # aggregated feature
        self.fi = nn.Sequential(
            nn.Conv2d(in_channel*2, out_channel, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, zf, xf):
        # linear transformation
        xf_trans = self.query(xf)
        zf_trans = self.support(zf)

        # linear transformation for message passing
        xf_g = self.g(xf)
        zf_g = self.g(zf)

        # calculate similarity
        shape_x = xf_trans.shape
        shape_z = zf_trans.shape

        zf_trans_plain = zf_trans.view(-1, shape_z[1], shape_z[2] * shape_z[3])
        zf_g_plain = zf_g.view(-1, shape_z[1], shape_z[2] * shape_z[3]).permute(0, 2, 1)
        xf_trans_plain = xf_trans.view(-1, shape_x[1], shape_x[2] * shape_x[3]).permute(0, 2, 1)

        similar = torch.matmul(xf_trans_plain, zf_trans_plain)
        similar = F.softmax(similar, dim=2)

        embedding = torch.matmul(similar, zf_g_plain).permute(0, 2, 1)
        embedding = embedding.view(-1, shape_x[1], shape_x[2], shape_x[3])

        # aggregated feature
        output = torch.cat([embedding, xf_g], 1)
        output = self.fi(output)
        return output



import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax

##################################################################################
class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x


class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class AFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2, x4):
        x = torch.cat([x1, x2, x4], dim=1)
        return self.conv(x)


class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane-3, kernel_size=1, stride=1, relu=True)
        )

        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out

###################################################################################################
######################################################################################################
class Att_1(nn.Module):
    def __init__(self, in_channel, out_channel,thita=1e-4):
        super(Att_1, self).__init__()
        self.thita = thita
        self.out_channel = out_channel
        self.value_conv = nn.Sequential(
            BasicConv(self.out_channel, self.out_channel, kernel_size=1, stride=1, relu=True),
        )
        
        self.key_conv = nn.Sequential(
            BasicConv(self.out_channel, self.out_channel, kernel_size=1, stride=1, relu=True),
        )
        
        self.query_conv = nn.Sequential(  
            BasicConv(self.out_channel,self.out_channel,kernel_size=1,stride=1,relu=True),
        )

        #########################################################
        #########################################################
        self.event_feature_extract = nn.Sequential(    
            BasicConv(out_channel*1,self.out_channel,kernel_size=1,stride=1,relu=True),
            BasicConv(out_channel*1,self.out_channel,kernel_size=1,stride=1,relu=True),
        )
        self.x_feature_extract = nn.Sequential(
            BasicConv(out_channel*3,self.out_channel,kernel_size=1,stride=1,relu=True),
            BasicConv(out_channel*1,self.out_channel,kernel_size=1,stride=1,relu=True),
        )

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
        self.INF = INF
        
    def forward(self, x1, x2,event):

        x = torch.cat([x1, x2], dim=1)
        x = self.x_feature_extract[0](x)
        event_feature = self.event_feature_extract[0](event)

        x_2 = F.interpolate(x,scale_factor=0.25)
        event_feature = F.interpolate(event_feature,scale_factor=0.25)

        x_2 = self.x_feature_extract[1](x_2)
        event_feature = self.event_feature_extract[1](event_feature)

        m_batchsize,C,width ,height = x_2.size()
        proj_query = self.query_conv(x_2).view(m_batchsize, -1, width*height).permute(0,2,1) # B X CX(N)
        proj_key = self.key_conv(event_feature).view(m_batchsize, -1, width*height) # B X C x (*W*H)
        
        energy = torch.bmm(proj_query, proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N)
        proj_value = self.value_conv(x_2).view(m_batchsize, -1, width*height) # B X C X N
 
        out = torch.bmm(proj_value,attention.permute(0, 2, 1) )
        out = out.view(m_batchsize, C, width, height)
 
        
        out = F.interpolate(out,scale_factor=4) 
        out = self.gamma * out
        
        out = self.thita * out + x
        event_feature = F.interpolate(event_feature,scale_factor=4)
        return out,event_feature
    

class Att_2(nn.Module):
    def __init__(self, in_channel, out_channel,thita=1e-4):
        super(Att_2, self).__init__()
        self.thita = thita
        self.out_channel = out_channel
        self.value_conv = nn.Sequential(
            BasicConv(self.out_channel, self.out_channel, kernel_size=1, stride=1, relu=True),
        )
        
        self.key_conv = nn.Sequential(
            BasicConv(self.out_channel, self.out_channel, kernel_size=1, stride=1, relu=True),
        )
        
        self.query_conv = nn.Sequential(   
            BasicConv(self.out_channel,self.out_channel,kernel_size=1,stride=1,relu=True),
        )

        self.event_feature_extract = nn.Sequential(    
            BasicConv(out_channel*2,self.out_channel,kernel_size=1,stride=1,relu=True),
            BasicConv(out_channel*1,self.out_channel,kernel_size=1,stride=1,relu=True),
        )
        self.x_feature_extract = nn.Sequential(
            BasicConv(out_channel*3,self.out_channel,kernel_size=1,stride=1,relu=True),
            BasicConv(out_channel*1,self.out_channel,kernel_size=1,stride=1,relu=True),
        )

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
        self.INF = INF
 
    def forward(self, x1, x2,event,last_event):
        x = torch.cat([x1, x2], dim=1)
        x = self.x_feature_extract[0](x)
        event = torch.cat([event,last_event],dim=1)
        event_feature = self.event_feature_extract[0](event)

        x_2 = F.interpolate(x,scale_factor=0.25)
        event_feature = F.interpolate(event_feature,scale_factor=0.25)

        x_2 = self.x_feature_extract[1](x_2)
        event_feature = self.event_feature_extract[1](event_feature)

        m_batchsize,C,width ,height = x_2.size()
        proj_query = self.query_conv(x_2).view(m_batchsize, -1, width*height).permute(0,2,1) # B X CX(N)
        proj_key = self.key_conv(event_feature).view(m_batchsize, -1, width*height) # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N)
        proj_value = self.value_conv(x_2).view(m_batchsize, -1, width*height) # B X C X N
 
        out = torch.bmm(proj_value,attention.permute(0, 2, 1) )
        out = out.view(m_batchsize, C, width, height)
 
        out = F.interpolate(out,scale_factor=4) 
        out = self.gamma * out
        out = self.thita * out + x
        
        event_feature = F.interpolate(event_feature,scale_factor=4)
        return out,event_feature
    


def INF(B,H,W):
     return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1)


##########################################################################################################
class STRA(nn.Module):
    def __init__(self, num_res,base_channel=32):
        super(STRA, self).__init__()
        self.thita = 1e-3
        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            EBlock(base_channel*2, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )


        self.event_c1=nn.Conv2d(in_channels=2, out_channels=10, kernel_size=1, stride=1, padding=0, bias=False)
        self.event_c2=nn.Conv2d(in_channels=10, out_channels=base_channel, kernel_size=1, stride=1, padding=0, bias=False)

        self.Event_Fusion_3 = nn.ModuleList([
            Att_1(base_channel*4,base_channel*1,self.thita)
        ])
        self.Event_Fusion_4 = nn.ModuleList([
            Att_2(base_channel*3,base_channel*1,self.thita)
        ])



    def forward(self, x,output_last_feature=None):

        event = x[:,range(3,5),:,:]
        x = x[:,range(0,3),:,:]

        event = self.event_c1(event)
        event = self.event_c2(event)

        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.Encoder[2](z)


        z21 = F.interpolate(res2, scale_factor=2)

        if output_last_feature is not None:
            res1,event_feature = self.Event_Fusion_4[0](res1,z21,event,output_last_feature)
        else:
            res1,event_feature = self.Event_Fusion_3[0](res1,z21,event)

        z = self.Decoder[0](z)
        z = self.feat_extract[3](z)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z = self.feat_extract[4](z)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)

        return z+x,event_feature

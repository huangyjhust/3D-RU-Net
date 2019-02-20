#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 20:58:07 2019

@author: customer
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from skimage.measure import label,regionprops

class ResBlock(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch, kernel, Inplace=True,Dilation=1):
        super(ResBlock, self).__init__()
        padding=((kernel[0]-1)//2,Dilation,Dilation)
        dilation=(1,Dilation,Dilation)
        self.Conv1=nn.Conv3d(in_ch, out_ch, kernel, padding=padding,dilation=dilation)
        self.BN1=torch.nn.InstanceNorm3d(out_ch)
        self.Relu=nn.ReLU(inplace=Inplace)
        self.Conv2=nn.Conv3d(out_ch, out_ch, kernel, padding=padding,dilation=dilation)
        self.BN2=torch.nn.InstanceNorm3d(out_ch)
        self.Conv3=nn.Conv3d(out_ch, out_ch, kernel, padding=padding,dilation=dilation)
        self.BN3=torch.nn.InstanceNorm3d(out_ch)
    def forward(self, x):
        x1 = self.Conv1(x)
        x2 = self.BN1(x1)
        x3 = self.Relu(x2)
        x4 = self.Conv2(x3)
        x5 = self.BN2(x4)
        x6 = self.Relu(x5)
        x7 = self.Conv3(x6)
        x8 = self.BN3(x7)
        x9 = torch.add(x8,x1)
        x10 = self.Relu(x9)
        return x10


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, Inplace, Dilation=1):
        super(inconv, self).__init__()
        self.conv = ResBlock(in_ch, out_ch, (1,3,3), Inplace,Dilation)
    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, p_kernel, Inplace,Dilation=1):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool3d(p_kernel),
            ResBlock(in_ch, out_ch, (3,3,3), Inplace,Dilation)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, p_kernel, c_kernel, Inplace=True,learn=False,Dilation=1):
        super(up, self).__init__()
        self.p_kernel=p_kernel
        self.learn=learn
        if self.learn:
            self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)#torch.upsample(in_ch, out_ch,)#nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)
        self.fuse = ResBlock(in_ch, out_ch, c_kernel, Inplace,Dilation)
        self.conv = nn.Conv3d(in_ch, out_ch, (1,1,1))
        self.Relu=nn.ReLU(inplace=Inplace)
    def forward(self, x1, x2):
        if not self.learn:
            x1 = F.upsample(x1, size=(x1.size()[2]*self.p_kernel[0],x1.size()[3]*self.p_kernel[1],x1.size()[4]*self.p_kernel[2]),mode='trilinear')
        x1 = self.conv(x1)
        x1 = self.Relu(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.fuse(x)
        return x

class OutconvG(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutconvG, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
class OutconvR(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutconvR, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class OutconvC(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutconvC, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class GlobalImageEncoder(nn.Module):
    def __init__(self, opt):
        super(GlobalImageEncoder, self).__init__()
        self.opt=opt
        self.n_classes=len(opt.DICT_CLASS.keys())
        self.Inplace=True
        self.Base=opt.BASE_CHANNELS
        self.inc = inconv(1, self.Base,self.Inplace,Dilation=opt.STAGE_DILATION[0])
        self.down1 = down(self.Base, self.Base*2,(1,2,2),self.Inplace,Dilation=opt.STAGE_DILATION[1])
        self.down2 = down(self.Base*2, self.Base*4, (2,2,2),self.Inplace,Dilation=opt.STAGE_DILATION[2])
        self.LocTop = OutconvG(self.Base*4, self.n_classes)
    def forward(self,x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        LocOut=self.LocTop(x3)
        LocOut=F.softmax(LocOut)
        return LocOut,[x1,x2,x3]    
    def TrainForward(self,x,y,GetGlobalFeat=False):
        y= F.max_pool3d(y,kernel_size=(2,4,4),stride=(2,4,4))
        LocOut,GlobalFeatPyramid=self.forward(x)
        if GetGlobalFeat:
            return LocOut,y,GlobalFeatPyramid
        else:
            return LocOut,y
class LocalRegionDecoder(nn.Module):
    def __init__(self, opt):
        super(LocalRegionDecoder, self).__init__()
        self.opt=opt
        self.n_classes=len(opt.DICT_CLASS.keys())
        self.Inplace=True
        self.Base=opt.BASE_CHANNELS
        self.up1 = up(self.Base*4, self.Base*2,(2,2,2),(3,3,3),self.Inplace,False,Dilation=opt.STAGE_DILATION[1])
        self.up2 = up(self.Base*2, self.Base,(1,2,2),(1,3,3),self.Inplace,False,Dilation=opt.STAGE_DILATION[0])
        self.SegTop1 = OutconvR(self.Base, self.n_classes)
        self.SegTop2 = OutconvC(self.Base, self.n_classes)
    def forward(self,GlobalFeatPyramid,RoIs):
        x1=GlobalFeatPyramid[0]
        x2=GlobalFeatPyramid[1]
        x3=GlobalFeatPyramid[2]
        P_Region=[]
        P_Contour=[]

        for i in range(len(RoIs)):
            Zstart=RoIs[i][0]
            Ystart=RoIs[i][1]
            Xstart=RoIs[i][2]
            Zend=RoIs[i][3]
            Yend=RoIs[i][4]
            Xend=RoIs[i][5]
            #RoI TensorPyramid
            RoiTensorPyramid=[x3[:,:,Zstart:Zend,Ystart:Yend,Xstart:Xend],\
                              x2[:,:,Zstart*2:Zend*2,Ystart*2:Yend*2,Xstart*2:Xend*2],\
                              x1[:,:,Zstart*2:Zend*2,Ystart*4:Yend*4,Xstart*4:Xend*4]]
            
            p = self.up1(RoiTensorPyramid[0], RoiTensorPyramid[1])
            p = self.up2(p, RoiTensorPyramid[2])
            p_r = self.SegTop1(p)
            p_r = F.softmax(p_r)

            p_c = self.SegTop2(p) 
            p_c = F.softmax(p_c)

            P_Region.append(p_r)
            P_Contour.append(p_c) 
        return P_Region,P_Contour
    def TrainForward(self,GlobalFeatPyramid,RoIs,y_region,y_contour):
        Y_Region=[]
        Y_Contour=[]
        #Extract in-region labels
        for i in range(len(RoIs)):
            Zstart=RoIs[i][0]
            Ystart=RoIs[i][1]
            Xstart=RoIs[i][2]
            Zend=RoIs[i][3]
            Yend=RoIs[i][4]
            Xend=RoIs[i][5]
            y_region_RoI=y_region[:,:,Zstart*2:Zend*2,Ystart*4:Yend*4,Xstart*4:Xend*4]
            y_contour_RoI=y_contour[:,:,Zstart*2:Zend*2,Ystart*4:Yend*4,Xstart*4:Xend*4]
            Y_Region.append(y_region_RoI)
            Y_Contour.append(y_contour_RoI)
        P_Region,P_Contour=self.forward(GlobalFeatPyramid,RoIs)
        return P_Region,P_Contour,Y_Region,Y_Contour
    
class RU_Net(nn.Module):
    def __init__(self, opt):
        super(RU_Net, self).__init__()
        self.opt=opt
        self.n_classes=len(opt.DICT_CLASS.keys())
        self.Inplace=True
        self.Base=48
        self.GlobalImageEncoder=GlobalImageEncoder(opt)
        self.LocalRegionDecoder=LocalRegionDecoder(opt)
    def forward_RoI_Loc(self, x,y):
        LocOut,Y=self.GlobalImageEncoder.TrainForward(x,y,False)
        return [LocOut,Y]
    def Localization(self,LocOut,Train=True):
        if Train:
            MAX_ROIS=self.opt.MAX_ROIS_TRAIN
        else:
            MAX_ROIS=self.opt.MAX_ROIS_TEST
        LocOut = LocOut.to(device='cpu')
        LocOut = LocOut.detach().numpy()
        RoIs=[]
        #num=0
        for i in range(1,self.n_classes):
            Heatmap = LocOut[0,i]
            Heatmap = (Heatmap-np.min(Heatmap))/(np.max(Heatmap)-np.min(Heatmap))
            Heatmap[Heatmap<0.5]=0
            Heatmap[Heatmap>=0.5]=1
            Heatmap*=255
            ConnectMap=label(Heatmap, connectivity= 2)
            Props = regionprops(ConnectMap)
            Area=np.zeros([len(Props)])
            Area=[]
            Bbox=[]
            for j in range(len(Props)):
                Area.append(Props[j]['area'])
                Bbox.append(list(Props[j]['bbox'])) 
                OverDesignRange=[1,2,2]
                for k in range(3):
                    if Bbox[j][k]-OverDesignRange[k]<0:
                        Bbox[j][k]=0
                    else:
                        Bbox[j][k]-=OverDesignRange[k]
                for k in range(3,6):
                    if Bbox[j][k]+OverDesignRange[k-3]>=Heatmap.shape[k-3]-1:
                        Bbox[j][k]=Heatmap.shape[k-3]-1
                    else:
                        Bbox[j][k]+=OverDesignRange[k-3]           
            Area=np.array(Area)
            Bbox=np.array(Bbox)
            argsort=np.argsort(Area)
            Area=Area[argsort]
            Bbox=Bbox[argsort]
            Area=Area[::-1]
            Bbox=Bbox[::-1,:]
            
            max_boxes=MAX_ROIS[self.opt.DICT_CLASS[i]]
            if Area.shape[0]>=max_boxes:
                OutBbox=Bbox[:max_boxes,:]
            elif Area.shape[0]==0:
                OutBbox=np.zeros([1,6],dtype=np.int)
                OutBbox[0]=[0,0,0,1,1,1]
            else:
                OutBbox=Bbox
            for j in range(OutBbox.shape[0]):
                RoIs.append(OutBbox[j,:])
            
        return RoIs
            
        
    def TrainForward(self, x, y_region, y_contour):
        LocOut,y_region_down,GlobalFeatPyramid=self.GlobalImageEncoder.TrainForward(x,y_region,True)
        RoIs=self.Localization(LocOut,Train=True)
        P_Region,P_Contour,Y_Region,Y_Contour=self.LocalRegionDecoder.TrainForward(GlobalFeatPyramid,RoIs,y_region,y_contour)

        
        return P_Region,P_Contour,Y_Region,Y_Contour,RoIs,[LocOut,y_region_down]
    def forward(self, x):
        LocOut,GlobalFeatPyramid=self.GlobalImageEncoder.forward(x)
        RoIs=self.Localization(LocOut,Train=False) 
        P_Region,P_Contour=self.LocalRegionDecoder(GlobalFeatPyramid,RoIs)
        return P_Region,P_Contour,RoIs
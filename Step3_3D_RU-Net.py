#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 15:10:18 2018

@author: customer
"""
import os
import SimpleITK as sitk
import numpy as np
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import time
from graphviz import Digraph
from skimage.measure import label,regionprops
from matplotlib import pyplot as pl
from skimage import filters
import cv2
inplace=True
ClassIndex={0:'Background', 1:'Cancer'}
NumRoIsTrain={'Background':0,'Cancer':1}
NumRoIsTest={'Background':0,'Cancer':1}
ResRates={0:'HighRes',1:'MidRes',2:'LowRes'}
ToSpacing={'HighRes':[1,1,4],'MidRes':[1.5,1.5,4],'LowRes':[2,2,4]}
Project='Colon'
Root='../Data/Normalized/'
ResRate=0#0,1,2
GPU='cuda:'+str(ValQuarter%2)
WeightPath='./'+Project+'Weights/'+Project+ResRates[ResRate]+'Params_'+str(ValQuarter)+'.pkl'
print 'Using GPU',GPU
class ResBlock(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch, kernel, inplace=True):
        super(ResBlock, self).__init__()
        #self.conv = nn.Sequential(
        self.Conv1=nn.Conv3d(in_ch, out_ch, kernel, padding=((kernel[0]-1)/2,(kernel[1]-1)/2,(kernel[2]-1)/2))
        self.BN1=nn.BatchNorm3d(out_ch)
        self.Relu=nn.ReLU(inplace=inplace)
        self.Conv2=nn.Conv3d(out_ch, out_ch, kernel, padding=((kernel[0]-1)/2,(kernel[1]-1)/2,(kernel[2]-1)/2))
        self.BN2=nn.BatchNorm3d(out_ch)
        self.Conv3=nn.Conv3d(out_ch, out_ch, kernel, padding=((kernel[0]-1)/2,(kernel[1]-1)/2,(kernel[2]-1)/2))
        self.BN3=nn.BatchNorm3d(out_ch)
    
        #nn.ReLU(inplace=inplace)
        #)

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
    def __init__(self, in_ch, out_ch, inplace):
        super(inconv, self).__init__()
        self.conv = ResBlock(in_ch, out_ch, (1,3,3), inplace)
        self.BN=nn.BatchNorm3d(out_ch)
        self.Relu=nn.ReLU(inplace=inplace)
    def forward(self, x):
        x = self.conv(x)
        x = self.BN(x)
        x = self.Relu(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, p_kernel, inplace):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool3d(p_kernel),
            ResBlock(in_ch, out_ch, (3,3,3), inplace)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, p_kernel, c_kernel, inplace=inplace,learn=False):
        super(up, self).__init__()
        self.p_kernel=p_kernel
        self.learn=learn
        if self.learn:
            self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)#torch.upsample(in_ch, out_ch,)#nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)
        self.fuse = ResBlock(in_ch, out_ch, c_kernel, inplace)
        self.conv = nn.Conv3d(in_ch, out_ch, (1,1,1))
        self.Relu=nn.ReLU(inplace=inplace)
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
#class OutconvC(nn.Module):
#    def __init__(self, in_ch, out_ch):
#        super(OutconvC, self).__init__()
#        self.conv1 = nn.Conv3d(in_ch, in_ch, (1,3,3))
#        self.conv2 = nn.Conv3d(in_ch, out_ch, 1)
#
#    def forward(self, x):
#        x = self.conv1(x)
#        x = self.conv2(x)
#        return x
class OutconvC(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutconvC, self).__init__()
        self.conv = nn.Conv3d(in_ch, in_ch, (1,3,3))

    def forward(self, x):
        x = self.conv(x)

        return x
class RU_Net(nn.Module):
    def __init__(self, n_channels, n_classes,inplace):
        super(RU_Net, self).__init__()
        self.n_classes=n_classes
        self.inplace=inplace
        self.Base=48
        self.inc = inconv(n_channels, self.Base,inplace)
        self.down1 = down(self.Base, self.Base*2,(1,2,2),inplace)
        self.down2 = down(self.Base*2, self.Base*4, (2,2,2),inplace)
        self.LocTop = OutconvG(self.Base*4, n_classes)
        self.up1 = up(self.Base*4, self.Base*2,(2,2,2),(3,3,3),inplace,False)
        self.up2 = up(self.Base*2, self.Base,(1,2,2),(1,3,3),inplace,False)
        self.SegTop1 = OutconvR(self.Base, n_classes)
        self.SegTop2 = OutconvG(self.Base, n_classes)
    def forward_RoI_Loc(self, x,y):
        y= F.max_pool3d(y,kernel_size=(2,4,4),stride=(2,4,4))
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        #x4 = self.down3(x3)
        LocOut=self.LocTop(x3)
        LocOut=F.softmax(LocOut)
        return [LocOut,y]

    def Localization(self,LocOut,Train=True):
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
                for k in range(3):
                    if Bbox[j][k]-2<0:
                        Bbox[j][k]=0
                    else:
                        Bbox[j][k]-=2
                for k in range(3,6):
                    if Bbox[j][k]+2>=Heatmap.shape[k-3]-1:
                        Bbox[j][k]=Heatmap.shape[k-3]-1
                    else:
                        Bbox[j][k]+=2                
            Area=np.array(Area)
            Bbox=np.array(Bbox)
            #print Area
            #print Bbox
            argsort=np.argsort(Area)
            Area=Area[argsort]
            Bbox=Bbox[argsort]
            Area=Area[::-1]
            Bbox=Bbox[::-1,:]
            #print Area
            #print Bbox
            if Train:
                max_boxes=NumRoIsTrain[ClassIndex[i]]
                #print max_boxes
                if Area.shape[0]>=max_boxes:
                    OutBbox=Bbox[:max_boxes,:]
                else:
                    OutBbox=np.zeros([1,6],dtype=np.int)
                    OutBbox[0]=[0,0,0,1,1,1]
                for j in range(OutBbox.shape[0]):
                    RoIs.append(OutBbox[j,:])
                    #num+=1
                    #print 'num=',num
            else:
                max_boxes=NumRoIsTest[ClassIndex[i]]
                #print max_boxes
                if Area.shape[0]>=max_boxes:
                    OutBbox=Bbox[:max_boxes,:]
                elif Area.shape[0]>0:
                    OutBbox=Bbox
                else:
                    OutBbox=np.zeros([1,6],dtype=np.int)
                    OutBbox[0]=[0,0,0,1,1,1]
                for j in range(OutBbox.shape[0]):
                    RoIs.append(OutBbox[j,:])
                    #num+=1
                    #print 'num=',num

        
            #print Area.shape[0],max_boxes
        return RoIs
            
        
    def train_forward(self, x, y_region, y_contour):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        LocOut=self.LocTop(x3)
        LocOut=F.softmax(LocOut)
        RoIs=self.Localization(LocOut,Train=True)        
        #print len(RoIs)
        P_Region=[]
        P_Contour=[]
        Y_Region=[]
        Y_Contour=[]
        for i in range(len(RoIs)):
            Zstart=RoIs[i][0]
            Ystart=RoIs[i][1]
            Xstart=RoIs[i][2]
            Zend=RoIs[i][3]
            Yend=RoIs[i][4]
            Xend=RoIs[i][5]
            #RoI Cropping Layer
            x3_RoI=x3[:,:,Zstart:Zend,Ystart:Yend,Xstart:Xend]
            x2_RoI=x2[:,:,Zstart*2:Zend*2,Ystart*2:Yend*2,Xstart*2:Xend*2]
            x1_RoI=x1[:,:,Zstart*2:Zend*2,Ystart*4:Yend*4,Xstart*4:Xend*4]
            
            y_region_RoI=y_region[:,:,Zstart*2:Zend*2,Ystart*4:Yend*4,Xstart*4:Xend*4]
            y_contour_RoI=y_contour[:,:,Zstart*2:Zend*2,Ystart*4:Yend*4,Xstart*4:Xend*4]
            p = self.up1(x3_RoI, x2_RoI)
            p = self.up2(p, x1_RoI)
            p_r = self.SegTop1(p)
            p_r = F.sigmoid(p_r)

            p_c = self.SegTop2(p)
            p_c = F.sigmoid(p_c)            
            #p = p.to('cpu').detach()
            #y_RoI = y_RoI.to('cpu').detach()
            P_Region.append(p_r)
            P_Contour.append(p_c)
            Y_Region.append(y_region_RoI)
            Y_Contour.append(y_contour_RoI)
        
        return P_Region,P_Contour,Y_Region,Y_Contour,RoIs
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        LocOut=self.LocTop(x3)
        LocOut=F.softmax(LocOut)
        RoIs=self.Localization(LocOut,Train=False)        
        
        P_Region=[]
        P_Contour=[]
        for i in range(len(RoIs)):
            Zstart=RoIs[i][0]
            Ystart=RoIs[i][1]
            Xstart=RoIs[i][2]
            Zend=RoIs[i][3]
            Yend=RoIs[i][4]
            Xend=RoIs[i][5]
            #RoI Cropping Layer
            x3_RoI=x3[:,:,Zstart:Zend,Ystart:Yend,Xstart:Xend]
            x2_RoI=x2[:,:,Zstart*2:Zend*2,Ystart*2:Yend*2,Xstart*2:Xend*2]
            x1_RoI=x1[:,:,Zstart*2:Zend*2,Ystart*4:Yend*4,Xstart*4:Xend*4]
            
            p = self.up1(x3_RoI,x2_RoI)
            p = self.up2(p, x1_RoI)
            p_r = self.SegTop1(p)
            p_r = F.sigmoid(p_r)
            p_r = p_r.to('cpu').detach().numpy()
            P_Region.append(p_r)

            p_c = self.SegTop2(p)
            p_c = F.sigmoid(p_c) 
            p_c = p_c.to('cpu').detach().numpy()
            P_Contour.append(p_c)
        
        return P_Region,P_Contour,RoIs

def MultiClassDiceLossFunc(y_pred,y_true):
    overlap=torch.zeros([1]).cuda(GPU)
    bottom=torch.zeros([1]).cuda(GPU)
    for i in range(1,len(ClassIndex)):
        overlap+=torch.sum(y_pred[0,i]*y_true[0,i])
        bottom+=torch.sum(y_pred[0,i])+torch.sum(y_true[0,i])
    return 1-2*overlap/bottom
#def GetMaximumBbox(Image):
    
def GetImage(Patient):

    ImageInput=sitk.ReadImage(Root+'/'+Patient+'/'+ResRates[ResRate]+'/'+'Image.mhd')
    ImageInput=sitk.GetArrayFromImage(ImageInput)
    Shape=ImageInput.shape
    #Maximum Bbox
    otsu=filters.threshold_otsu(ImageInput[ImageInput.shape[0]/2])
    Seg=np.zeros(ImageInput.shape)
    Seg[ImageInput>=otsu]=255
    Seg=Seg.astype(np.int)
    ConnectMap=label(Seg, connectivity= 2)
    Props = regionprops(ConnectMap)
    Area=np.zeros([len(Props)])
    Area=[]
    Bbox=[]
    for j in range(len(Props)):
        Area.append(Props[j]['area'])
        Bbox.append(Props[j]['bbox'])
    Area=np.array(Area)
    Bbox=np.array(Bbox)
    argsort=np.argsort(Area)
    Area=Area[argsort]
    Bbox=Bbox[argsort]
    Area=Area[::-1]
    Bbox=Bbox[::-1,:]
    MaximumBbox=Bbox[0]            
            
    Image=np.zeros([1,1,ImageInput.shape[0],ImageInput.shape[1],ImageInput.shape[2]])
    Image[0,0]=ImageInput
    Image=Image.astype(np.float)/255
    Image=Image[:,:,MaximumBbox[0]:MaximumBbox[3],MaximumBbox[1]:MaximumBbox[4],MaximumBbox[2]:MaximumBbox[5]]
    Image=torch.from_numpy(Image)
    Image=Image.float()
    Image = Image.to(device=GPU)

    Label1=sitk.ReadImage(Root+'/'+Patient+'/'+ResRates[ResRate]+'/'+'Label.mhd')
    Label1=sitk.GetArrayFromImage(Label1)
    Label2=sitk.ReadImage(Root+'/'+Patient+'/'+ResRates[ResRate]+'/'+'Contour.mhd')
    Label2=sitk.GetArrayFromImage(Label2)
    
    LabelRegion=np.zeros([1,2,ImageInput.shape[0],ImageInput.shape[1],ImageInput.shape[2]])
    LabelRegion[0,1]=Label1
    LabelRegion[0,0]=1-Label1

    LabelContour=np.zeros([1,2,ImageInput.shape[0],ImageInput.shape[1],ImageInput.shape[2]])
    LabelContour[0,1]=Label2
    LabelContour[0,0]=1-Label2
    
    LabelRegion=LabelRegion[:,:,MaximumBbox[0]:MaximumBbox[3],MaximumBbox[1]:MaximumBbox[4],MaximumBbox[2]:MaximumBbox[5]]
    LabelRegion=torch.from_numpy(LabelRegion)
    LabelRegion=LabelRegion.float()
    LabelRegion=LabelRegion.to(device=GPU)
    
    LabelContour=LabelContour[:,:,MaximumBbox[0]:MaximumBbox[3],MaximumBbox[1]:MaximumBbox[4],MaximumBbox[2]:MaximumBbox[5]]
    LabelContour=torch.from_numpy(LabelContour)
    LabelContour=LabelContour.float()
    LabelContour=LabelContour.to(device=GPU)
    return Image,LabelRegion,LabelContour,Shape,MaximumBbox

def Predict(Patient):
    Image,LabelRegion,LabelContour,Shape,MaximumBbox=GetImage('Val/'+Patient)
    Label=LabelRegion.to('cpu').detach().numpy()
    time1=time.time()
    PredSeg=Model.forward(Image)
    time2=time.time()
    print 'time used:',time2-time1
    
    RegionOutput=np.zeros(Label.shape)
    RegionWeight=np.zeros(Label.shape)+0.001
    RoIs=PredSeg[2]
    for i in range(len(PredSeg[0])):
        Coord=RoIs[i]*np.array([2,4,4,2,4,4])
        Weight=np.ones(np.asarray(PredSeg[0][i][0].shape))
        RegionOutput[0,:,Coord[0]:Coord[3],Coord[1]:Coord[4],Coord[2]:Coord[5]]+=PredSeg[0][i][0]#.to('cpu').detach().numpy()
        RegionWeight[0,:,Coord[0]:Coord[3],Coord[1]:Coord[4],Coord[2]:Coord[5]]+=Weight
    RegionOutput/=RegionWeight

    ContourOutput=np.zeros(Label.shape)
    ContourWeight=np.zeros(Label.shape)+0.001
    RoIs=PredSeg[2]
    for i in range(len(PredSeg[0])):
        Coord=RoIs[i]*np.array([2,4,4,2,4,4])
        Weight=np.ones(np.asarray(PredSeg[0][i][0].shape))
        ContourOutput[0,:,Coord[0]:Coord[3],Coord[1]:Coord[4],Coord[2]:Coord[5]]+=PredSeg[1][i][0]#.to('cpu').detach().numpy()
        ContourWeight[0,:,Coord[0]:Coord[3],Coord[1]:Coord[4],Coord[2]:Coord[5]]+=Weight
    ContourOutput/=ContourWeight
    
    OutputWhole1=np.zeros(Shape,dtype=np.uint8)
    OutputWhole2=np.zeros(Shape,dtype=np.uint8)
    OutputWhole=np.zeros(Shape,dtype=np.uint8)
    OutputWhole1[MaximumBbox[0]:MaximumBbox[3],MaximumBbox[1]:MaximumBbox[4],MaximumBbox[2]:MaximumBbox[5]]=(RegionOutput[0,1]*255).astype(np.uint8)
    OutputWhole2[MaximumBbox[0]:MaximumBbox[3],MaximumBbox[1]:MaximumBbox[4],MaximumBbox[2]:MaximumBbox[5]]=(ContourOutput[0,1]*255).astype(np.uint8)

    OutputWhole[OutputWhole1>=128]=1
    OutputWhole[OutputWhole1<128]=0
    
    Loss=1-2*np.sum(RegionOutput[0,1]*Label[0,1])/(np.sum(RegionOutput[0,1])+np.sum(Label[0,1]))
    OutputWhole1=sitk.GetImageFromArray(OutputWhole1)
    OutputWhole1.SetSpacing(ToSpacing[ResRates[ResRate]])
    
    OutputWhole2=sitk.GetImageFromArray(OutputWhole2)
    OutputWhole2.SetSpacing(ToSpacing[ResRates[ResRate]])
    
    for Rid in range(len(RoIs)):
        color=(Rid+1,Rid+1,Rid+1)
        
        Coord=RoIs[Rid]*np.array([2,4,4,2,4,4])+np.array([MaximumBbox[0],MaximumBbox[1],MaximumBbox[2],MaximumBbox[0],MaximumBbox[1],MaximumBbox[2]])
        for protect in range(3):
            if Coord[protect+3]>=OutputWhole.shape[protect+0]:
                Coord[protect+3]=OutputWhole.shape[protect+0]-1

        
        Rgb=np.zeros([OutputWhole.shape[1],OutputWhole.shape[2],3],dtype=np.uint8)
        Rgb[:,:,0]=OutputWhole[Coord[0]]
        OutputWhole[Coord[0]]=cv2.rectangle(Rgb,(Coord[2],Coord[1]),(Coord[5],Coord[4]),color=color,thickness=2)[:,:,0]
        Rgb[:,:,0]=OutputWhole[Coord[3]]
        OutputWhole[Coord[3]]=cv2.rectangle(Rgb,(Coord[2],Coord[1]),(Coord[5],Coord[4]),color=color,thickness=2)[:,:,0]
        
        Rgb=np.zeros([OutputWhole.shape[0],OutputWhole.shape[1],3],dtype=np.uint8)
        Rgb[:,:,0]=OutputWhole[:,:,Coord[2]]
        OutputWhole[:,:,Coord[2]]=cv2.rectangle(Rgb,(Coord[1],Coord[0]),(Coord[4],Coord[3]),color=color,thickness=2)[:,:,0]
        Rgb[:,:,0]=OutputWhole[:,:,Coord[5]]
        OutputWhole[:,:,Coord[5]]=cv2.rectangle(Rgb,(Coord[1],Coord[0]),(Coord[4],Coord[3]),color=color,thickness=2)[:,:,0]
        
        Rgb=np.zeros([OutputWhole.shape[0],OutputWhole.shape[2],3],dtype=np.uint8)
        Rgb[:,:,0]=OutputWhole[:,Coord[1],:]
        OutputWhole[:,Coord[1],:]=cv2.rectangle(Rgb,(Coord[2],Coord[0]),(Coord[5],Coord[3]),color=color,thickness=2)[:,:,0]
        Rgb[:,:,0]=OutputWhole[:,Coord[4],:]
        OutputWhole[:,Coord[4],:]=cv2.rectangle(Rgb,(Coord[2],Coord[0]),(Coord[5],Coord[3]),color=color,thickness=2)[:,:,0]
        #for z in range(Coord[0],Coord[3]):
        #    Rgb[:,:,0]=OutputWhole[z]
        #    OutputWhole[z]=cv2.rectangle(Rgb,(Coord[2],Coord[1]),(Coord[5],Coord[4]),color=(3,3,3))[:,:,0]
            
    OutputWhole=sitk.GetImageFromArray(OutputWhole)
    OutputWhole.SetSpacing(ToSpacing[ResRates[ResRate]])
    if os.path.exists('./Output'+Project+'/'+Patient)==False:
        os.mkdir('./Output'+Project+'/'+Patient)
    sitk.WriteImage(OutputWhole,'./Output'+Project+'/'+Patient+'/'+ResRates[ResRate]+'/Pred.mhd')
    sitk.WriteImage(OutputWhole1,'./Output'+Project+'/'+Patient+'/'+ResRates[ResRate]+'/PredRegion.mhd')
    sitk.WriteImage(OutputWhole2,'./Output'+Project+'/'+Patient+'/'+ResRates[ResRate]+'/PredContour.mhd')
    return Loss

lr=0.0001
Model=RU_Net(n_channels=1,n_classes=len(ClassIndex),inplace=inplace)
Model=Model.to(GPU)
optimizer = optim.Adam(Model.parameters(),lr=lr)#, momentum=0.9, weight_decay=0.0005)

TrainPatient=os.listdir(Root+'Train/')
TrainPatient=sorted(TrainPatient)
ValPatient=os.listdir(Root+'Val/')
ValPatient=sorted(ValPatient)
print PatientNames
NumVal = len(ValPatient)
NumTrain=len(TrainPatient)
#print TrainPatient
print ValPatient
Load=True
try:
    Model.load_state_dict(torch.load(WeightPath))
except:
#Initialization
    for epoch in range(20):
        for iteration in range(NumTrain):
            Patient=TrainPatient[random.randint(0,NumTrain-1)]
            Image,LabelRegion,LabelContour,Shape,MaximumBbox=GetImage('Train/'+Patient)
            PredRoI=Model.forward_RoI_Loc(Image,LabelRegion)
            optimizer.zero_grad()
            loss=MultiClassDiceLossFunc(PredRoI[0],PredRoI[1])
            loss.backward()
            optimizer.step()
            print loss
torch.save(Model.state_dict(), WeightPath)
#co-training
Lowest=1
for epoch in range(30):
    for iteration in range(NumTrain):
        Patient=TrainPatient[random.randint(0,NumTrain-1)]
        Image,LabelRegion,LabelContour,Shape,MaximumBbox=GetImage('Train/'+Patient)
        Label=LabelRegion
        PredRoI=Model.forward_RoI_Loc(Image,Label)
        optimizer.zero_grad()
        LossG=MultiClassDiceLossFunc(PredRoI[0],PredRoI[1])
        LossG.backward()
        optimizer.step()
        #print 'global loss=',loss
        PredSeg=Model.train_forward(Image,LabelRegion,LabelContour)
        LossR=torch.zeros([1]).cuda(GPU)

        for i in range(len(PredSeg[0])):
            LossR+=MultiClassDiceLossFunc(PredSeg[0][i],PredSeg[2][i])
        LossR/=len(PredSeg[0])
        
        LossC=torch.zeros([1]).cuda(GPU)
        #loss=torch
        for i in range(len(PredSeg[0])):
            LossC+=MultiClassDiceLossFunc(PredSeg[1][i],PredSeg[3][i])
        LossC/=len(PredSeg[0])
        CWeight=0.5
        LossAll=LossR+CWeight*LossC
        LossAll.backward()
        optimizer.step()
        
        LossG=LossG.to('cpu').detach().numpy()
        LossR=LossR.to('cpu').detach().numpy()
        LossC=LossC.to('cpu').detach().numpy()
        if LossG>0.5:
            print 'Hard Patient=',Patient 
        print 'loss={g=',LossG,',r=',LossR,',c=',LossC,'}'
    Loss=0
    for iteration in range(NumVal):
        PatientVal=ValPatient[iteration]
        Loss_temp=Predict('Val/'+PatientVal)
        Loss+=Loss_temp
        print PatientVal,' Loss=',Loss_temp
    Loss/=NumVal
    if Loss<Lowest:
        print 'Loss improved from ',Lowest,'to ',Loss
        torch.save(Model.state_dict(), WeightPath)
        print 'saved to ',WeightPath
        Lowest=Loss
    else:
        print 'not improved'
    print '\n\nValLoss=',Loss
    print 'Best Loss=',Lowest

Model.load_state_dict(torch.load(WeightPath))
Loss=0
for iteration in range(NumVal):
    PatientVal=ValPatient[iteration]
    Loss_temp=Predict(PatientVal)
    Loss+=Loss_temp
    print PatientVal,' Loss=',Loss_temp
Loss/=NumVal
if Loss<Lowest:
    print 'Loss improved from ',Lowest,'to ',Loss
    torch.save(Model.state_dict(), WeightPath)
    print 'saved to ',WeightPath
    Lowest=Loss
else:
    print 'not improved'
print '\n\nValLoss=',Loss




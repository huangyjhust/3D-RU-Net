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
ClassIndex={0:'Background', 1:'Mandible', 2:'Masserter'}
NumRoIsTrain={'Background':0,'Mandible':1,'Masserter':2}
NumRoIsTest={'Background':0,'Mandible':1,'Masserter':2}
Project='Head'
Root='/media/customer/Disk1/Head/Code/ProcessedHigh'
ValQuarter=3
GPU='cuda:'+str(ValQuarter%2)
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
        x6 = torch.add(x5,x1)
        x7 = self.Relu(x6)
        #x6 = self.Relu(x5)
        #x7 = self.Conv3(x6)
        #x8 = self.BN3(x7)
        #x9 = torch.add(x8,x1)
        #x10 = self.Relu(x9)
        
        return x7#x10


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, inplace):
        super(inconv, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, (1,3,3), padding=(0,1,1))#ResBlock(in_ch, out_ch, (1,3,3), inplace)
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


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
class RU_Net(nn.Module):
    def __init__(self, n_channels, n_classes,inplace):
        super(RU_Net, self).__init__()
        self.n_classes=n_classes
        self.inplace=inplace
        self.inc = inconv(n_channels, 32,inplace)
        self.down1 = down(32, 64,(1,2,2),inplace)
        self.down2 = down(64, 128,(2,2,2),inplace)
        self.down3 = down(128, 256,(2,2,2),inplace)
        #self.down4 = down(512, 512,inplace)
        self.LocTop = outconv(256, n_classes)
        self.up1 = up(256, 128,(2,2,2),(3,3,3),inplace,False)
        self.up2 = up(128, 64,(2,2,2),(3,3,3),inplace,False)
        self.up3 = up(64, 32,(1,2,2),(1,3,3),inplace,False)
        #self.up4 = up(128, 64,inplace)
        self.SegTop = outconv(32, n_classes)
    def forward_RoI_Loc(self, x,y):
        y= F.max_pool3d(y,kernel_size=(4,8,8),stride=(4,8,8))
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        LocOut=self.LocTop(x4)
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
                Bbox.append(Props[j]['bbox'])
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
            
        
    def train_forward(self, x, y):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        LocOut=self.LocTop(x4)
        LocOut=F.softmax(LocOut)
        RoIs=self.Localization(LocOut,Train=True)        
        #print len(RoIs)
        Pout=[]
        Yout=[]
        for i in range(len(RoIs)):
            Zstart=RoIs[i][0]
            Ystart=RoIs[i][1]
            Xstart=RoIs[i][2]
            Zend=RoIs[i][3]
            Yend=RoIs[i][4]
            Xend=RoIs[i][5]
            #RoI Cropping Layer
            x4_RoI=x4[:,:,Zstart:Zend,Ystart:Yend,Xstart:Xend]
            x3_RoI=x3[:,:,Zstart*2:Zend*2,Ystart*2:Yend*2,Xstart*2:Xend*2]
            x2_RoI=x2[:,:,Zstart*4:Zend*4,Ystart*4:Yend*4,Xstart*4:Xend*4]
            x1_RoI=x1[:,:,Zstart*4:Zend*4,Ystart*8:Yend*8,Xstart*8:Xend*8]
            
            y_RoI=y[:,:,Zstart*4:Zend*4,Ystart*8:Yend*8,Xstart*8:Xend*8]
            p = self.up1(x4_RoI, x3_RoI)
            p = self.up2(p, x2_RoI)
            p = self.up3(p, x1_RoI)
            p = self.SegTop(p)
            p = F.sigmoid(p)
            #p = p.to('cpu').detach()
            #y_RoI = y_RoI.to('cpu').detach()
            Pout.append(p)
            Yout.append(y_RoI)
        
        return Pout,Yout,RoIs
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        LocOut=self.LocTop(x4)
        LocOut=F.softmax(LocOut)
        RoIs=self.Localization(LocOut,Train=False)        
        
        Pout=[]
        for i in range(len(RoIs)):
            Zstart=RoIs[i][0]
            Ystart=RoIs[i][1]
            Xstart=RoIs[i][2]
            Zend=RoIs[i][3]
            Yend=RoIs[i][4]
            Xend=RoIs[i][5]
            #RoI Cropping Layer
            x4_RoI=x4[:,:,Zstart:Zend,Ystart:Yend,Xstart:Xend]
            x3_RoI=x3[:,:,Zstart*2:Zend*2,Ystart*2:Yend*2,Xstart*2:Xend*2]
            x2_RoI=x2[:,:,Zstart*4:Zend*4,Ystart*4:Yend*4,Xstart*4:Xend*4]
            x1_RoI=x1[:,:,Zstart*4:Zend*4,Ystart*8:Yend*8,Xstart*8:Xend*8]
            
            p = self.up1(x4_RoI, x3_RoI)
            p = self.up2(p, x2_RoI)
            p = self.up3(p, x1_RoI)
            p = self.SegTop(p)
            p = F.sigmoid(p)
            p = p.to('cpu').detach().numpy()
            Pout.append(p)

        
        return Pout,RoIs

def DiceLossFunc(y_pred,y_true):
    overlap=torch.sum(y_pred*y_true)
    bottom=torch.sum(y_pred)+torch.sum(y_true)
    return 1-2*overlap/bottom
def ClassSensitiveDiceLossFunc(y_pred,y_true):
    overlap=torch.sum(y_pred*y_true)
    bottom=torch.sum(y_pred)+torch.sum(y_true)
    return 1-2*overlap/bottom
def MultiClassDiceLossFunc(y_pred,y_true):
    overlap=torch.zeros([1]).cuda(GPU)
    bottom=torch.zeros([1]).cuda(GPU)
    for i in range(1,len(ClassIndex)):
        overlap+=torch.sum(y_pred[0,i]*y_true[0,i])
        bottom+=torch.sum(y_pred[0,i])+torch.sum(y_true[0,i])
    return 1-2*overlap/bottom

#def GetMaximumBbox(Image):
    
def GetImage(Patient):

    Image1=sitk.ReadImage(Root+'/'+Patient+'ImgManWin.mhd')
    Image1=sitk.GetArrayFromImage(Image1)
    Shape=Image1.shape
    Image2=sitk.ReadImage(Root+'/'+Patient+'ImgMasWin.mhd')
    Image2=sitk.GetArrayFromImage(Image2)

    #Maximum Bbox

    otsu=filters.threshold_otsu(Image2[Image2.shape[0]/2])
    Seg=np.zeros(Image2.shape)
    Seg[Image2>=otsu]=255
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
            
    Image=np.zeros([1,2,Image1.shape[0],Image1.shape[1],Image1.shape[2]])
    Image[0,0]=Image1
    Image[0,1]=Image2
    Image=Image.astype(np.float)/255
    Image=Image[:,:,MaximumBbox[0]:MaximumBbox[3],MaximumBbox[1]:MaximumBbox[4],MaximumBbox[2]:MaximumBbox[5]]
    Image=torch.from_numpy(Image)
    Image=Image.float()
    Image = Image.to(device=GPU)

    Label1=sitk.ReadImage(Root+'/'+Patient+'Mandible.mhd')
    Direction=Label1.GetDirection()
    Origin=Label1.GetOrigin()
    Spacing=Label1.GetSpacing()
    Size=Label1.GetSize()
    NewSpacing=np.array(Spacing)*np.array([8,8,4])
    NewSize=np.array(Size)/np.array([8,8,4])
    Resample = sitk.ResampleImageFilter()
    Resample.SetOutputDirection(Direction)
    Resample.SetOutputOrigin(Origin)
    Resample.SetSize(NewSize)
    Resample.SetInterpolator(sitk.sitkNearestNeighbor)
    Resample.SetOutputSpacing(NewSpacing)
    Label1D = Resample.Execute(Label1)
    Label1D = sitk.GetArrayFromImage(Label1D)

    Label1=sitk.GetArrayFromImage(Label1)
    Label2=sitk.ReadImage(Root+'/'+Patient+'Masseter.mhd')
    Label2=sitk.GetArrayFromImage(Label2)
    
    Label=np.zeros([1,3,Image1.shape[0],Image1.shape[1],Image1.shape[2]])
    Label[0,1]=Label1
    Label[0,2]=Label2
    Label[0,0]=1-np.maximum(Label1,Label2)

    Label=Label[:,:,MaximumBbox[0]:MaximumBbox[3],MaximumBbox[1]:MaximumBbox[4],MaximumBbox[2]:MaximumBbox[5]]
    Label=torch.from_numpy(Label)
    Label=Label.float()
    Label=Label.to(device=GPU)
    return Image,Label,Shape,MaximumBbox

def Predict(Patient):
    Image,Label,Shape,MaximumBbox=GetImage(Patient)
    Label=Label.to('cpu').detach().numpy()
    time1=time.time()
    PredSeg=Model.forward(Image)
    time2=time.time()
    print 'time used:',time2-time1
    
    Output=np.zeros(Label.shape)
    W_Output=np.zeros(Label.shape)+0.001
    RoIs=PredSeg[1]
    for i in range(len(PredSeg[0])):
        Coord=RoIs[i]*np.array([4,8,8,4,8,8])
        Weight=np.ones(np.asarray(PredSeg[0][i][0].shape))
        Output[0,:,Coord[0]:Coord[3],Coord[1]:Coord[4],Coord[2]:Coord[5]]+=PredSeg[0][i][0]#.to('cpu').detach().numpy()
        W_Output[0,:,Coord[0]:Coord[3],Coord[1]:Coord[4],Coord[2]:Coord[5]]+=Weight
    Output/=W_Output
    Output1=(Output[0,1]*255).astype(np.uint8)
    Output2=(Output[0,2]*255).astype(np.uint8)
    
    OutputWhole1=np.zeros(Shape,dtype=np.uint8)
    OutputWhole=np.zeros(Shape,dtype=np.uint8)
    OutputWhole1[MaximumBbox[0]:MaximumBbox[3],MaximumBbox[1]:MaximumBbox[4],MaximumBbox[2]:MaximumBbox[5]]=Output1
    OutputWhole2=np.zeros(Shape,dtype=np.uint8)
    OutputWhole2[MaximumBbox[0]:MaximumBbox[3],MaximumBbox[1]:MaximumBbox[4],MaximumBbox[2]:MaximumBbox[5]]=Output2
    OutputWhole1[OutputWhole1>=128]=255
    OutputWhole1[OutputWhole1<128]=0
    OutputWhole2[OutputWhole2>=128]=255
    OutputWhole2[OutputWhole2<128]=0
    #Image1=sitk.GetImageFromArray(Image1)
    #Image2=sitk.GetImageFromArray(Image2)
    #Image1.SetSpacing([1.0,1.0,2.0])
    #Image2.SetSpacing([1.0,1.0,2.0])
    #sitk.WriteImage(Image1,'./Image1.mhd')
    #sitk.WriteImage(Image2,'./Image2.mhd')
    OutputWhole[OutputWhole1>=128]=1
    OutputWhole[OutputWhole2>=128]=2
    OutputWhole1/=255
    OutputWhole2/=255
    Loss=2*np.sum(Output[0,1]*Label[0,1])/(np.sum(Output[0,1])+np.sum(Label[0,1]))+2*np.sum(Output[0,2]*Label[0,2])/(np.sum(Output[0,2])+np.sum(Label[0,2]))
    Loss/=2
    Loss=1-Loss
    OutputWhole1=sitk.GetImageFromArray(OutputWhole1)
    OutputWhole2=sitk.GetImageFromArray(OutputWhole2)
    OutputWhole1.SetSpacing([1.0,1.0,2.0])
    OutputWhole2.SetSpacing([1.0,1.0,2.0])
    
    
    for Rid in range(len(RoIs)):
        color=(Rid+1,Rid+1,Rid+1)
        
        Coord=RoIs[Rid]*np.array([4,8,8,4,8,8])+np.array([MaximumBbox[0],MaximumBbox[1],MaximumBbox[2],MaximumBbox[0],MaximumBbox[1],MaximumBbox[2]])
        
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
    OutputWhole.SetSpacing([1.0,1.0,2.0])
    sitk.WriteImage(OutputWhole,'/Data'+Project+'/'+Patient+'Pred.mhd')
    sitk.WriteImage(OutputWhole1,'/Data'+Project+'/'+Patient+'Pred1.mhd')
    sitk.WriteImage(OutputWhole2,'/Data'+Project+'/'+Patient+'Pred2.mhd')
    return Loss
Patient='1'
lr=0.0001
Model=RU_Net(n_channels=2,n_classes=3,inplace=inplace)
Model=Model.to(GPU)
optimizer = optim.Adam(Model.parameters(),lr=lr)#, momentum=0.9, weight_decay=0.0005)


PatientNames_pre=os.listdir(Root)
PatientNames=[]
for i in range(len(PatientNames_pre)):
    if PatientNames_pre[i].endswith('ImgManWin.mhd'):
        PatientNames.append(PatientNames_pre[i][0:-13])
PatientNames=sorted(PatientNames)
print PatientNames
NumVal = len(PatientNames)//4
NumTrain=len(PatientNames)-NumVal
ValPatient=PatientNames[ValQuarter*NumVal:(ValQuarter+1)*NumVal:]
TrainPatient1=PatientNames[0:ValQuarter*NumVal]
TrainPatient2=PatientNames[(ValQuarter+1)*NumVal:]
TrainPatient=TrainPatient1+TrainPatient2
print TrainPatient
Load=True
try:
    Model.load_state_dict(torch.load('./'+Project+'Weights/'+Project+'Params_'+str(ValQuarter)+'.pkl'))
except:
#Initialization
    for epoch in range(5):
        for iteration in range(NumTrain):
            Patient=TrainPatient[random.randint(0,NumTrain-1)]
            Image,Label,Shape,MaximumBbox=GetImage(Patient)
            PredRoI=Model.forward_RoI_Loc(Image,Label)
            optimizer.zero_grad()
            loss=MultiClassDiceLossFunc(PredRoI[0],PredRoI[1])
            loss.backward()
            optimizer.step()
            print loss
torch.save(Model.state_dict(), './'+Project+'Weights/'+Project+'Params_'+str(ValQuarter)+'.pkl')
#co-training
Lowest=1
for epoch in range(10):
    for iteration in range(NumTrain):
        Patient=TrainPatient[random.randint(0,NumTrain-1)]
        if Patient=='41' or Patient=='9' or Patient=='16':
            continue
        Image,Label,Shape,MaximumBbox=GetImage(Patient)
        PredRoI=Model.forward_RoI_Loc(Image,Label)
        optimizer.zero_grad()
        loss=MultiClassDiceLossFunc(PredRoI[0],PredRoI[1])
        loss.backward()
        optimizer.step()
        #print 'global loss=',loss
        PredSeg=Model.train_forward(Image,Label)
        loss=torch.zeros([1]).cuda(GPU)
        #loss=torch
        for i in range(len(PredSeg[0])):
            loss+=MultiClassDiceLossFunc(PredSeg[0][i],PredSeg[1][i])
        loss/=len(PredSeg[0])
        loss.backward()
        optimizer.step()
        print 'local loss=',loss
    Loss=0
    for iteration in range(NumVal):
        PatientVal=ValPatient[iteration]
        Loss_temp=Predict(PatientVal)
        Loss+=Loss_temp
        print PatientVal,' Loss=',Loss_temp
    Loss/=NumVal
    if Loss<Lowest:
        print 'Loss improved from ',Lowest,'to ',Loss
        torch.save(Model.state_dict(), './'+Project+'Weights/'+Project+'Params_'+str(ValQuarter)+'.pkl')
        print 'saved to ./'+Project+'Weights/'+Project+'Params_'+str(ValQuarter)+'.pkl'
        Lowest=Loss
    else:
        print 'not improved'
    print '\n\nValLoss=',Loss

Model.load_state_dict(torch.load('./'+Project+'Weights/'+Project+'Params_'+str(ValQuarter)+'.pkl'))
Loss=0
for iteration in range(NumVal):
    PatientVal=ValPatient[iteration]
    Loss_temp=Predict(PatientVal)
    Loss+=Loss_temp
    print PatientVal,' Loss=',Loss_temp
Loss/=NumVal
if Loss<Lowest:
    print 'Loss improved from ',Lowest,'to ',Loss
    torch.save(Model.state_dict(), './'+Project+'Weights/'+Project+'Params_'+str(ValQuarter)+'.pkl')
    print 'saved to ./'+Project+'Weights/'+Project+'Params_'+str(ValQuarter)+'.pkl'
    Lowest=Loss
else:
    print 'not improved'
print '\n\nValLoss=',Loss




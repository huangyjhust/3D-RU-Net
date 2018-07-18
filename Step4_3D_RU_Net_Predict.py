#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 13:03:37 2018

@author: HuangyjSJTU
"""

import numpy as np
from Step3_3D_RU_Net_Train import RU_Net

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
from skimage import data,util,transform

inplace=True
ClassIndex={0:'Background', 1:'Cancer'}
NumRoIsTrain={'Background':0,'Cancer':2}
NumRoIsTest={'Background':0,'Cancer':1}
ResRates={0:'HighRes',1:'MidRes',2:'LowRes'}
ToSpacing={'HighRes':[1,1,4],'MidRes':[1.5,1.5,4],'LowRes':[2,2,4]}
DownSample=[2,4,4]
Root='./Data/Test/'

def GetImage(Patient,ResRate):
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
        
    return Image,Shape,MaximumBbox


Project='Colon'
Root='../Data/Normalized/'
GPU='cuda:0'

if __name__=='__main__':
    PatientNames=os.listdir(Root)
    PatientNames=sorted(PatientNames)
    print PatientNames
    NumVal = len(PatientNames)//4
    ValPatient=PatientNames[ValQuarter*NumVal:(ValQuarter+1)*NumVal:]
    print ValPatient
    ModelPyramid=[]
    for i in range(len(ResRates)):
        ModelPyramid.append(RU_Net(n_channels=1,n_classes=len(ClassIndex),inplace=inplace))
        ModelPyramid[i]=ModelPyramid[i].to(GPU)
        ModelPyramid[i].load_my_state_dict(torch.load('./'+Project+'Weights/'+Project+ResRates[i]+'Params_'+str(ValQuarter)+'.pkl'))
		ModelPyramid[i].train(mode=True)
    for i in range(NumVal):
        Patient=ValPatient[i]
        ImagePyramid=[]
        ShapePyramid=[]
        BboxPyramid=[]
        PredPyramid=[]
        for j in range(len(ResRates)):
            Image,Shape,MaximumBbox=GetImage(Patient,j)
            if j==0:
                NewSize=Shape
            ImagePyramid.append(Image)
            ShapePyramid.append(Shape)
            BboxPyramid.append(MaximumBbox)
        time1=time.time()
        for j in range(len(ResRates)):
            Shape=ShapePyramid[j]
            MaximumBbox=BboxPyramid[j]
            Pred=ModelPyramid[j].forward(ImagePyramid[j])
            RegionOutput=np.zeros(ImagePyramid[j].shape)
            RegionWeight=np.zeros(ImagePyramid[j].shape)+0.001
            RoIs=Pred[2]
            for k in range(len(Pred[0])):
                Coord=RoIs[k]*np.array([2,4,4,2,4,4])
                Weight=np.ones(np.asarray(Pred[0][k][0,1:2].shape))
                RegionOutput[0,:,Coord[0]:Coord[3],Coord[1]:Coord[4],Coord[2]:Coord[5]]+=Pred[0][k][0,1:2]
                RegionWeight[0,:,Coord[0]:Coord[3],Coord[1]:Coord[4],Coord[2]:Coord[5]]+=Weight
            RegionOutput/=RegionWeight
            OutputWhole=np.zeros(Shape,dtype=np.uint8)
            OutputWhole[MaximumBbox[0]:MaximumBbox[3],MaximumBbox[1]:MaximumBbox[4],MaximumBbox[2]:MaximumBbox[5]]=(RegionOutput[0,0]*255).astype(np.uint8)
            Spacing = ToSpacing[ResRates[j]]
            NewSpacing = [1.0,1.0,4.0]
            Shape=Shape[::-1]
                  
            OutputWhole=sitk.GetImageFromArray(OutputWhole)
            OutputWhole.SetSpacing(Spacing)
            Resample = sitk.ResampleImageFilter()
            Resample.SetSize(NewSize[::-1])
            Resample.SetInterpolator(sitk.sitkLinear)
            Resample.SetOutputSpacing(NewSpacing)
            OutputWhole = Resample.Execute(OutputWhole)
            OutputWhole=sitk.GetArrayFromImage(OutputWhole)
            PredPyramid.append(OutputWhole)

        EnsemblePred=np.zeros(NewSize)
        for j in range(len(ResRates)):
            EnsemblePred+=PredPyramid[j].astype(np.float)
        EnsemblePred/=3
        EnsemblePred[EnsemblePred>=128]=255
        EnsemblePred[EnsemblePred<128]=0
        EnsemblePred/=255
        EnsemblePred=sitk.GetImageFromArray(EnsemblePred)
        sitk.WriteImage(EnsemblePred,'./Output'+Project+'/'+Patient+'/Final.mhd')
        print "time used: ",time.time()-time1
            
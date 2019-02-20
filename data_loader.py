#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 21:27:54 2019

@author: customer
"""
import numpy as np
import random
import SimpleITK as sitk
from skimage.measure import label,regionprops
from skimage import filters
import torch
#Maximum Bbox Cropping to Reduce Image Dimension
def MaxBodyBox(input):
    Otsu=filters.threshold_otsu(input[input.shape[0]//2])
    Seg=np.zeros(input.shape)
    Seg[input>=Otsu]=255
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
    return Otsu,MaximumBbox

def DataLoader(Patient,opt,Subset='Train'):
    assert Subset in ['Train','Valid','Test']    
    #Image Loading
    ImageInput=sitk.ReadImage(opt.DATA_ROOT+'/'+Subset+'/'+Patient+'/HighRes/'+'Image_2.mhd')
    ImageInput=sitk.GetArrayFromImage(ImageInput)
    RegionLabel=sitk.ReadImage(opt.DATA_ROOT+'/'+Subset+'/'+Patient+'/HighRes/'+'Label.mhd')
    RegionLabel=sitk.GetArrayFromImage(RegionLabel)
    ContourLabel=sitk.ReadImage(opt.DATA_ROOT+'/'+Subset+'/'+Patient+'/HighRes/'+'Contour.mhd')
    ContourLabel=sitk.GetArrayFromImage(ContourLabel)
    #Orig Shape Backup
    Shape=ImageInput.shape
    #Body Bbox Compute
    Otsu,MaximumBbox=MaxBodyBox(ImageInput)
           

    #Apply BodyBbox Cropping
    ImageInput=ImageInput[MaximumBbox[0]:MaximumBbox[3],MaximumBbox[1]:MaximumBbox[4],MaximumBbox[2]:MaximumBbox[5]]    
    RegionLabel=RegionLabel[MaximumBbox[0]:MaximumBbox[3],MaximumBbox[1]:MaximumBbox[4],MaximumBbox[2]:MaximumBbox[5]]
    ContourLabel=ContourLabel[MaximumBbox[0]:MaximumBbox[3],MaximumBbox[1]:MaximumBbox[4],MaximumBbox[2]:MaximumBbox[5]]    
    
    if Subset=='Train':
        Xinvert=random.randint(0,1)
        IntensityScale=random.uniform(0.9,1.1)
    else:
        Xinvert=False
        IntensityScale=1
    #Apply Intensity Jitterring 
    ImageInput=((ImageInput-128.0)*IntensityScale+128.0)/255
    ImageInput[ImageInput>1]=1
    ImageInput[ImageInput<0]=0
    #Apply Random Flipping
    if Xinvert:
        ImageInput=ImageInput[:,:,::-1].copy()
        RegionLabel=RegionLabel[:,:,::-1].copy()
        ContourLabel=ContourLabel[:,:,::-1].copy()
        
    #To Tensor
    ImageTensor=np.zeros([1,1,ImageInput.shape[0],ImageInput.shape[1],ImageInput.shape[2]])
    ImageTensor[0,0]=ImageInput
    ImageTensor=ImageTensor.astype(np.float)
    ImageTensor=torch.from_numpy(ImageTensor)
    ImageTensor=ImageTensor.float()
    ImageTensor = ImageTensor.to(device=opt.GPU)

    RegionLabelTensor=np.zeros([1,2,ImageInput.shape[0],ImageInput.shape[1],ImageInput.shape[2]])
    RegionLabelTensor[0,1]=RegionLabel
    RegionLabelTensor[0,0]=1-RegionLabel
    RegionLabelTensor=torch.from_numpy(RegionLabelTensor)
    RegionLabelTensor=RegionLabelTensor.float()
    RegionLabelTensor=RegionLabelTensor.to(device=opt.GPU)
    
    ContourLabelTensor=np.zeros([1,2,ImageInput.shape[0],ImageInput.shape[1],ImageInput.shape[2]])
    ContourLabelTensor[0,1]=ContourLabel
    ContourLabelTensor[0,0]=1-ContourLabel
    ContourLabelTensor=torch.from_numpy(ContourLabelTensor)
    ContourLabelTensor=ContourLabelTensor.float()
    ContourLabelTensor=ContourLabelTensor.to(device=opt.GPU)
    
    
    return ImageTensor,RegionLabelTensor,ContourLabelTensor,Shape,MaximumBbox

def ArbitraryDataLoader(Patient,opt,Subset='Test'): 
    #Image Loading
    ImageInput=sitk.ReadImage(opt.DATA_ROOT+'/'+Subset+'/'+Patient+'/HighRes/'+'Image_2.mhd')
    ImageInput=sitk.GetArrayFromImage(ImageInput)/255.0
    #Orig Shape Backup
    Shape=ImageInput.shape
    #Body Bbox Compute
    Otsu,MaximumBbox=MaxBodyBox(ImageInput)
           

    #Apply BodyBbox Cropping
    ImageInput=ImageInput[MaximumBbox[0]:MaximumBbox[3],MaximumBbox[1]:MaximumBbox[4],MaximumBbox[2]:MaximumBbox[5]]    
        
    #To Tensor
    ImageTensor=np.zeros([1,1,ImageInput.shape[0],ImageInput.shape[1],ImageInput.shape[2]])
    ImageTensor[0,0]=ImageInput
    ImageTensor=ImageTensor.astype(np.float)
    ImageTensor=torch.from_numpy(ImageTensor)
    ImageTensor=ImageTensor.float()
    ImageTensor = ImageTensor.to(device=opt.GPU)


    
    
    return ImageTensor,Shape,MaximumBbox   
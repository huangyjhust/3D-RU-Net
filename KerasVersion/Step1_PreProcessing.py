#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 13:26:15 2018

@author: HuangyjSJTU
"""
import SimpleITK as sitk
import numpy as np
import sys
import os
sys.path.append('./lib/')
import matplotlib.pyplot as pl
from PIL import Image as Img
from FindFiles import findfiles
import dicom
import cv2
from skimage import filters
#For intensity normalization

DataRoot='../Data/send/'
ModelName='/t2-fov/'
ManualNormalize=True
ResRate=['HighRes','MidRes','LowRes']
ToSpacing={'HighRes':[1,1,4],'MidRes':[1.5,1.5,4],'LowRes':[2,2,4]}

def ReadImageAndLabel(CasePath):
    #Reading Images
    Image=sitk.ReadImage(CasePath+'Image.mhd')
    Spacing=Image.GetSpacing()
    Origin = Image.GetOrigin()
    Direction = Image.GetDirection()
    
    Reader = sitk.ImageSeriesReader()

    #Reading Labels
    name=findfiles(CasePath+'label/','*.PNG')
    name=sorted(name)
    for i in range(len(name)):
        name[i]=CasePath+'label/'+name[i]
    #print name
    #Sometimes labels are inverted along Z axis and should be rectified in this dataset
    name=name[::-1]
    for i in range(len(name)):
        print name[i]+'\n'
    Reader.SetFileNames(name)
    Label = Reader.Execute()
    LabelArray=sitk.GetArrayFromImage(Label)
    LabelArray=((255-LabelArray[:,:,:,1])).astype(np.uint8)/255
    Label=sitk.GetImageFromArray(LabelArray)
    Label.SetSpacing(Spacing)
    Label.SetOrigin(Origin)
    Label.SetDirection(Direction)
    return Image,Label

def Resampling(Image,Label):
    Size=Image.GetSize()
    Spacing=Image.GetSpacing()
    Origin = Image.GetOrigin()
    Direction = Image.GetDirection()
    ImagePyramid=[]
    LabelPyramid=[]
    for i in range(3):
        NewSpacing = ToSpacing[ResRate[i]]
        NewSize=[int(Size[0]*Spacing[0]/NewSpacing[0]),int(Size[1]*Spacing[1]/NewSpacing[1]),int(Size[2]*Spacing[2]/NewSpacing[2])]       
        Resample = sitk.ResampleImageFilter()
        Resample.SetOutputDirection(Direction)
        Resample.SetOutputOrigin(Origin)
        Resample.SetSize(NewSize)
        Resample.SetInterpolator(sitk.sitkLinear)
        Resample.SetOutputSpacing(NewSpacing)
        NewImage = Resample.Execute(Image)
        ImagePyramid.append(NewImage)
        
        Resample = sitk.ResampleImageFilter()
        Resample.SetOutputDirection(Direction)
        Resample.SetOutputOrigin(Origin)
        Resample.SetSize(NewSize)
        Resample.SetOutputSpacing(NewSpacing)
        Resample.SetInterpolator(sitk.sitkNearestNeighbor)
        NewLabel = Resample.Execute(Label)
        LabelPyramid.append(NewLabel)
    return ImagePyramid,LabelPyramid

#We shift the mean value to enhance the darker side
UpperBound=1.0
LowerBound=-4.0

def Normalization(Image):
    Spacing=Image.GetSpacing()
    Origin = Image.GetOrigin()
    Direction = Image.GetDirection()
    Array=sitk.GetArrayFromImage(Image)
    Mask=Array.copy()
    otsu=filters.threshold_otsu(Array)
    Mask[Array<otsu]=0
    Mask[Array>=otsu]=1
#    for i in range(Mask.shape[0]):
#        pl.imshow(Mask[i],cmap='gray')
#        pl.show()
    Avg=np.average(Array,weights=Mask)
    Std=np.sqrt(np.average(abs(Array - Avg)**2,weights=Mask))
    Array=(Array.astype(np.float32)-Avg)/Std
    Array[Array>UpperBound]=UpperBound
    Array[Array<LowerBound]=LowerBound 
    Array=((Array.astype(np.float64)-np.min(Array))/(np.max(Array)-np.min(Array))*255).astype(np.uint8)
    Image=sitk.GetImageFromArray(Array)
    Image.SetDirection(Direction)
    Image.SetOrigin(Origin)
    Image.SetSpacing(Spacing)
    return Image

if __name__=='__main__':
    Patient='ForTest'
    Image,Label=ReadImageAndLabel('/media/customer/Disk1/3D RU-Net_Keras/Data/'+Patient+'/')
    Image=Normalization(Image)
    ImagePyramid,LabelPyramid=Resampling(Image,Label)
    for i in range(3):
        sitk.WriteImage(ImagePyramid[i],'../Data/'+Patient+ResRate[i]+'Image.mhd')
        sitk.WriteImage(LabelPyramid[i],'../Data/'+Patient+ResRate[i]+'Label.mhd')
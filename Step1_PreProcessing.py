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
from skimage.measure import label,regionprops
#For intensity normalization

DataRoot='../Data/send/'
ModelName='/t2-fov/'
ManualNormalize=True
ResRate=['HighRes','MidRes','LowRes']
ToSpacing={'HighRes':[1,1,4],'MidRes':[1.5,1.5,4],'LowRes':[2,2,4]}

def ReadImageAndLabel(CasePath,inverted=False):
    #Reading Images
    #Image=sitk.ReadImage(CasePath+'Image.mhd')
    Reader = sitk.ImageSeriesReader()
    name=findfiles(CasePath+'img/','*.dcm')
    for i in range(len(name)):
        name[i]=int(name[i][0:-4])
    name=sorted(name)
    name=name[::-1]
    for i in range(len(name)):
        #print name[i],'\n'
        name[i]=CasePath+'img/'+str(name[i])+'.dcm'
 
    #print name
    #Sometimes labels are inverted along Z axis and should be rectified in this dataset
    #name=name[::-1]
    Reader.SetFileNames(name)
    Image = Reader.Execute()
    Spacing=Image.GetSpacing()
    Origin = Image.GetOrigin()
    Direction = Image.GetDirection()
    


    #Reading Labels
    name=findfiles(CasePath+'label/','*.PNG')
    name=sorted(name)
    for i in range(len(name)):
        name[i]=CasePath+'label/'+name[i]
    #print name
    #Sometimes labels are inverted along Z axis and should be rectified in this dataset
    if inverted:
        pass
    else:
        name=name[::-1]
#    for i in range(len(name)):
#        print name[i]+'\n'
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
    Array_new=Array.copy()
    Array_new+=np.min(Array_new)
    Array_new=Array_new[Array_new.shape[0]/2-5:Array_new.shape[0]/2+5]
    Mask=Array_new.copy()
    for i in range(Array_new.shape[0]):
        otsu=filters.threshold_otsu(Array_new[i])
        Mask[i][Array_new[i]<0.5*otsu]=0
        Mask[i][Array_new[i]>=0.5*otsu]=1
    MaskSave=sitk.GetImageFromArray(Mask)
    MaskSave=sitk.BinaryDilate(MaskSave,10)
    MaskSave=sitk.BinaryErode(MaskSave,10)
    Mask=sitk.GetArrayFromImage(MaskSave)
    #MaskSave=sitk.BinaryFillhole(MaskSave,foregroundValue=1)
#    Filter=sitk.BinaryFillholeImageFilter()
#    Filter.SetForegroundValue(1)
#    Filter.Execute(MaskSave)

#    Filter=sitk.MedianImageFilter()
#    Filter.SetRadius(3)
#    Filter.Execute(MaskSave)
    
#    for i in range(Mask.shape[0]):
#        pl.imshow(Mask[i],cmap='gray')
#        pl.show()
    Avg=np.average(Array[Array_new.shape[0]/2-5:Array_new.shape[0]/2+5],weights=Mask)
    Std=np.sqrt(np.average(abs(Array[Array_new.shape[0]/2-5:Array_new.shape[0]/2+5] - Avg)**2,weights=Mask))
    Array=(Array.astype(np.float32)-Avg)/Std
    Array[Array>UpperBound]=UpperBound
    Array[Array<LowerBound]=LowerBound 
    Array=((Array.astype(np.float64)-np.min(Array))/(np.max(Array)-np.min(Array))*255).astype(np.uint8)
    Image=sitk.GetImageFromArray(Array)
    Image.SetDirection(Direction)
    Image.SetOrigin(Origin)
    Image.SetSpacing(Spacing)
    return Image,MaskSave

if __name__=='__main__':
    PatientNames=os.listdir('../Data/send/')
    PatientNames=sorted(PatientNames)
    for i in range(len(PatientNames)):
        PatientName=PatientNames[i]#'ForTest'
        print PatientName
        Image,Label=ReadImageAndLabel('../Data/send/'+PatientName+'/t2-fov/')
        Image,Mask=Normalization(Image)
        ImagePyramid,LabelPyramid=Resampling(Image,Label)
        for i in range(3):
            if os.path.exists('../Data/Normalized/'+PatientName)==False:
                os.mkdir('../Data/Normalized/'+PatientName)
            if os.path.exists('../Data/Normalized/'+PatientName+'/'+ResRate[i])==False:
                os.mkdir('../Data/Normalized/'+PatientName+'/'+ResRate[i])
            sitk.WriteImage(Mask,'../Data/Normalized/'+PatientName+'/'+'BodyMask.mhd')
            sitk.WriteImage(ImagePyramid[i],'../Data/Normalized/'+PatientName+'/'+ResRate[i]+'/'+'Image.mhd')
            sitk.WriteImage(LabelPyramid[i],'../Data/Normalized/'+PatientName+'/'+ResRate[i]+'/'+'Label.mhd')
    InvertPatientNames=['10262598']#['10156691','10251705','10255379','10259589','10249293','10257949','10258893','10260041','10264476']
    for i in range(len(InvertPatientNames)):
        PatientName=InvertPatientNames[i]#'ForTest'
        Image,Label=ReadImageAndLabel('../Data/send/'+PatientName+'/t2-fov/',inverted=True)
        Image,Mask=Normalization(Image)
        ImagePyramid,LabelPyramid=Resampling(Image,Label)
        for i in range(3):
            if os.path.exists('../Data/Normalized/'+PatientName)==False:
                os.mkdir('../Data/Normalized/'+PatientName)
            sitk.WriteImage(Mask,'../Data/Normalized/'+PatientName +'/'+'BodyMask.mhd')
            sitk.WriteImage(ImagePyramid[i],'../Data/Normalized/'+PatientName+'/'+ResRate[i]+'/'+'Image.mhd')
            sitk.WriteImage(LabelPyramid[i],'../Data/Normalized/'+PatientName+'/'+ResRate[i]+'/'+'Label.mhd')
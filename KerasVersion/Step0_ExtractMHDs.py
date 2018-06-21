#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 13:43:10 2018

@author: customer
"""

#For patients' privacy protection
#Since Image.mhd is already extracted, this script is for understanding and doesn't need to be executed.
import SimpleITK as sitk
import numpy as np

def mhdExtract(CasePath):
    Reader = sitk.ImageSeriesReader()
    DicomNames=Reader.GetGDCMSeriesFileNames(CasePath)
    Reader.SetFileNames(DicomNames)
    Image = Reader.Execute()
    #print Image.ImagePositionPatient[2]
    #Image=sitk.Cast(Image,sitk.sitkUInt16)
    Spacing = Image.GetSpacing()
    Origin = Image.GetOrigin()
    Direction = Image.GetDirection()
    Array = sitk.GetArrayFromImage(Image)
    
    Extracted=sitk.GetImageFromArray(Array)    
    Extracted.SetDirection(Direction)
    Extracted.SetOrigin(Origin)
    Extracted.SetSpacing(Spacing)
    return Extracted

Image=mhdExtract('../Data/ForTest/t2-fov/img/')
sitk.WriteImage(Image,'../Data/ForTest/Image.mhd')
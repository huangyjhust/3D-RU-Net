# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 10:25:13 2017

@author: Administrator
"""
#This is an illustration of how contour labels for training are prepared.
#As the training data is not fully released, it's not executable and only for understanding now.
import SimpleITK as sitk
import numpy as np
import sys
import os


sys.path.append('./lib/')
import cv2
ResRate='MidRes'
Root='../Data/MHDs/'
Target='Label'
if __name__=='__main__':
    PatientNames=os.listdir(Root)
    for i in range(len(PatientNames)):
        Patient=PatientNames[i]
        Image=sitk.ReadImage(Root+Patient+'/'+ResRate+'Image.mhd')
        Spacing = Image.GetSpacing()
        Origin = Image.GetOrigin()
        Direction = Image.GetDirection()
        
        Image=sitk.GetArrayFromImage(Image)
        Label=sitk.ReadImage(Root+Patient+'/'+ResRate+Target+'.mhd')
        Label=sitk.GetArrayFromImage(Label)
        #Erode and Subtraction for contour label generation.
        #It works but may not be optimal.
        Contour=np.zeros(Image.shape,dtype=np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3)) 
        for z in range(Image.shape[0]):
            if np.sum(Label[z]>0):
                LabelErode=Label[z]-cv2.erode(Label[z],kernel)
                Contour[z]=LabelErode
        Contour=sitk.GetImageFromArray(Contour)
        Contour.SetOrigin(Origin)
        Contour.SetSpacing(Spacing)
        Contour.SetDirection(Direction)
        sitk.WriteImage(Contour,Root+Patient+'/'+ResRate+'Contour.mhd')
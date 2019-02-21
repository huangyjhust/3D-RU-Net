#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 13:03:37 2018

@author: customer
"""

import numpy as np
from model import RU_Net
from train import Config
from data_loader import ArbitraryDataLoader

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



def Predict(Model,ImageTensor,Shape,MaximumBbox):
    with torch.no_grad():
        PredSeg=Model.forward(ImageTensor)
    RegionOutput=np.zeros(ImageTensor.shape)
    RegionWeight=np.zeros(ImageTensor.shape)+0.001
    RoIs=PredSeg[2]
    #apply RoI predictions to a body-cropped large volume container
    #average predictions if RoIs are overlapped
    for i in range(len(PredSeg[0])):
        Coord=RoIs[i]*np.array([2,4,4,2,4,4])
        Weight=np.ones(np.asarray(PredSeg[0][i][0].shape))
        RegionOutput[0,:,Coord[0]:Coord[3],Coord[1]:Coord[4],Coord[2]:Coord[5]]+=PredSeg[0][i][0,1:].to('cpu').detach().numpy()
        RegionWeight[0,:,Coord[0]:Coord[3],Coord[1]:Coord[4],Coord[2]:Coord[5]]+=Weight[1:]
    RegionOutput/=RegionWeight
    


    return RegionOutput

if __name__=='__main__':

    opt=[Config('RF64'),Config('RF88'),Config('RF112')]
    Models=[RU_Net(opt[0]).to(opt[0].GPU),RU_Net(opt[1]).to(opt[1].GPU),RU_Net(opt[2]).to(opt[2].GPU)]
    for mid,Model in enumerate(Models):
        Model.load_state_dict(torch.load(opt[mid].WEIGHT_PATH))
    Root='./Data/Test/'
    PatientNames=os.listdir(Root)
    PatientNames=sorted(PatientNames)
    NumPatients=len(PatientNames)
    for i in range(NumPatients):
        Patient=PatientNames[i]
        ImageTensor,Shape,MaximumBbox=ArbitraryDataLoader(Patient,opt[0],'Test')
        RegionOutput=np.zeros(ImageTensor.shape)
        
        time1=time.time()
        #Ensemble by averaging predictions
        for j in range(len(Models)):
            RegionOutput+=Predict(Models[j],ImageTensor,Shape,MaximumBbox)
        RegionOutput/=len(Models)
        print("time used: ",time.time()-time1)
        
        #body-cropped volume back to whole volume container
        OutputWhole=np.zeros(Shape,dtype=np.float)
        OutputWhole[MaximumBbox[0]:MaximumBbox[3],MaximumBbox[1]:MaximumBbox[4],MaximumBbox[2]:MaximumBbox[5]]=RegionOutput[0,0]
        #Back to ITK images for storage
        OutputWhole*=255
        OutputWhole=OutputWhole.astype(np.uint8)
        OutputWhole=sitk.GetImageFromArray(OutputWhole)
        OutputWhole.SetSpacing(opt[0].TO_SPACING)
        sitk.WriteImage(OutputWhole,'./Output/'+Patient+'/EnsemblePreds.mhd')
        
            

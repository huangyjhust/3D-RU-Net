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
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import cv2

from data_loader import DataLoader
from model import RU_Net
inplace=True


STAGE_DILATIONS={'RF64':[1,1,1],'RF88':[1,1,2],'RF112':[1,2,2]}
TAG='RF112'
class Config():
    def __init__(self,TAG):
        self.TAG=TAG
        self.STAGE_DILATION=STAGE_DILATIONS[TAG]
        self.DICT_CLASS={0:'Background', 1:'Cancer'}
        self.MAX_ROIS_TEST={'Background':0,'Cancer':10}
        self.MAX_ROIS_TRAIN={'Background':0,'Cancer':2}
        self.MAX_ROI_SIZE=[24,96,96]
        self.TO_SPACING=[1,1,4]
        self.DOWN_SAMPLE=[2,4,4]
        self.DATA_ROOT='./Data/'
        self.INPLACE=True
        self.GPU='cuda:1'
        self.MAX_EPOCHS=50
        self.WEIGHT_PATH='./Weights/'+self.TAG+'.pkl'
        self.TEST_ONLY=False
        self.BASE_CHANNELS=48
opt=Config(TAG)


def MultiClassDiceLossFunc(y_pred,y_true):
    overlap=torch.zeros([1]).cuda(opt.GPU)
    bottom=torch.zeros([1]).cuda(opt.GPU)
    for i in range(1,len(opt.DICT_CLASS.keys())):
        overlap+=torch.sum(y_pred[0,i]*y_true[0,i])
        bottom+=torch.sum(y_pred[0,i])+torch.sum(y_true[0,i])
    return 1-2*(overlap+1e-4)/(bottom+1e-4)
def RoIDiceLossFunc(y_pred,y_true):
    overlap=torch.zeros([1]).cuda(opt.GPU)
    bottom=torch.zeros([1]).cuda(opt.GPU)
    for i in range(len(y_pred)):
        for j in range(1,len(opt.DICT_CLASS.keys())):
            overlap+=torch.sum(y_pred[i][0,j]*y_true[i][0,j])
            bottom+=torch.sum(y_pred[i][0,j])+torch.sum(y_true[i][0,j])
    return (1-2*overlap/bottom)


def Predict(Patient,Subset):
    Image,LabelRegion,LabelContour,Shape,MaximumBbox=DataLoader(Patient,opt,Subset)
    Label=LabelRegion.to('cpu').detach().numpy()

    with torch.no_grad():
        PredSeg=Model.forward(Image)
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
    RegionOutput[RegionOutput>=0.5]=1
    RegionOutput[RegionOutput<0.5]=0
    Loss=1-2*np.sum(RegionOutput[0,1]*Label[0,1])/(np.sum(RegionOutput[0,1])+np.sum(Label[0,1]))
    OutputWhole1=sitk.GetImageFromArray(OutputWhole1)
    OutputWhole1.SetSpacing(opt.TO_SPACING)
    
    OutputWhole2=sitk.GetImageFromArray(OutputWhole2)
    OutputWhole2.SetSpacing(opt.TO_SPACING)
    
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
            
    OutputWhole=sitk.GetImageFromArray(OutputWhole)
    OutputWhole.SetSpacing(opt.TO_SPACING)
    if os.path.exists('./Output/'+Patient)==False:
        os.makedirs('./Output/'+Patient)
    sitk.WriteImage(OutputWhole,'./Output/'+Patient+'/Pred_'+opt.TAG+'.mhd')
    sitk.WriteImage(OutputWhole1,'./Output/'+Patient+'/PredRegion_'+opt.TAG+'.mhd')
    sitk.WriteImage(OutputWhole2,'./Output/'+Patient+'/PredContour'+opt.TAG+'.mhd')
    return Loss,len(RoIs)
def ToTensor(input):
    return 0
if __name__=='__main__':
    lr=0.0001
    Model=RU_Net(opt)
    Model=Model.to(opt.GPU)
    
    optimizer1 = optim.Adam(list(Model.GlobalImageEncoder.parameters()),lr=lr,amsgrad=True)
    
    optimizer2 = optim.Adam(list(Model.GlobalImageEncoder.parameters())+\
                            list(Model.LocalRegionDecoder.parameters()),lr=lr,amsgrad=True)  

    TrainPatient=os.listdir(opt.DATA_ROOT+'Train')
    ValPatient=os.listdir(opt.DATA_ROOT+'Valid')
    TestPatient=os.listdir(opt.DATA_ROOT+'Test')
    NumTrain=len(TrainPatient)
    NumTest=len(TestPatient)
    NumVal=len(ValPatient)

    if not opt.TEST_ONLY:
        try:
            Model.load_state_dict(torch.load(opt.WEIGHT_PATH))
            print('Weights Loaded!')
        except:
            for epoch in range(40):
                Model.train()
                for iteration in range(NumTrain):
                    Model.train()#
                    Patient=TrainPatient[random.randint(0,NumTrain-1)]
                    Image,LabelRegion,LabelContour,Shape,MaximumBbox=DataLoader(Patient,opt,'Train')
                    Label=LabelRegion
                    optimizer1.zero_grad()
                    PredSeg=Model.forward_RoI_Loc(Image,LabelRegion)#Model.train_forward(Image,LabelRegion,LabelContour,UseRoI=True)
                    LossG=MultiClassDiceLossFunc(PredSeg[0],PredSeg[1])
                    LossAll=LossG
                    LossAll.backward()
                    optimizer1.step()
                    LossG=LossG.to('cpu').detach().numpy()
                    print('loss={g=',LossG,'}')
                Loss=[]
            torch.save(Model.state_dict(), opt.WEIGHT_PATH)
        
        #co-training
        Lowest=1
        for epoch in range(opt.MAX_EPOCHS):
            print('Epoch ',str(epoch),'/'+str(opt.MAX_EPOCHS))
            Model.train()#set_training(True)
            for iteration in range(NumTrain):
                Patient=TrainPatient[random.randint(0,NumTrain-1)]
                Image,LabelRegion,LabelContour,Shape,MaximumBbox=DataLoader(Patient,opt,'Train')
                optimizer2.zero_grad()
                PredSeg=Model.TrainForward(Image,LabelRegion,LabelContour)
                LossG=MultiClassDiceLossFunc(PredSeg[-1][0],PredSeg[-1][1])
                LossR=RoIDiceLossFunc(PredSeg[0],PredSeg[2])
                LossC=RoIDiceLossFunc(PredSeg[1],PredSeg[3])
                CWeight=1.0
                LossAll=LossG+LossR+CWeight*LossC
                LossAll.backward()
                optimizer2.step()
                LossG=LossG.to('cpu').detach().numpy()
                LossR=LossR.to('cpu').detach().numpy()
                LossC=LossC.to('cpu').detach().numpy()
                if LossG>0.5:
                    print('Hard Patient=',Patient) 
                print('loss={g=',LossG,',r=',LossR,',c=',LossC,'}')
            Loss=[]
            Model.eval()#set_training(False)
            for iteration in range(NumTest):
                Patient=TestPatient[iteration]
                Loss_temp,NumRoIs=Predict(Patient,'Test')
                Loss+=[Loss_temp]
                print(Patient,' Loss=',Loss_temp)
            Loss=np.mean(np.array(Loss))
            if Loss<Lowest:
                print('Loss improved from ',Lowest,'to ',Loss)
                torch.save(Model.state_dict(), opt.WEIGHT_PATH)
                print('saved to ',opt.WEIGHT_PATH)
                Lowest=Loss
            else:
                print('not improved')
            print('\n\nValLoss=',Loss)
            print('Best Loss=',Lowest)
    else:
        Model.load_state_dict(torch.load(opt.WEIGHT_PATH))
        Model.eval()
        #Lowest=1
        Loss=0
        NumRoIs=0
        for iteration in range(NumTest):
            Patient=TestPatient[iteration]
            Loss_temp,NumRoI=Predict(Patient,'Test')
            NumRoIs+=NumRoI
            Loss+=Loss_temp
            print(Patient,' Loss=',Loss_temp)
        print('Mean RoI = ',NumRoIs/NumTest)
        Loss/=NumTest
        print('\n\nValLoss=',Loss)




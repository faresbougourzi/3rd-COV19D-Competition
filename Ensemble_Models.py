#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 17:28:35 2023

@author: bougourzi
"""

# Bougourzi Fares
from albumentations.pytorch import ToTensorV2
import albumentations as A
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix
import os
from tqdm import tqdm
import numpy as np
import torchvision.transforms as transforms
import torchvision
import torch.optim as optim

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

#############################################
class GlobalMaxPool3d(nn.Module):
    """
    Reduce max over last three dimensions.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.max(dim=-1, keepdim=True)[0]
        x = x.max(dim=-2, keepdim=True)[0]
        x = x.max(dim=-3, keepdim=True)[0]
        return x
    
class ResNetBasicStem(nn.Module):
    def __init__(self):
        super(ResNetBasicStem, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv3d(2, 16, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1], dilation=1, ceil_mode=False)
            )
    def forward(self, x):
        x= self.layer(x)
        
        return x
    
class BottleneckTransform(nn.Module):
    def __init__(self, in_dim, int_dim, out_dim, stride):
        super(BottleneckTransform, self).__init__()
        self.skip =nn.Sequential(nn.Conv3d(in_dim, out_dim, kernel_size=(1, 1, 1), stride=(1, stride, stride), bias=False),
                      nn.BatchNorm3d(out_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))

        self.a = nn.Sequential(
            nn.Conv3d(in_dim, int_dim, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(int_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
            )
        self.b = nn.Sequential(
            nn.Conv3d(int_dim, int_dim, kernel_size=(1, 3, 3), stride=(1, stride, stride), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(int_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
            )
        self.c = nn.Sequential(
            nn.Conv3d(int_dim, out_dim, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False),
            nn.BatchNorm3d(out_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )       
        self.act =  nn.ReLU(inplace=True)       
    def forward(self, x):
        skip = self.skip(x)
        x = self.a(x)
        x = self.b(x)
        x = self.c(x)
        x = self.act(x+skip)
        return x
       
class Resnet3DS(nn.Module):
    def __init__(self):
        super(Resnet3DS, self).__init__()
        self.stem = ResNetBasicStem()
        self.s1 = BottleneckTransform(in_dim = 16, int_dim=16, out_dim = 64, stride=1)
        self.max = nn.MaxPool3d(kernel_size=[2, 1, 1], stride=[2, 1, 1], padding=[0, 0, 0], dilation=1, ceil_mode=False)       
        self.s2 = BottleneckTransform(in_dim = 64, int_dim=32, out_dim = 128, stride=2)
        self.s3 = BottleneckTransform(in_dim = 128, int_dim=64, out_dim = 256, stride=1)
        self.s4 = BottleneckTransform(in_dim = 256, int_dim=128, out_dim = 512, stride=2)
        self.head = nn.Sequential(nn.AdaptiveMaxPool3d((16,14,14)),
                                  nn.Conv3d(512, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
                                  nn.BatchNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),      
                                  nn.ReLU(inplace=True),
                                  nn.MaxPool3d(kernel_size=[4, 2, 2], stride=[4, 2, 2], padding=[0, 0, 0], dilation=1, ceil_mode=False),
                                  nn.Conv3d(256, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
                                  nn.BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                  nn.ReLU(inplace=True),
                                  nn.Conv3d(128,128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
                                  nn.BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                  nn.ReLU(inplace=True),                                  
                                  GlobalMaxPool3d()
                                  # 
                                  )
        self.cls = nn.Linear(128, 4, bias=True)
        
    def forward(self, x):
        x = self.stem(x)
        x = self.s1(x)
        x = self.max(x)
        x = self.s2(x)
        
        x = self.s3(x)
        x = self.s4(x)        
        x = self.head(x)
        x = x.view(x.shape[0], x.shape[1])
  
        return self.cls(x)
####################################

from albumentations.pytorch import ToTensorV2
import torchvision.transforms as transforms
   
from torch.utils import data
import albumentations as A
img_size = 224
test_transforms2 =     A.ReplayCompose([
        A.Resize(height=img_size, width=img_size),    
        A.Normalize(
            mean=[0.0],
            std=[1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2()
    ], additional_targets={'image0': 'image'})

##################################################    
    
class Covid_loader_pt2(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, path, transform=None):
        'Initialization'
        self.list_IDs = list_IDs
        self.path = path
        self.transform = transform

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        pth = self.path
        
        labels = np.load(os.path.join(pth, 'labels.npy'))[ID]
           
        X2  = np.load(os.path.join(pth, str(ID) + '_inf.npy')).astype(np.uint8)          
        X3  = np.load(os.path.join(pth, str(ID) + '_lung2.npy')).astype(np.uint8) 
       
        if self.transform is not None:
            Xs1 = []

            data1 = self.transform(image=np.expand_dims(X2[:,:,0],2), image0=X3[:,:,0])

            img2 = data1['image']
            img3 = data1['image0']
            img = torch.concat([img2, img3], dim = 0)
            Xs1.append(img)

      
            # augmentations = self.transform(image=img1, mask=y1, mask0 = y2)
            for i in range(X2.shape[2]-1):
                image2_data = A.ReplayCompose.replay(data1['replay'], image=np.expand_dims(X2[:,:,i+1],2), image0=X3[:,:,i+1])
                img2 = image2_data['image']
                img3 = image2_data['image0']
                img = torch.concat([img2, img3], dim = 0)
                Xs1.append(img)
     
        Xs1 = torch.stack(Xs1).transpose(0, 1).transpose(1, 2)#.squeeze(dim = 1)        
        Xs1 = F.interpolate(Xs1, [64, 224], mode ='bicubic').transpose(1, 2)
       
        return Xs1, labels      
####################################
vl_path = './NumpysSevSegZ2/Test'
vl_labels = np.load('./NumpysSevSegZ2/Test/labels.npy')

vl_indxs = list(range(len(vl_labels)))
test_set1 = Covid_loader_pt2(
    list_IDs=vl_indxs,
    path=vl_path,
    transform=test_transforms2
)
###########################
test_loader1 = torch.utils.data.DataLoader(test_set1, batch_size = 1)
device = torch.device("cuda:0")

name = './ModelsF/3D/vl/Models/' + str(0)+'_bt.pt'
model = Resnet3DS()
model.load_state_dict(torch.load(name))       
model = model.to(device)

name = './ModelsF/3D/F1/Models/' + str(0)+'_bt.pt'
model1 = Resnet3DS()
model1.load_state_dict(torch.load(name))       
model1 = model1.to(device)

name = './ModelsF/3D/F2/Models/' + str(0)+'_bt.pt'
model2 = Resnet3DS()
model2.load_state_dict(torch.load(name))       
model2 = model2.to(device)

name = './ModelsF/3D/F3/Models/' + str(0)+'_bt.pt'
model3 = Resnet3DS()
model3.load_state_dict(torch.load(name))       
model3 = model3.to(device)

name = './ModelsF/3D/F4/Models/' + str(0)+'_bt.pt'
model4 = Resnet3DS()
model4.load_state_dict(torch.load(name))       
model4 = model4.to(device)

name = './ModelsF/3D/F5/Models/' + str(0)+'_bt.pt'
model5 = Resnet3DS()
model5.load_state_dict(torch.load(name))       
model5 = model5.to(device)

###########################

labels_ts = []
probs_mdl1= np.zeros([len(test_set1),4])
probs_mdl2= np.zeros([len(test_set1),4])
probs_mdl3= np.zeros([len(test_set1),4])
probs_mdl4= np.zeros([len(test_set1),4])
probs_mdl5= np.zeros([len(test_set1),4])
probs_mdl0= np.zeros([len(test_set1),4])

softmax = nn.Softmax(dim=1)
itr = -1
for batch in tqdm(test_loader1): 
    itr += 1
    images1, labels = batch
    images1 = images1.squeeze(dim=1).float().to(device)

    model.eval()
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    model5.eval()
    with torch.no_grad():
        preds0 = model(images1)
        preds1 = model1(images1)
        preds2 = model2(images1) 
        preds3 = model3(images1) 
        preds4 = model4(images1) 
        preds5 = model5(images1)
    probs_mdl0[itr,0:] = softmax(preds0).cpu()      
    probs_mdl1[itr,0:] = softmax(preds1).cpu()
    probs_mdl2[itr,0:] = softmax(preds2).cpu()
    probs_mdl3[itr,0:] = softmax(preds3).cpu()
    probs_mdl4[itr,0:] = softmax(preds4).cpu()
    probs_mdl5[itr,0:] = softmax(preds5).cpu()    
    labels_ts.append(vl_labels[itr])                 
    del images1; del labels  

###########################
predsf0 = np.argmax(probs_mdl0, axis=1)
# total_correct = 0
predsf1 = np.argmax(probs_mdl1, axis=1)
# ######
# total_correct = 0
predsf2 = np.argmax(probs_mdl2, axis=1)
# ######
# total_correct = 0
predsf3 = np.argmax(probs_mdl3, axis=1)
# ######
# total_correct = 0
predsf4 = np.argmax(probs_mdl4, axis=1)
# ######
# total_correct = 0
predsf5 = np.argmax(probs_mdl5, axis=1)
probs_sum =probs_mdl0 +probs_mdl1 +probs_mdl2+ probs_mdl3+probs_mdl4 +probs_mdl5
predsf = np.argmax(probs_sum, axis=1)

preds_all = np.zeros([len(test_set1),15])
preds_all[:,0] = predsf0
preds_all[:,1] = predsf1
preds_all[:,2] = predsf2
preds_all[:,3] = predsf3
preds_all[:,4] = predsf4
preds_all[:,5] = predsf5


########################################################################
########################################################################
########################################################################
########################################################################
########################################################################
########################################################################
########################################################################
########################################################################
########################################################################


from albumentations.pytorch import ToTensorV2
import albumentations as A
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix
import os
from tqdm import tqdm
import numpy as np
import torchvision.transforms as transforms
import torchvision
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

#############################################
img_size = 299

class TwoInputsNet1(nn.Module):
    def __init__(self):
        super(TwoInputsNet1, self).__init__()
        
        self.encoder1 =  nn.Conv2d(32, 3, kernel_size = 3, stride = 1, padding = 1)
        self.encoder2 =  nn.Conv2d(16, 3, kernel_size = 3, stride = 1, padding = 1)
        self.encoder3 =  nn.Conv2d(6, 3, kernel_size = 3, stride = 1, padding = 1)

        self.model = timm.create_model('inception_resnet_v2', pretrained=True)

        self.model.classif = nn.Linear(in_features=1536, out_features=4, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, input1, input2):
        out1 = self.act(self.encoder1(input1))
        out2 = self.act(self.encoder2(input2))
        out1 = self.act(self.encoder3(torch.cat([out1, out2], dim = 1)))
        out1 = self.model(out1)
        return out1
    
class TwoInputsNet(nn.Module):
    def __init__(self):
        super(TwoInputsNet, self).__init__()    
        self.model1 =  TwoInputsNet1()
        self.model2 =  TwoInputsNet1()
        self.model1.model.classif = nn.Identity()
        self.model2.model.classif = nn.Identity()

        self.fc = nn.Sequential(nn.Linear(in_features=1536*2, out_features=1536, bias=True),
                                nn.ReLU(True),
                                nn.Linear(1536, 4))


    def forward(self, input1, input2, inpt1, inpt2):
        out1 = self.model1(input1, input2)
        out2 = self.model2(inpt1, inpt2)
        out1 = self.fc(torch.cat([out1, out2], dim = 1))
        return out1 
########################################################################
from torch.utils import data  
class Covid_loader_pt4(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, path, transform=None):
        'Initialization'
        self.list_IDs = list_IDs
        self.path = path
        self.transform = transform

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        pth = self.path
        
        labels = np.load(os.path.join(pth, 'labels.npy'))[ID]
        
        X1  = np.load(os.path.join(pth, str(ID) + '_inf.npy'))
        X2  = np.load(os.path.join(pth, str(ID) + '_lung2.npy'))
       
        if self.transform is not None:
            X1s = []
            X2s = []
            for i in range(X1.shape[2]):
                X1s.append(self.transform(X1[:,:,i]))
                X2s.append(self.transform(X2[:,:,i]))
        # print(len(Xs))        
        X1s = torch.stack(X1s).squeeze(dim = 1)
        X2s = torch.stack(X2s).squeeze(dim = 1)
        
        X1s1 = F.interpolate(X1s.unsqueeze(dim = 0).unsqueeze(dim = 0), [32, 299, 299])
        X1s2 = F.interpolate(X1s.unsqueeze(dim = 0).unsqueeze(dim = 0), [16, 299, 299])

        X2s1 = F.interpolate(X2s.unsqueeze(dim = 0).unsqueeze(dim = 0), [32, 299, 299])
        X2s2 = F.interpolate(X2s.unsqueeze(dim = 0).unsqueeze(dim = 0), [16, 299, 299])

        return X1s1, X1s2, X2s1, X2s2, labels 
    
######## 2B-InceptResneth #####################################
img_size = 299
test_transforms = transforms.Compose([
        transforms.ToPILImage(mode='L'),
        transforms.Resize((img_size,img_size)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
])
############################
vl_path = './NumpysSevSegZ2/Test'
vl_labels = np.load('./NumpysSevSegZ2/Test/labels.npy')

vl_indxs = list(range(len(vl_labels)))

test_set2 = Covid_loader_pt4(
    list_IDs=vl_indxs,
    path=vl_path,
    transform=test_transforms
)
###########################
test_loader2 = torch.utils.data.DataLoader(test_set2, batch_size = 1)
device = torch.device("cuda:0")

name = './ModelsF/2D/vl/Models/' + str(0)+'_bt.pt'
model = TwoInputsNet()
model.load_state_dict(torch.load(name))       
model = model.to(device)

name = './ModelsF/2D/F1/Models/' + str(0)+'_bt.pt'
model1 = TwoInputsNet()
model1.load_state_dict(torch.load(name))       
model1 = model1.to(device)

name = './ModelsF/2D/F2/Models/' + str(0)+'_bt.pt'
model2 = TwoInputsNet()
model2.load_state_dict(torch.load(name))       
model2 = model2.to(device)

name = './ModelsF/2D/F3/Models/' + str(0)+'_bt.pt'
model3 = TwoInputsNet()
model3.load_state_dict(torch.load(name))       
model3 = model3.to(device)

name = './ModelsF/2D/F4/Models/' + str(0)+'_bt.pt'
model4 = TwoInputsNet()
model4.load_state_dict(torch.load(name))       
model4 = model4.to(device)

name = './ModelsF/2D/F5/Models/' + str(0)+'_bt.pt'
model5 = TwoInputsNet()
model5.load_state_dict(torch.load(name))       
model5 = model5.to(device)

###########################

labels_ts = []
probs_mdl1= np.zeros([len(test_set1),4])
probs_mdl2= np.zeros([len(test_set1),4])
probs_mdl3= np.zeros([len(test_set1),4])
probs_mdl4= np.zeros([len(test_set1),4])
probs_mdl5= np.zeros([len(test_set1),4])
probs_mdl0= np.zeros([len(test_set1),4])

softmax = nn.Softmax(dim=1)
itr = -1
for batch in tqdm(test_loader2): 
    itr += 1
    images1, images2, imag1, imag2, labels = batch
    images1 = images1.squeeze(dim=1).squeeze(dim=1).float().to(device)
    images2 = images2.squeeze(dim=1).squeeze(dim=1).float().to(device)
    imag1 = imag1.squeeze(dim=1).squeeze(dim=1).float().to(device)
    imag2 = imag2.squeeze(dim=1).squeeze(dim=1).float().to(device)  

    model.eval()
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    model5.eval()
    with torch.no_grad():
        preds0 = model(images1, images2, imag1, imag2)
        preds1 = model1(images1, images2, imag1, imag2)
        preds2 = model2(images1, images2, imag1, imag2) 
        preds3 = model3(images1, images2, imag1, imag2) 
        preds4 = model4(images1, images2, imag1, imag2) 
        preds5 = model5(images1, images2, imag1, imag2)
    probs_mdl0[itr,0:] = softmax(preds0).cpu()      
    probs_mdl1[itr,0:] = softmax(preds1).cpu()
    probs_mdl2[itr,0:] = softmax(preds2).cpu()
    probs_mdl3[itr,0:] = softmax(preds3).cpu()
    probs_mdl4[itr,0:] = softmax(preds4).cpu()
    probs_mdl5[itr,0:] = softmax(preds5).cpu()    
    labels_ts.append(vl_labels[itr])                 
    del images1; del labels  

##############

###########################
predsf0 = np.argmax(probs_mdl0, axis=1)
# total_correct = 0
predsf1 = np.argmax(probs_mdl1, axis=1)
# ######
# total_correct = 0
predsf2 = np.argmax(probs_mdl2, axis=1)
# ######
# total_correct = 0
predsf3 = np.argmax(probs_mdl3, axis=1)
# ######
# total_correct = 0
predsf4 = np.argmax(probs_mdl4, axis=1)
# ######
# total_correct = 0
predsf5 = np.argmax(probs_mdl5, axis=1)
probs_sum2 =probs_mdl0 +probs_mdl1 +probs_mdl2+ probs_mdl3+probs_mdl4 +probs_mdl5
predsf2 = np.argmax(probs_sum2, axis=1)
probs_sum3= probs_sum+probs_sum2
predsf3 = np.argmax(probs_sum3, axis=1)

preds_all[:,6] = predsf0
preds_all[:,7] = predsf1
preds_all[:,8] = predsf2
preds_all[:,9] = predsf3
preds_all[:,10] = predsf4
preds_all[:,11] = predsf5
preds_all[:,12] = predsf
preds_all[:,13] = predsf2
preds_all[:,14] = predsf3


saving_models = "./Results1/"
if not os.path.exists(saving_models):
    os.makedirs(saving_models)

import csv
save_mild = saving_models + '/12models.csv'
file1 = open(save_mild, 'w')
writer1 = csv.writer(file1)

        
for i in range(len(preds_all)):
    writer1.writerow([preds_all[i]])        
           
file1.close() 




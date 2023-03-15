#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 22:06:35 2022

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
# from Data_loader3do import Covid_loader_pt3, Covid_loader_pt4
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
img_size = 299
batch_size = 6
epochs = 60

#############################################
class TwoInputsNet(nn.Module):
    def __init__(self):
        super(TwoInputsNet, self).__init__()
        self.encoder1 =  nn.Conv2d(32, 3, kernel_size = 3, stride = 1, padding = 1)
        self.encoder2 =  nn.Conv2d(16, 3, kernel_size = 3, stride = 1, padding = 1)
        self.encoder3 =  nn.Conv2d(6, 3, kernel_size = 3, stride = 1, padding = 1)
        
        self.model = torchvision.models.inception_v3(pretrained=True)
        self.model.fc  =  nn.Linear(2048, 4) 
        self.act = nn.ReLU(inplace=True)

    def forward(self, input1, input2):
        out1 = self.act(self.encoder1(input1))
        out2 = self.act(self.encoder2(input2))
        out1 = self.act(self.encoder3(torch.cat([out1, out2], dim = 1)))
        out1 = self.model(out1)
        return out1
########################################################################
########################################################################
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
        X  = np.load(os.path.join(pth, str(ID) + '_lung.npy'))
       
        if self.transform is not None:
            Xs = []
            for i in range(X.shape[2]):
                Xs.append(self.transform(X[:,:,i]))       
        Xs = torch.stack(Xs).squeeze(dim = 1)
        Xs1 = F.interpolate(Xs.unsqueeze(dim = 0).unsqueeze(dim = 0), [32, 299, 299])
        Xs2 = F.interpolate(Xs.unsqueeze(dim = 0).unsqueeze(dim = 0), [16, 299, 299])
        return Xs1, Xs2, labels 
######## Inception1 #####################################
img_size = 299

test_transforms = transforms.Compose([
        transforms.ToPILImage(mode='L'),
        transforms.Resize((img_size,img_size)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
])

vl_path = './NumpysSevSeg/Val'
vl_labels = np.load('./NumpysSevSeg/Val/labels.npy')

vl_indxs = list(range(len(vl_labels)))
test_set = Covid_loader_pt4(
    list_IDs=vl_indxs,
    path=vl_path,
    transform=test_transforms
)

###########################
validate_loader = torch.utils.data.DataLoader(test_set, batch_size = 1)
name = './Models 3df/Incept_2inp_focal_best.pt'
device = torch.device("cuda:0")

model = TwoInputsNet()
model.load_state_dict(torch.load(name))       
model = model.to(device)

###########################

labels_ts = np.zeros([len(test_set),1])
probs_mdl1= np.zeros([len(test_set),4])

softmax = nn.Softmax(dim=1)
itr = -1
for batch in tqdm(validate_loader): 
    itr += 1
    images1,images2, labels = batch
    images1 = images1.squeeze(dim=1).squeeze(dim=1).float().to(device)
    images2 = images2.squeeze(dim=1).squeeze(dim=1).float().to(device)    
    model.eval()
    with torch.no_grad():
        preds = model(images1, images2)        
    probs_mdl1[itr,0:] = softmax(preds).cpu()
    labels_ts[itr,0] = labels                
    del images1; del labels  

##############
########  Inception3 2################################################################

img_size = 299

test_transforms = transforms.Compose([
        transforms.ToPILImage(mode='L'),
        transforms.Resize((img_size,img_size)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
])

vl_path = './NumpysSevSeg/Val'
vl_labels = np.load('./NumpysSevSeg/Val/labels.npy')

vl_indxs = list(range(len(vl_labels)))
test_set = Covid_loader_pt4(
    list_IDs=vl_indxs,
    path=vl_path,
    transform=test_transforms
)

###########################
validate_loader = torch.utils.data.DataLoader(test_set, batch_size = 1)
name = './Models 3df/Incept_2inp2_focal_best.pt'
device = torch.device("cuda:0")

model = TwoInputsNet()
model.load_state_dict(torch.load(name))       
model = model.to(device)

###########################

labels_ts = np.zeros([len(test_set),1])
probs_mdl2= np.zeros([len(test_set),4])

softmax = nn.Softmax(dim=1)
itr = -1
for batch in tqdm(validate_loader): 
    itr += 1
    images1,images2, labels = batch
    images1 = images1.squeeze(dim=1).squeeze(dim=1).float().to(device)
    images2 = images2.squeeze(dim=1).squeeze(dim=1).float().to(device)    
    model.eval()
    with torch.no_grad():
        preds = model(images1, images2)        
    probs_mdl2[itr,0:] = softmax(preds).cpu()
    labels_ts[itr,0] = labels                
    del images1; del labels
 
########################################################################
########################################################################
####### Inception V4######################################
class TwoInputsNet(nn.Module):
    def __init__(self):
        super(TwoInputsNet, self).__init__()
        # self.encoder1 = DoubleConv(64, 3)
        self.encoder1 =  nn.Conv2d(32, 3, kernel_size = 3, stride = 1, padding = 1)
        self.encoder2 =  nn.Conv2d(16, 3, kernel_size = 3, stride = 1, padding = 1)
        self.encoder3 =  nn.Conv2d(6, 3, kernel_size = 3, stride = 1, padding = 1)

        self.model = timm.create_model('inception_v4', pretrained=True)

        self.model.last_linear = nn.Linear(in_features=1536, out_features=4, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, input1, input2):
        out1 = self.act(self.encoder1(input1))
        out2 = self.act(self.encoder2(input2))
        out1 = self.act(self.encoder3(torch.cat([out1, out2], dim = 1)))
        out1 = self.model(out1)
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
        X  = np.load(os.path.join(pth, str(ID) + '_lung2.npy'))
       
        if self.transform is not None:
            Xs = []
            for i in range(X.shape[2]):
                Xs.append(self.transform(X[:,:,i]))       
        Xs = torch.stack(Xs).squeeze(dim = 1)
        Xs1 = F.interpolate(Xs.unsqueeze(dim = 0).unsqueeze(dim = 0), [32, 299, 299])
        Xs2 = F.interpolate(Xs.unsqueeze(dim = 0).unsqueeze(dim = 0), [16, 299, 299])
        return Xs1, Xs2, labels 
######## Inception1 #####################################
img_size = 299

test_transforms = transforms.Compose([
        transforms.ToPILImage(mode='L'),
        transforms.Resize((img_size,img_size)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
])

vl_path = './NumpysSevSeg/Val'
vl_labels = np.load('./NumpysSevSeg/Val/labels.npy')

vl_indxs = list(range(len(vl_labels)))
test_set = Covid_loader_pt4(
    list_IDs=vl_indxs,
    path=vl_path,
    transform=test_transforms
)

###########################
validate_loader = torch.utils.data.DataLoader(test_set, batch_size = 1)
name = './Models 3df/Incept4_2inp_focal_best.pt'
device = torch.device("cuda:0")

model = TwoInputsNet()
model.load_state_dict(torch.load(name))       
model = model.to(device)

###########################

labels_ts = np.zeros([len(test_set),1])
probs_mdl3= np.zeros([len(test_set),4])

softmax = nn.Softmax(dim=1)
itr = -1
for batch in tqdm(validate_loader): 
    itr += 1
    images1,images2, labels = batch
    images1 = images1.squeeze(dim=1).squeeze(dim=1).float().to(device)
    images2 = images2.squeeze(dim=1).squeeze(dim=1).float().to(device)    
    model.eval()
    with torch.no_grad():
        preds = model(images1, images2)        
    probs_mdl3[itr,0:] = softmax(preds).cpu()
    labels_ts[itr,0] = labels                
    del images1; del labels  

##############
########  Inception 2################################################################

######## Inception1 #####################################
img_size = 299

test_transforms = transforms.Compose([
        transforms.ToPILImage(mode='L'),
        transforms.Resize((img_size,img_size)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
])

vl_path = './NumpysSevSeg/Val'
vl_labels = np.load('./NumpysSevSeg/Val/labels.npy')

vl_indxs = list(range(len(vl_labels)))
test_set = Covid_loader_pt4(
    list_IDs=vl_indxs,
    path=vl_path,
    transform=test_transforms
)

###########################
validate_loader = torch.utils.data.DataLoader(test_set, batch_size = 1)
name = './Models 3df/Incept4_2inp2_focal_best.pt'
device = torch.device("cuda:0")

model = TwoInputsNet()
model.load_state_dict(torch.load(name))       
model = model.to(device)

###########################

labels_ts = np.zeros([len(test_set),1])
probs_mdl4= np.zeros([len(test_set),4])

softmax = nn.Softmax(dim=1)
itr = -1
for batch in tqdm(validate_loader): 
    itr += 1
    images1,images2, labels = batch
    images1 = images1.squeeze(dim=1).squeeze(dim=1).float().to(device)
    images2 = images2.squeeze(dim=1).squeeze(dim=1).float().to(device)    
    model.eval()
    with torch.no_grad():
        preds = model(images1, images2)        
    probs_mdl4[itr,0:] = softmax(preds).cpu()
    labels_ts[itr,0] = labels                
    del images1; del labels
 
########################################################################
########################################################################
########################################################################
########################################################################
####### Inception Resnet2######################################
class TwoInputsNet(nn.Module):
    def __init__(self):
        super(TwoInputsNet, self).__init__()
        
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
####################################

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
        X  = np.load(os.path.join(pth, str(ID) + '_lung2.npy'))
       
        if self.transform is not None:
            Xs = []
            for i in range(X.shape[2]):
                Xs.append(self.transform(X[:,:,i]))       
        Xs = torch.stack(Xs).squeeze(dim = 1)
        Xs1 = F.interpolate(Xs.unsqueeze(dim = 0).unsqueeze(dim = 0), [32, 299, 299])
        Xs2 = F.interpolate(Xs.unsqueeze(dim = 0).unsqueeze(dim = 0), [16, 299, 299])
        return Xs1, Xs2, labels 
######## Inception1 #####################################
img_size = 299

test_transforms = transforms.Compose([
        transforms.ToPILImage(mode='L'),
        transforms.Resize((img_size,img_size)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
])

vl_path = './NumpysSevSeg/Val'
vl_labels = np.load('./NumpysSevSeg/Val/labels.npy')

vl_indxs = list(range(len(vl_labels)))
test_set = Covid_loader_pt4(
    list_IDs=vl_indxs,
    path=vl_path,
    transform=test_transforms
)

###########################
validate_loader = torch.utils.data.DataLoader(test_set, batch_size = 1)
name = './Models 3df/Inceptres2_2inp_focal_best.pt'
device = torch.device("cuda:0")

model = TwoInputsNet()
model.load_state_dict(torch.load(name))       
model = model.to(device)

###########################

labels_ts = np.zeros([len(test_set),1])
probs_mdl5= np.zeros([len(test_set),4])

softmax = nn.Softmax(dim=1)
itr = -1
for batch in tqdm(validate_loader): 
    itr += 1
    images1,images2, labels = batch
    images1 = images1.squeeze(dim=1).squeeze(dim=1).float().to(device)
    images2 = images2.squeeze(dim=1).squeeze(dim=1).float().to(device)    
    model.eval()
    with torch.no_grad():
        preds = model(images1, images2)        
    probs_mdl5[itr,0:] = softmax(preds).cpu()
    labels_ts[itr,0] = labels                
    del images1; del labels  

##############
########  Inception 2################################################################

######## Inception1 #####################################
img_size = 299

test_transforms = transforms.Compose([
        transforms.ToPILImage(mode='L'),
        transforms.Resize((img_size,img_size)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
])

vl_path = './NumpysSevSeg/Val'
vl_labels = np.load('./NumpysSevSeg/Val/labels.npy')

vl_indxs = list(range(len(vl_labels)))
test_set = Covid_loader_pt4(
    list_IDs=vl_indxs,
    path=vl_path,
    transform=test_transforms
)

###########################
validate_loader = torch.utils.data.DataLoader(test_set, batch_size = 1)
name = './Models 3df/Inceptres2_2inp2_focal_best.pt'
device = torch.device("cuda:0")

model = TwoInputsNet()
model.load_state_dict(torch.load(name))       
model = model.to(device)

###########################

labels_ts = np.zeros([len(test_set),1])
probs_mdl6= np.zeros([len(test_set),4])

softmax = nn.Softmax(dim=1)
itr = -1
for batch in tqdm(validate_loader): 
    itr += 1
    images1,images2, labels = batch
    images1 = images1.squeeze(dim=1).squeeze(dim=1).float().to(device)
    images2 = images2.squeeze(dim=1).squeeze(dim=1).float().to(device)    
    model.eval()
    with torch.no_grad():
        preds = model(images1, images2)        
    probs_mdl6[itr,0:] = softmax(preds).cpu()
    labels_ts[itr,0] = labels                
    del images1; del labels
 
########################################################################
########################################################################    


total_correct = 0
predsf1 = np.argmax(probs_mdl1, axis=1)
total_correct = np.sum(predsf1 == np.squeeze(labels_ts))
print('Inceptionv31')
print(total_correct)
print(total_correct/len(test_set))
######
total_correct = 0
predsf2 = np.argmax(probs_mdl2, axis=1)
total_correct = np.sum(predsf2 == np.squeeze(labels_ts))
print('Inceptionv32')
print(total_correct)
print(total_correct/len(test_set))
######
total_correct = 0
predsf3 = np.argmax(probs_mdl3, axis=1)
total_correct = np.sum(predsf3 == np.squeeze(labels_ts))
print('Inceptionv41')
print(total_correct)
print(total_correct/len(test_set))
######
total_correct = 0
predsf4 = np.argmax(probs_mdl4, axis=1)
total_correct = np.sum(predsf4 == np.squeeze(labels_ts))
print('Inceptionv42')
print(total_correct)
print(total_correct/len(test_set))
######
total_correct = 0
predsf5 = np.argmax(probs_mdl5, axis=1)
total_correct = np.sum(predsf5 == np.squeeze(labels_ts))
print('InceptionvRes1')
print(total_correct)
print(total_correct/len(test_set))
######
total_correct = 0
predsf6 = np.argmax(probs_mdl6, axis=1)
total_correct = np.sum(predsf6 == np.squeeze(labels_ts))
print('InceptionvRes2')
print(total_correct)
print(total_correct/len(test_set))
######

probs_sum = probs_mdl1 + probs_mdl2+ probs_mdl3+probs_mdl4 + probs_mdl5+ probs_mdl6
total_correct = 0
predsf = np.argmax(probs_sum, axis=1)
total_correct = np.sum(predsf == np.squeeze(labels_ts))
print('Ensemble5')
print(total_correct)
print(total_correct/len(test_set))



preds_all = np.zeros([len(test_set),8])
preds_all[:,0] = predsf1
preds_all[:,1] = predsf2
preds_all[:,2] = predsf3
preds_all[:,3] = predsf4
preds_all[:,4] = predsf5
preds_all[:,5] = predsf6
preds_all[:,6] = predsf
preds_all[:,7] = np.squeeze(labels_ts)


print('F1scores')
print(f1_score(labels_ts, predsf1, average='macro'))
print(f1_score(labels_ts, predsf2, average='macro'))
print(f1_score(labels_ts, predsf3, average='macro'))
print(f1_score(labels_ts, predsf4, average='macro'))
print(f1_score(labels_ts, predsf5, average='macro'))
print(f1_score(labels_ts, predsf6, average='macro'))

print(f1_score(labels_ts, predsf, average='macro'))
print('Confusion Matrices')
#######  ########  #######   #######
print(confusion_matrix(labels_ts, predsf1))
print(confusion_matrix(labels_ts, predsf2))
print(confusion_matrix(labels_ts, predsf3))
print(confusion_matrix(labels_ts, predsf4))
print(confusion_matrix(labels_ts, predsf5))
print(confusion_matrix(labels_ts, predsf6))
print(confusion_matrix(labels_ts, predsf))


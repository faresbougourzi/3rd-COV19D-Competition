#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 01:35:51 2023

@author: bougourzi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import torchvision

import torchvision.transforms as transforms
import numpy as np

from tqdm import tqdm

import os

from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix
import timm


img_size = 299
batch_size = 6
epochs = 60

############################
# The Data Loader
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

############################
# Train and Test data agmententations
train_transforms = transforms.Compose([
        transforms.ToPILImage(mode='L'),
        transforms.Resize((img_size,img_size)),
        transforms.RandomRotation(degrees = (-10,10)),
        transforms.RandomApply([transforms.ColorJitter(brightness=(0.7,1.5), contrast=0, saturation=0, hue=0)],p=0.1),
        transforms.RandomApply([transforms.ColorJitter(brightness=0, contrast=(0.7,1.5), saturation=0, hue=0)],p=0.1),
        transforms.RandomApply([transforms.ColorJitter(brightness=0, contrast=0, saturation=(0.7,1.5), hue=0)],p=0.1),       
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
])
        
test_transforms = transforms.Compose([
        transforms.ToPILImage(mode='L'),
        transforms.Resize((img_size,img_size)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
])


torch.set_printoptions(linewidth=120)
############################
# Load the Data and Create the Batches

tr_path = './NumpysSevSegZ2/Train5'
vl_path = './NumpysSevSegZ2/Val5'

tr_labels = np.load('./NumpysSevSegZ2/Train5/labels.npy')
vl_labels= np.load('./NumpysSevSegZ2/Val5/labels.npy')

tr_indxs= list(range(len(tr_labels)))
train_set = Covid_loader_pt4(
        list_IDs = tr_indxs, 
        path = tr_path, 
        transform=train_transforms
)

vl_indxs= list(range(len(vl_labels)))
test_set = Covid_loader_pt4(
        list_IDs = vl_indxs, 
        path = vl_path, 
        transform=test_transforms
)

############################
# Loss Function
class FocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, gamma=1.5):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        pt = torch.exp(-BCE_loss)
        F_loss = (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()
############################
# Function for calculating the correct prediction  

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()     
############################
# The Proposed 2B-InceptResnet Architecture
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
    
####################################

device = torch.device("cuda:0")    
train_loader = torch.utils.data.DataLoader(train_set, batch_size = 16, shuffle = True)      
validate_loader = torch.utils.data.DataLoader(test_set, batch_size = 32)
criterion = FocalLoss()

epoch_count = []
Accracy_tr = []
Accracy_ts = []

Acc_best = -2
saving_models = "./Models2D"
if not os.path.exists(saving_models):
    os.makedirs(saving_models)

modl_n = '2D_F5'
modl_nn = 'UNet'
runs = 2

for ii in range(rans):
    model_sp = saving_models +"/" + modl_n + "/Models"
    if not os.path.exists(model_sp):
        os.makedirs(model_sp)
    
    name_model_final = model_sp+ '/' + str(ii) + '_fi.pt'
    name_model_bestF1 =  model_sp+ '/' + str(ii) + '_bt.pt'
    
    model_spR = saving_models +"/" + modl_n + "/Results"
    if not os.path.exists(model_spR):
        os.makedirs(model_spR)
        
    training_tsx = model_spR+ '/' + str(ii) + '.txt'     
    
    Acc_best = -2
    epoch_count = []
    Accracy_tr = []
    Accracy_ts = []
    model = TwoInputsNet().to(device) 
    ##################################   
    for epoch in range(40):
        epoch_count.append(epoch)
        lr = 0.0001
        if epoch > 20:
            lr = 0.00001
        if epoch > 30:
            lr = 0.000001
         
    
        optimizer = optim.Adam(model.parameters(), lr=lr)
        train_loss = 0
        validation_loss = 0
        total_correct_tr = 0
        total_correct_val = 0
        total_correct_tr2 = 0
        total_correct_val2 = 0
    
        label_f1tr = []
        pred_f1tr = []
        
        label_f1tr2 = []
        pred_f1tr2 = []    
    
        for batch in tqdm(train_loader):
            images1, images2, imag1, imag2, labels = batch
            images1 = images1.squeeze(dim=1).squeeze(dim=1).float().to(device)
            images2 = images2.squeeze(dim=1).squeeze(dim=1).float().to(device)
            imag1 = imag1.squeeze(dim=1).squeeze(dim=1).float().to(device)
            imag2 = imag2.squeeze(dim=1).squeeze(dim=1).float().to(device)    
            labels = labels.long().to(device)
    
            torch.set_grad_enabled(True)
            model.train()
            preds = model(images1, images2, imag1, imag2)
            loss = criterion(preds, labels)
    
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            train_loss += loss.item()
            total_correct_tr += get_num_correct(preds, labels)
    
    
            label_f1tr.extend(labels.cpu().numpy().tolist())
            pred_f1tr.extend(preds.argmax(dim=1).tolist())
           
    
            del images1
            del labels
    
        label_f1vl = []
        pred_f1vl = []
        label_f1vl2 = []
        pred_f1vl2 = []    
    
        for batch in tqdm(validate_loader):
            images1, images2, imag1, imag2, labels = batch
            images1 = images1.squeeze(dim=1).squeeze(dim=1).float().to(device)
            images2 = images2.squeeze(dim=1).squeeze(dim=1).float().to(device)
            imag1 = imag1.squeeze(dim=1).squeeze(dim=1).float().to(device)
            imag2 = imag2.squeeze(dim=1).squeeze(dim=1).float().to(device) 
            labels = labels.long().to(device)
    
    
            model.eval()
            with torch.no_grad():
                preds = model(images1, images2, imag1, imag2)
    
            loss = criterion(preds, labels)
    
            validation_loss += loss.item()
            total_correct_val += get_num_correct(preds, labels)
    
            label_f1vl.extend(labels.cpu().numpy().tolist())
            pred_f1vl.extend(preds.argmax(dim=1).tolist())
    
            del images1
            del labels
    
        print('Ep: ', epoch, 'AC_tr: ', total_correct_tr/len(train_set), 'AC_ts: ',
              total_correct_val/len(test_set),'AC_tr2: ', total_correct_tr2/len(train_set), 'AC_ts2: ',
              total_correct_val2/len(test_set), 'Loss_tr: ', train_loss/len(train_set))
        print('CM_tr: ')
        print(confusion_matrix(label_f1tr, pred_f1tr))    
        print('CM_vl: ')
        print(confusion_matrix(label_f1vl, pred_f1vl))
        Acc_best2 = f1_score(label_f1vl, pred_f1vl, average='macro')
        with open(training_tsx, "a") as f:
            print('Epoch {}/{}'.format(epoch, epochs - 1), file=f)
            print('-' * 10, file=f)             
            print('Ep: ', epoch, 'AC_tr: ', total_correct_tr/len(train_set), 'AC_ts: ',
                  total_correct_val/len(test_set),'AC_tr2: ', total_correct_tr2/len(train_set), 'AC_ts2: ',
                  total_correct_val2/len(test_set), 'Loss_tr: ', train_loss/len(train_set), file=f)            
            print('CM_tr: ', file=f)
            print(confusion_matrix(label_f1tr, pred_f1tr), file=f)    
            print('CM_vl: ', file=f)
            print(confusion_matrix(label_f1vl, pred_f1vl), file=f) 
            
        print('MaF1_tr: ', f1_score(label_f1tr, pred_f1tr, average='macro'), 'MaF1_vl: ',
              f1_score(label_f1vl, pred_f1vl, average='macro'))
        with open(training_tsx, "a") as f:
            print('MaF1_tr: ', f1_score(label_f1tr, pred_f1tr, average='macro'), 'MaF1_vl: ',
                  f1_score(label_f1vl, pred_f1vl, average='macro'), file=f)        
    
        Accracy_tr.append(total_correct_tr/len(train_set))
        Accracy_ts.append(Acc_best2)
    
        if Acc_best2 > Acc_best:
            Acc_best = Acc_best2
            torch.save(model.state_dict(), name_model_bestF1)
    
    print(Acc_best)    
    model.load_state_dict(torch.load(name_model_bestF1))
    ###########################
    
    label_f1vl = []
    pred_f1vl = []   
    
    for batch in tqdm(validate_loader):
        images1, images2, imag1, imag2, labels = batch
        images1 = images1.squeeze(dim=1).squeeze(dim=1).float().to(device)
        images2 = images2.squeeze(dim=1).squeeze(dim=1).float().to(device)
        imag1 = imag1.squeeze(dim=1).squeeze(dim=1).float().to(device)
        imag2 = imag2.squeeze(dim=1).squeeze(dim=1).float().to(device) 
        labels = labels.long().to(device)
    
        model.eval()
        with torch.no_grad():
            preds = model(images1, images2, imag1, imag2)
    
        loss = criterion(preds, labels)
    
        validation_loss += loss.item()
        total_correct_val += get_num_correct(preds, labels)
    
        label_f1vl.extend(labels.cpu().numpy().tolist())
        pred_f1vl.extend(preds.argmax(dim=1).tolist())
    
        del images1
        del labels
    
    print(Acc_best)
    
    print(confusion_matrix(label_f1vl, pred_f1vl))
    with open(training_tsx, "a") as f:
        print('-' * 10, file=f) 
        print('Results', file=f)         
        print(Acc_best, file=f)    
        print(confusion_matrix(label_f1vl, pred_f1vl), file=f)        
        
    

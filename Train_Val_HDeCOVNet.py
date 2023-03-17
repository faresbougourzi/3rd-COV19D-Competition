#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 01:34:09 2023

@author: bougourzi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 01:06:27 2023

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



img_size = 224

from albumentations.pytorch import ToTensorV2
import torchvision.transforms as transforms
   
from torch.utils import data
import albumentations as A
train_transforms = A.ReplayCompose(
    [
        A.Resize(height=img_size, width=img_size),
        A.Rotate(limit=40, p=1.0),
        A.VerticalFlip(p=0.2),
        A.VerticalFlip(p=0.2), A.Blur(blur_limit=(3, 3), p=0.2),
        A.MultiplicativeNoise(multiplier=1.5, p=0.2), 
        A.MultiplicativeNoise(multiplier=0.5, p=0.2), 
        A.RandomBrightness(limit=0.2, always_apply=False, p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=0.2),
        A.RandomContrast (limit=0.2, always_apply=False, p=0.2),
        # A.RandomFog (fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.08, always_apply=False, p=0.2),
        # A.RandomGravel(gravel_roi=(0.1, 0.4, 0.9, 0.9), number_of_patches=2, always_apply=False, p=0.2),
        A.RandomGridShuffle (grid=(3, 3), always_apply=False, p=0.2),
        A.RandomToneCurve (scale=0.1, always_apply=False, p=0.2),
        # A.CLAHE (clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=0.5),                
        A.Normalize(
            mean=[0.0],
            std=[1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2()
    ], additional_targets={'image0': 'image'}
)
        
test_transforms =     A.ReplayCompose([
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

            for i in range(X2.shape[2]-1):
                image2_data = A.ReplayCompose.replay(data1['replay'], image=np.expand_dims(X2[:,:,i+1],2), image0=X3[:,:,i+1])
                img2 = image2_data['image']
                img3 = image2_data['image0']
                img = torch.concat([img2, img3], dim = 0)
                Xs1.append(img)

     
        Xs1 = torch.stack(Xs1).transpose(0, 1).transpose(1, 2)#.squeeze(dim = 1)
        
        Xs1 = F.interpolate(Xs1, [64, 224], mode ='bicubic').transpose(1, 2)
        # print(Xs1.shape)
       
        return Xs1, labels  

##################################################

torch.set_printoptions(linewidth=120)

tr_path = './NumpysSevSegZ2/Train5'
vl_path = './NumpysSevSegZ2/Val5'

tr_labels = np.load('./NumpysSevSegZ2/Train5/labels.npy')
vl_labels= np.load('./NumpysSevSegZ2/Val5/labels.npy')

tr_indxs= list(range(len(tr_labels)))
train_set = Covid_loader_pt2(
        list_IDs = tr_indxs, 
        path = tr_path, 
        transform=train_transforms
)

vl_indxs= list(range(len(vl_labels)))
test_set = Covid_loader_pt2(
        list_IDs = vl_indxs, 
        path = vl_path, 
        transform=test_transforms
)

#############################

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


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

####################################
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
        # self.max2 = nn.MaxPool3d(kernel_size=[1, 2, 2], stride=[1, 2, 2], padding=[0, 0, 0], dilation=1, ceil_mode=False)
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
        # x = self.max2(x)
        x = self.head(x)
        # # print(x.size[0])
        x = x.view(x.shape[0], x.shape[1])  
        return self.cls(x)
####################################

device = torch.device("cuda:0")
train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True)
validate_loader = torch.utils.data.DataLoader(test_set, batch_size=16)

Accracy_vl_mean = []
Accracy_ts_mean = []

MaF1_vl_mean = []
MaF1_ts_mean = [] 
 
WeF1_vl_mean = []
WeF1_ts_mean = [] 

dataset_idx = '3D'
modl_n = '3D_F5'
modl_nn = 'UNETR'

epochs = 100

runs = 2
for itr in range(runs):
    model_sp = "./Models" + dataset_idx +"/" + modl_n + "/Models"
    if not os.path.exists(model_sp):
        os.makedirs(model_sp)
    
    name_model_final = model_sp+ '/' + str(itr) + '_fi.pt'
    name_model_bestF1 =  model_sp+ '/' + str(itr) + '_bt.pt'
    # name_model_bestsw =  model_sp+ '/' + str(itr) + '_swa.pt'
    
    model_spR = "./Models" + dataset_idx +"/" + modl_n + "/Results"
    if not os.path.exists(model_spR):
        os.makedirs(model_spR)
        
    training_tsx = model_spR+ '/' + str(itr) + '.txt' 
    
    Acc_best = -2
    Acc_bestsw = -2
    epoch_count = []
    Accracy_tr = []
    Accracy_ts = []
    
    AccracyRA_tr = []
    AccracyRA_vl = []
    AccracyRA_sw = []
    
    MaF1_tr = []
    MaF1_vl = []
    MaF1_sw = []
    
    LR = 0.0001
    model = Resnet3DS().to(device)
    criterion = FocalLoss()    
    
    epoch_count = []
    Accracy_tr = []
    Accracy_ts = []
    
    Acc_best = -2

    ##################################  
    LR = 0.0001 
    for epoch in range(epochs):
        epoch_count.append(epoch)
        lr = LR
        if epoch > 30:
            lr = LR / 2
        if epoch > 50:
            lr = LR / 2 / 2
        if epoch > 70:
            lr = LR / 2 / 2 / 2
        if epoch > 80:
            lr = LR / 2 / 2 / 2 / 5
        if epoch > 90:
            lr = LR / 2 / 2 / 2 / 5 / 5   
    
        optimizer = optim.Adam(model.parameters(), lr=lr)
        train_loss = 0
        validation_loss = 0
        total_correct_tr = 0
        total_correct_val = 0
        total_correct_tr2 = 0
        total_correct_val2 = 0
       
        label_f1tr = []
        pred_f1tr = []
        
        for batch in tqdm(train_loader):
            images1, labels = batch
            images1 = images1.squeeze(dim=1).float().to(device)
            # images2 = images2.squeeze(dim = 1).squeeze(dim = 1).float().to(device)
            # images3 = images3.squeeze(dim = 1).squeeze(dim = 1).float().to(device)
            labels = labels.long().to(device)
            # labels2 = labels2.long().to(device)
            torch.set_grad_enabled(True)
            model.train()
            preds = model(images1)
            loss = criterion(preds, labels)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            train_loss += loss.item()
            total_correct_tr += get_num_correct(preds, labels)
            # total_correct_tr2 += get_num_correct(preds2, labels2)
    
            label_f1tr.extend(labels.cpu().numpy().tolist())
            pred_f1tr.extend(preds.argmax(dim=1).tolist())      
        
            del images1
            del labels
        
        label_f1vl = []
        pred_f1vl = []
       
    
        for batch in tqdm(validate_loader):
            images1, labels = batch
            images1 = images1.squeeze(dim=1).float().to(device)
            # images2 = images2.squeeze(dim = 1).squeeze(dim = 1).float().to(device)
            # images3 = images3.squeeze(dim = 1).squeeze(dim = 1).float().to(device)
            labels = labels.long().to(device)
            # labels2 = labels2.long().to(device)
    
            model.eval()
            with torch.no_grad():
                preds = model(images1)
    
            loss = criterion(preds, labels)
            # loss2 = criterion(preds2, labels2)
            # loss = loss1+loss2
    
            validation_loss += loss.item()
            total_correct_val += get_num_correct(preds, labels)
            
    
            label_f1vl.extend(labels.cpu().numpy().tolist())
            pred_f1vl.extend(preds.argmax(dim=1).tolist())
            
            del images1
            del labels
    
        print('Ep: ', epoch, 'AC_tr: ', total_correct_tr/len(train_set), 'AC_ts: ',
              total_correct_val/len(test_set),'AC_tr2: ', total_correct_tr2/len(train_set), 'AC_ts2: ',
              total_correct_val2/len(test_set), 'Loss_tr: ', train_loss/len(train_set))

        print('MaF1_tr: ', f1_score(label_f1tr, pred_f1tr, average='macro'), 'MaF1_vl: ',
              f1_score(label_f1vl, pred_f1vl, average='macro'))
        print('CM_tr: ')
        print(confusion_matrix(label_f1tr, pred_f1tr))    
        print('CM_vl: ')
        print(confusion_matrix(label_f1vl, pred_f1vl))
        
        
        ##################################        
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
          
            
        with open(training_tsx, "a") as f:
            print('MaF1_tr: ', f1_score(label_f1tr, pred_f1tr, average='macro'), 'MaF1_vl: ',
                  f1_score(label_f1vl, pred_f1vl, average='macro'), file=f) 
            
                
        Acc_best2 = f1_score(label_f1vl, pred_f1vl, average='macro')    
        if Acc_best2 > Acc_best:
            Acc_best = Acc_best2
            torch.save(model.state_dict(), name_model_bestF1)
            
        AccracyRA_tr.append(balanced_accuracy_score(label_f1tr, pred_f1tr))
        AccracyRA_vl.append(balanced_accuracy_score(label_f1vl, pred_f1vl))
               
        MaF1_tr.append(f1_score(label_f1tr, pred_f1tr , average='macro'))
        MaF1_vl.append(f1_score(label_f1vl, pred_f1vl , average='macro'))                    
    
    print(Acc_best)   
    model.load_state_dict(torch.load(name_model_bestF1))
    ###########################
    
    label_f1vl = []
    pred_f1vl = []   
    
    for batch in tqdm(validate_loader):
        images1, labels = batch
        images1 = images1.squeeze(dim=1).float().to(device)
        # images2 = images2.squeeze(dim = 1).squeeze(dim = 1).float().to(device)
        # images3 = images3.squeeze(dim = 1).squeeze(dim = 1).float().to(device)
        labels = labels.long().to(device)
        # labels2 = labels2.long().to(device)

        model.eval()
        with torch.no_grad():
            preds = model(images1)
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

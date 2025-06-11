import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, transforms
import torchvision.models.segmentation as models
import matplotlib.pyplot as plt
import time
from datetime import datetime
import os
import json
from PIL import Image
from tempfile import TemporaryDirectory
import tqdm
from dataLoader import MagClassDataset, get_weighted_data_loader
from train import train_model

current_time = datetime.now()
architecture = 'deeplabv3_resnet50'
label_type = 'arch-segmentation'
num_classes = 1
epoch_size_train = 40*124*2#7680
epoch_size_val = 40*32*4#1280
batch_size = 40#32
num_workers = 40#40
description = 'dice-loss'
trainging_mode = 'final-fc'#'all'#'head'#'final-fc'#'first-conv'
initial_weights = 'default'#r'/mnt/field/test/ml/cg/Classification Models/resplit_like_autoencoder - final-fc - 2025-05-06 134724'#'default'#
lr = 0.02#0.1
momentum = 0.9
step_size = 10
gamma = 0.5 # 0.6
num_epochs = 5


model_path = r'/mnt/magbucket/segmentation/Models'

train_dataset = MagClassDataset(r'/mnt/magbucket/segmentation/train.hdf5',label_type=label_type)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, pin_memory=True,num_workers=num_workers,shuffle=True)  

val_dataset = MagClassDataset(r'/mnt/magbucket/segmentation/valid.hdf5',augment=False,label_type=label_type)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, pin_memory=True,num_workers=num_workers,shuffle=True)

dataloaders = {}
dataloaders['train'] = train_loader
dataloaders['val'] = val_loader

log = {}
log['model_path'] = model_path
log['name'] = ' - '.join([description, trainging_mode,str(current_time)[:-7].replace(':','')])
log['architecture'] = architecture
log['label_type'] = label_type
log['num_classes'] = num_classes
log['hdf5_file'] = train_dataset.hdf5_file
log['trainging_mode'] = trainging_mode
log['initial_weights'] = initial_weights
log['crop_ranges'] = train_dataset.crop_ranges
log['crop_jitter'] = train_dataset.crop_jitter
log['max_white_noise'] = train_dataset.max_white_noise
log['epoch_size_train'] = epoch_size_train
log['epoch_size_val'] = epoch_size_val
log['batch_size'] = batch_size
log['num_workers'] = num_workers
log['lr'] = lr
log['momentum'] = momentum
log['step_size'] = step_size
log['gamma'] = gamma
log['num_epochs'] = num_epochs

os.mkdir(os.path.join(log['model_path'],log['name']))

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")



model = models.deeplabv3_resnet50(pretrained=True)

# Adjust the classifier head for your number of classes, e.g., binary or multi-class segmentation
num_classes = len(train_dataset.label_fields)
model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))


if initial_weights != 'default':
    model.load_state_dict(torch.load(os.path.join(initial_weights,'best_model_params.pt'), weights_only=True))
    model.eval()

model = model.to(device)

criterion = nn.BCEWithLogitsLoss()

class DiceLoss(nn.Module):
    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice

criterion = DiceLoss()

# Choose parameters to optimise

if trainging_mode=='head':
    print('head')
    optimizer_ft = optim.SGD(model.classifier.parameters(), lr=lr, momentum=momentum)
elif trainging_mode=='final-fc':
    print('final-fc')
    optimizer_ft = optim.SGD(model.classifier[4].parameters(), lr=lr, momentum=momentum)
elif trainging_mode=='all':
    print('all')
    optimizer_ft = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
elif trainging_mode=='first-conv':
    print('first-conv')
    optimizer_ft = optim.SGD(model.features[0].parameters(), lr=lr, momentum=momentum)   
    
# Decay LR by a factor of gamma every step_size epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step_size, gamma=gamma)

model, log = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                          device, dataloaders, log, num_epochs=num_epochs)

for k in log.keys():
    print(k,log[k],type(log[k]))

if initial_weights == 'default':    
    with open(os.path.join(log['model_path'],log['name'],'training_log.json'), 'w') as f:
        record = {}
        record[0] = log
        json.dump(record, f)
else:
    with open(os.path.join(log['initial_weights'],'training_log.json'), 'r') as f:
        record = json.load(f)
        record[len(record)] = log
    with open(os.path.join(log['model_path'],log['name'],'training_log.json'), 'w') as f:
        json.dump(record, f)
        
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
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
architecture = 'mobilenet_v3_large'
epoch_size_train = 40*124*2#7680
epoch_size_val = 40*32*4#1280
batch_size = 40#32
num_workers = 40#40
description = 'resplit_like_autoencoder'
trainging_mode = 'all'#'all'#'head'#'final-fc'#'first-conv'
initial_weights = r'/mnt/field/test/ml/cg/Classification Models/resplit_like_autoencoder - final-fc - 2025-05-06 134724'#'default'#
lr = 0.02#0.1
momentum = 0.9
step_size = 10
gamma = 0.5 # 0.6
num_epochs = 60


model_path = r'/mnt/field/test/ml/cg/Classification Models'

train_dataset = MagClassDataset(r'/mnt/field/test/ml/cg/Classification Datasets/resplit_like_autoencoder/train.hdf5',label='arch-segmentation')
train_loader = get_weighted_data_loader(train_dataset,epoch_size_train,batch_size,num_workers=num_workers)

val_dataset = MagClassDataset(r'/mnt/field/test/ml/cg/Classification Datasets/resplit_like_autoencoder/valid.hdf5',augment=False,label='arch-segmentation')
val_loader = get_weighted_data_loader(val_dataset,epoch_size_val,batch_size,num_workers=num_workers)

dataloaders = {}
dataloaders['train'] = train_loader
dataloaders['val'] = val_loader

log = {}
log['model_path'] = model_path
log['name'] = ' - '.join([description, trainging_mode,str(current_time)[:-7].replace(':','')])
log['architecture'] = architecture
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


mobilenetv3 = torchvision.models.mobilenet_v3_large(weights='DEFAULT')
    
num_ftrs = mobilenetv3.classifier[3].in_features


mobilenetv3.classifier[3] = nn.Sequential(nn.Linear(num_ftrs, 1),
                                          nn.Sigmoid())

if initial_weights != 'default':
    mobilenetv3.load_state_dict(torch.load(os.path.join(initial_weights,'best_model_params.pt'), weights_only=True))
    mobilenetv3.eval()

mobilenetv3 = mobilenetv3.to(device)

criterion = nn.BCEWithLogitsLoss()

# Choose parameters to optimise

if trainging_mode=='head':
    print('head')
    optimizer_ft = optim.SGD(mobilenetv3.classifier.parameters(), lr=lr, momentum=momentum)
elif trainging_mode=='final-fc':
    print('final-fc')
    optimizer_ft = optim.SGD(mobilenetv3.classifier[3].parameters(), lr=lr, momentum=momentum)
elif trainging_mode=='all':
    print('all')
    optimizer_ft = optim.SGD(mobilenetv3.parameters(), lr=lr, momentum=momentum)
elif trainging_mode=='first-conv':
    print('first-conv')
    optimizer_ft = optim.SGD(mobilenetv3.features[0].parameters(), lr=lr, momentum=momentum)   
    
# Decay LR by a factor of gamma every step_size epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=step_size, gamma=gamma)

mobilenetv3, log = train_model(mobilenetv3, criterion, optimizer_ft, exp_lr_scheduler,
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
        
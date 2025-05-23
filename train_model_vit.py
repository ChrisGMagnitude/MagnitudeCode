import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from vit_pytorch import ViT
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
architecture = 'ViT'
epoch_size_train = 64*128*1#7680
epoch_size_val = 64*32*1#1280
batch_size = 64#32
num_workers = 32#40
description = 'ViT AE-Split some-DINO-pretraining-extraFCL'
trainging_mode = 'final-fc'#'all'#'final-fc'
initial_weights = r'/mnt/field/test/ml/cg/DINO Models/Run 3 DINOViT - mid 5e-4 lr - full epoch - 2025-05-06 141728 - epoch31'#'default'#
lr = 1e-1#0.1
momentum = 0.9
step_size = 10
gamma = 0.5 # 0.6
num_epochs = 5
image_size = 416


model_path = r'/mnt/field/test/ml/cg/Classification Models'

train_dataset = MagClassDataset(r'/mnt/field/test/ml/cg/Classification Datasets/resplit_like_autoencoder/train.hdf5',ViT_im_size=True)
train_loader = get_weighted_data_loader(train_dataset,epoch_size_train,batch_size,num_workers=num_workers)

val_dataset = MagClassDataset(r'/mnt/field/test/ml/cg/Classification Datasets/resplit_like_autoencoder/valid.hdf5',augment=False,ViT_im_size=True)
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
log['image_size'] = image_size
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


model = ViT(
    image_size = image_size,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 8,
    mlp_dim = 2048
)
    



if initial_weights != 'default':
    model.load_state_dict(torch.load(os.path.join(initial_weights,'ViT-Params.pt'), weights_only=True,map_location=torch.device(device)))
    model.eval()

model = nn.Sequential(model,
                      nn.Linear(1000, 1),
                      nn.Sigmoid())

model = model.to(device)

criterion = nn.BCEWithLogitsLoss()

# Choose parameters to optimise

if trainging_mode=='final-fc':
    print('final-fc')
    optimizer_ft = optim.SGD(model[1].parameters(), lr=lr, momentum=momentum)
elif trainging_mode=='all':
    print('all')
    optimizer_ft = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    
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
        
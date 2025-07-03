import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, transforms
import torchvision.models.segmentation as models
from torchvision.ops import sigmoid_focal_loss
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
architecture = 'lraspp_mobilenet'
label_type = 'merged-segmentation'
num_classes = 4
epoch_size_train = 20*124*10#7680
epoch_size_val = 20*32*5#1280
batch_size = 40#32
num_workers = 8#40
description = 'lraspp-noModern'
trainging_mode = 'all'#'all'#'head'#'final-fc'#'first-conv'
initial_weights = r'/mnt/magbucket/segmentation/Models/lraspp-noModern - head - 2025-07-03 105410'#'default'#
initial_weights_file = 'last_model_params.pt'#'default'#
lr = 0.001#0.0005#0.02#0.1
momentum = 0.9
step_size = 10
gamma = 0.6 # 0.6
weight_decay=0.003
num_epochs = 50
interp_id_lookup = {}
interp_id_lookup["combinedMask"] = ['Agricultural (Strong)Mask',
                                    'Agricultural (Weak)Mask',
                                    'Archaeology Possible (Strong)Mask',
                                    'Archaeology Possible (Weak)Mask',
                                    'Archaeology Probable (Strong)Mask',
                                    'Archaeology Probable (Weak)Mask',
                                    'Undetermined (Strong)Mask',
                                    'Undetermined (Weak)Mask']
interp_id_lookup["naturalMask"] = ['Natural (Strong)Mask',
                                   'Natural (Weak)Mask']
#interp_id_lookup["modernMask"] = ['IndustrialModernMask']



model_path = r'/mnt/magbucket/segmentation/Models'

val_dataset = MagClassDataset(r'/mnt/magbucket/segmentation/valid.hdf5',augment=False,label_type=label_type,interp_id_lookup=interp_id_lookup)

train_dataset = MagClassDataset(r'/mnt/magbucket/segmentation/train.hdf5',augment=True,label_type=label_type,
                               crop_jitter=[0.2,0.4,1.6], max_white_noise=0.001,interp_id_lookup=interp_id_lookup)


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, pin_memory=True,num_workers=num_workers,shuffle=True)  
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, pin_memory=True,num_workers=num_workers,shuffle=True)

#train_loader = get_weighted_data_loader(train_dataset,epoch_size_train,batch_size,num_workers=num_workers)
#val_loader = get_weighted_data_loader(val_dataset,epoch_size_val,batch_size,num_workers=num_workers)


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
log['interp_id_lookup'] = interp_id_lookup
log['trainging_mode'] = trainging_mode
log['initial_weights'] = initial_weights
log['initial_weights'] = initial_weights
log['crop_ranges'] = train_dataset.crop_ranges
log['crop_jitter'] = train_dataset.crop_jitter
log['max_white_noise'] = train_dataset.max_white_noise
log['translate'] = train_dataset.translate
log['epoch_size_train'] = epoch_size_train
log['epoch_size_val'] = epoch_size_val
log['batch_size'] = batch_size
log['num_workers'] = num_workers
log['lr'] = lr
log['momentum'] = momentum
log['weight_decay'] = weight_decay
log['step_size'] = step_size
log['gamma'] = gamma
log['num_epochs'] = num_epochs


os.mkdir(os.path.join(log['model_path'],log['name']))

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

num_classes = len(train_dataset.label_fields)

#model = models.deeplabv3_resnet50(pretrained=True)
#model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))

model = models.lraspp_mobilenet_v3_large(pretrained=True)
model.classifier.low_classifier = torch.nn.Conv2d(40, num_classes, kernel_size=(1, 1), stride=(1, 1))
model.classifier.high_classifier = torch.nn.Conv2d(128, num_classes, kernel_size=(1, 1), stride=(1, 1))


if initial_weights != 'default':
    model.load_state_dict(torch.load(os.path.join(initial_weights,initial_weights_file), weights_only=True))
    model.eval()

model = model.to(device)

#criterion = nn.BCEWithLogitsLoss()

class DiceLoss(nn.Module):
    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice

criterion = DiceLoss()

#class FocalLoss(nn.Module):
#    def forward(self, inputs, targets):
#        loss = sigmoid_focal_loss(inputs, targets,reduction='mean')
#        return 1 - loss

#criterion = FocalLoss()

#criterion = sigmoid_focal_loss()

# Choose parameters to optimise

if trainging_mode=='head':
    print('head')
    optimizer_ft = optim.SGD(model.classifier.parameters(), lr=lr, momentum=momentum,weight_decay=weight_decay)
elif trainging_mode=='final-fc':
    print('final-fc')
    optimizer_ft = optim.SGD([model.classifier.low_classifier.parameters(),model.classifier.high_classifier.parameters()], lr=lr, momentum=momentum,weight_decay=weight_decay)
elif trainging_mode=='all':
    print('all')
    optimizer_ft = optim.SGD(model.parameters(), lr=lr, momentum=momentum,weight_decay=weight_decay)
elif trainging_mode=='first-conv':
    print('first-conv')
    optimizer_ft = optim.adam(model.features[0].parameters(), lr=lr, momentum=momentum,weight_decay=weight_decay)
    
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
        
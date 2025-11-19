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
from pynvml import *
#Working2gp
 
current_time = datetime.now()
architecture = 'FCN_resnet50'
label_type = 'merged-segmentation'
num_classes = 4
epoch_size_train = 20*124*10#7680
epoch_size_val = 20*32*5#1280
batch_size = 56#32
num_workers = 8#40
description = 'GAN-Wasserstien-gp-Working4-noBN-hypreparamchange-FDlowLR'
trainging_mode = 'all'#'all'#'generator'#'discriminator'
initial_weights = r'/mnt/magbucket/segmentation/Models/GAN-Wasserstien-gp-Working4-noBN-hypreparamchange - all - 2025-11-18 173741'#'default'#
initial_weights_file = '54_epoch_model_params.pt'#'default'#
initial_weights_d = r'/mnt/magbucket/segmentation/Models/GAN-Wasserstien-gp-Working4-noBN-hypreparamchange - all - 2025-11-18 173741'#'default'#
initial_weights_file_d = '54_epoch_netD_params.pt'
lr_g = 0.000002
lr_d = 0.00002
momentum = 0.9
step_size = 10
gamma = 0.6 # 0.6
weight_decay=0.003
num_epochs = 56
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
interp_id_lookup["modernMask"] = ['IndustrialModernMask']



model_path = r'/mnt/magbucket/segmentation/Models'

val_dataset = MagClassDataset(r'/mnt/magbucket/segmentation/valid.hdf5',augment=True,label_type=label_type,
                              crop_jitter=[0.2,0.4,1.6], max_white_noise=0.001,interp_id_lookup=interp_id_lookup)

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
log['initial_weights_d'] = initial_weights_d
log['crop_ranges'] = train_dataset.crop_ranges
log['crop_jitter'] = train_dataset.crop_jitter
log['max_white_noise'] = train_dataset.max_white_noise
log['translate'] = train_dataset.translate
log['epoch_size_train'] = epoch_size_train
log['epoch_size_val'] = epoch_size_val
log['batch_size'] = batch_size
log['num_workers'] = num_workers
log['lr_d'] = lr_d
log['lr_g'] = lr_g
log['momentum'] = momentum
log['weight_decay'] = weight_decay
log['step_size'] = step_size
log['gamma'] = gamma
log['num_epochs'] = num_epochs


os.mkdir(os.path.join(log['model_path'],log['name']))

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

num_classes = 3#len(train_dataset.label_fields)

model = models.deeplabv3_resnet50(pretrained=True)
model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))



if initial_weights != 'default':
    model.load_state_dict(torch.load(os.path.join(initial_weights,initial_weights_file), weights_only=True))
    model.eval()

model = model.to(device)



#GAN Discriminator

nc = 6
ndf = 64

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=True),
            #nn.BatchNorm2d(ndf * 2, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=True),
            #nn.BatchNorm2d(ndf * 4, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=True),
            #nn.BatchNorm2d(ndf * 8, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=True),
            nn.Flatten(),
            nn.Linear(23*23, 1),
            #nn.Sigmoid()
        )
    def forward(self, input):
        return self.main(input)

netD = Discriminator()

if initial_weights_d != 'default':
    netD.load_state_dict(torch.load(os.path.join(initial_weights_d,initial_weights_file_d), weights_only=True))
    netD.eval()

netD = netD.to(device)

criterion = nn.BCELoss()

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

beta1 = 0.5

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr_d, betas=(beta1, 0.999))
#params = list(model.classifier.parameters()) + list(model.aux_classifier.parameters())
optimizerG = optim.Adam([{'params':model.backbone.parameters(), 'lr':lr_g/100},
                         {'params':model.classifier.parameters()},
                         {'params':model.aux_classifier.parameters()}
                         ], lr=lr_g, betas=(beta1, 0.999))


model, log = train_model(model, netD, optimizerG, optimizerD, criterion,
                        device, dataloaders, log, num_epochs=num_epochs)
                         #optimizer_ft, exp_lr_scheduler,
                         #device, dataloaders, log, num_epochs=num_epochs)

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
        
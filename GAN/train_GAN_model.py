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
architecture = 'FCN_resnet50'
label_type = 'merged-segmentation'
num_classes = 4
epoch_size_train = 20*124*10#7680
epoch_size_val = 20*32*5#1280
batch_size = 40#32
num_workers = 8#40
description = 'FCN-noModern'
trainging_mode = 'all'#'all'#'head'#'final-fc'#'first-conv'
initial_weights = r'/mnt/magbucket/segmentation/Models/FCN-noModern - head - 2025-07-03 162852'#'default'#
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

#model = models.lraspp_mobilenet_v3_large(pretrained=True)
#model.classifier.low_classifier = torch.nn.Conv2d(40, num_classes, kernel_size=(1, 1), stride=(1, 1))
#model.classifier.high_classifier = torch.nn.Conv2d(128, num_classes, kernel_size=(1, 1), stride=(1, 1))

model = models.fcn_resnet50(pretrained=True)
model.classifier[4] = torch.nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
model.aux_classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))


if initial_weights != 'default':
    model.load_state_dict(torch.load(os.path.join(initial_weights,initial_weights_file), weights_only=True))
    model.eval()

model = model.to(device)

#GAN Discriminator

nc = 3
ndf = 64

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

netD = Discriminator().to(device)

criterion = nn.BCELoss()

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

beta1 = 0.5

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(model.parameters(), lr=lr, betas=(beta1, 0.999))

# Choose parameters to optimise

#if trainging_mode=='head':
#    print('head')
#    params = list(model.classifier.parameters()) + list(model.aux_classifier.parameters())
#    optimizer_ft = optim.SGD(params, lr=lr, momentum=momentum,weight_decay=weight_decay)
#elif trainging_mode=='final-fc':
#    print('final-fc')
#    optimizer_ft = optim.SGD([model.classifier.low_classifier.parameters(),model.classifier.high_classifier.parameters()], lr=lr, momentum=momentum,weight_decay=weight_decay)
#elif trainging_mode=='all':
#    print('all')
#    optimizer_ft = optim.SGD(model.parameters(), lr=lr, momentum=momentum,weight_decay=weight_decay)
#elif trainging_mode=='first-conv':
#    print('first-conv')
#    optimizer_ft = optim.adam(model.features[0].parameters(), lr=lr, momentum=momentum,weight_decay=weight_decay)
    

    
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
        
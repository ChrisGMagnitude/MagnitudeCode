import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import pandas as pd
import torchvision
from vit_pytorch import ViT
#from pytorch_pretrained_vit import ViT
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
import random

image_size = 416#416#392

model_path = r'/mnt/field/test/ml/cg/Classification Models'

train_dataset = MagClassDataset(r'/mnt/field/test/ml/cg/Classification Datasets/resplit_like_autoencoder/train.hdf5',ViT_im_size=image_size)

val_dataset = MagClassDataset(r'/mnt/field/test/ml/cg/Classification Datasets/resplit_like_autoencoder/valid.hdf5',augment=False,ViT_im_size=image_size)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")


'''model = ViT('B_16_imagenet1k', pretrained=True)'''


'''model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_lc')'''


model = ViT(
    image_size = image_size,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 8,
    mlp_dim = 2048
)
initial_weights = r'/mnt/field/test/ml/cg/DINO Models/Run 3 DINOViT - mid 5e-4 lr - full epoch - 2025-05-06 141728 - epoch31'

if initial_weights != 'default':
    model.load_state_dict(torch.load(os.path.join(initial_weights,'ViT-Params.pt'), weights_only=True,map_location=torch.device(device)))
    model.eval()

model.eval()

embeddings = []
labels = []

arch_index = list(np.where(train_dataset.labels==1)[0])
other_index = list(np.where(train_dataset.labels!=1)[0])

arch_sample = random.sample(arch_index,5000)
other_sample = random.sample(other_index,5000)

sample = arch_sample+other_sample
sample.sort()

print(len(arch_index))
print(len(other_index))
for i in sample:

    print(i)
    image, label = train_dataset[i]

    image = image[None,:,:,:]

    #print(image.shape)

    output = model(image)


    
    output = output.detach().numpy()

    embeddings.append(output)

    labels.append(label)

embeddings = np.concatenate(embeddings)

pd.DataFrame(np.array(labels)).to_csv(os.path.join(model_path,'ExampleEmbeddings','mag-trained-dino_label.csv'))
pd.DataFrame(embeddings).to_csv(os.path.join(model_path,'ExampleEmbeddings','mag-trained-dino.csv'))


embeddings = []
labels = []

arch_index = list(np.where(val_dataset.labels==1)[0])
other_index = list(np.where(val_dataset.labels!=1)[0])


arch_sample = random.sample(arch_index,1000)
other_sample = random.sample(other_index,1000)

sample = arch_sample+other_sample
sample.sort()

print(len(arch_index))
print(len(other_index))

for i in sample:
    print(i)
    image, label = val_dataset[i]

    image = image[None,:,:,:]

    #print(image.shape)

    output = model(image)


    
    output = output.detach().numpy()

    embeddings.append(output)

    labels.append(label)

embeddings = np.concatenate(embeddings)

pd.DataFrame(np.array(labels)).to_csv(os.path.join(model_path,'ExampleEmbeddings','mag-trained-dino_label_val.csv'))
pd.DataFrame(embeddings).to_csv(os.path.join(model_path,'ExampleEmbeddings','mag-trained-dino_val.csv'))

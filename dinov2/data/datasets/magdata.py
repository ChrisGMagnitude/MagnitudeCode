# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import csv
from enum import Enum
import logging
import os
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import h5py
import torch


from torchvision.transforms import v2,Compose

from .extended import ExtendedVisionDataset


logger = logging.getLogger("dinov2")
_Target = int



class MagData(ExtendedVisionDataset):
    def __init__(self,
                *,
                split,
                root: str,
                transforms: Optional[Callable] = None,
                transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = None,
                augment=True, crop_ranges=[[-1,2],[-3,5],[-10,20]], crop_jitter=[0.25,0.5,2], max_white_noise=0.05,ViT_im_size = False):
        """
        Arguments:
            hdf5_file (string): Path to the HDF5 file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__(root, transforms, transform, target_transform)

        hdf5_file = os.path.join(root,split.lower()+'.hdf5')

        self.hdf5_file = hdf5_file
        self.fh = h5py.File(self.hdf5_file, "r")
        self.crop_ranges = crop_ranges
        self.crop_jitter = crop_jitter
        self.max_white_noise = max_white_noise
        self.augment = augment
        self.ViT_im_size = ViT_im_size
           
        
        print('MagData')
        print('root',root)
        print('split',split)

        
        
    def __len__(self):
        return len(self.fh["images"])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.fh["images"][idx]
        image[np.isnan(image)] = 0
        image[image==-9999] = 0
        image = self.clip_and_normalise_data(image)
        image = self.apply_transforms(image)

        #Add noise
        image = image + (0.001**0.5)*torch.randn(image.shape)

        #sample = [image]
        target = 0
        #print(image)
        #print(type(image))
        #print(image.shape)
        #print(sum(torch.isnan(image)))
        if torch.isnan(image).any():
            #print(image)
            print('pretransform')
            stop
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        #print(image)
        for im in image['local_crops']:
            if torch.isnan(im).any():
                #print(image['local_crops']) 
                print('local_crops posttransform')
                stop
        for im in image['global_crops']:
            if torch.isnan(im).any():
                #print(image['global_crops']) 
                print('global_crops posttransform')
                stop

        return image, target
        
    def clip_and_normalise_data(self,image):
        
        output = []
        
        
        for i,r in enumerate(self.crop_ranges):
            
            # Clip data
            if self.augment:
                jitter = np.random.uniform(low=-self.crop_jitter[i],
                                       high=self.crop_jitter[i],
                                       size=2)
                                       
                low_clip = r[0]+jitter[0]    
                high_clip = r[1]+jitter[1] 

            else:
                low_clip = r[0]   
                high_clip = r[1] 
                
            im = np.clip(image, low_clip, high_clip)
            
            # Normalise data
            im = (im - low_clip)/(high_clip - low_clip)
            
            
            
            output.append(im)
            
        
        return np.array(output)
        
    def apply_transforms(self,image):
        
        image = torch.from_numpy(image)
        
        
        if self.ViT_im_size:
            crop_size = self.ViT_im_size
        else:
            crop_size = int(round(np.sqrt(image.shape[1]**2/2)))
        
        rand = np.random.uniform(low=0,high=self.max_white_noise)
        
        if self.augment:
            transformer = Compose([
                                            v2.CenterCrop((crop_size,crop_size)),
                                            v2.RandomRotation(degrees=(0, 360)),
                                            v2.RandomHorizontalFlip(),
                                            #v2.GaussianNoise(sigma=rand)
                                            ])
        else:
            transformer = Compose([
                                            v2.CenterCrop((crop_size,crop_size)),
                                            ])
            
            
        image = transformer(image)
        
        return image.type(torch.float)
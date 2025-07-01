import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.transforms import v2

import h5py

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class MagClassDataset(Dataset):

    def __init__(self, hdf5_file, 
                 augment=True, crop_ranges=[[-1,2],[-3,5],[-10,20]], crop_jitter=[0.25,0.5,2], max_white_noise=0.05,label_type='binary',ViT_im_size = False,translate=0.1):
        """
        Arguments:
            csv_file (string): Path to the HDF5 file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.hdf5_file = hdf5_file
        self.fh = h5py.File(self.hdf5_file, "a")
        self.label_type = label_type

        if label_type=='binary':
            label = self.fh["labels"]
            label = [l.decode() for l in label]
            binary_label = np.zeros(len(label))
            binary_label[np.array(label)=='archae']=1
            self.labels = binary_label.astype(float)
            self.classes = np.unique(self.labels)
        elif label_type=='classification':
            self.labels = self.fh["labels"]
            self.classes = np.unique(self.labels)
        elif label_type=='arch-segmentation':
            self.label_fields = ["archMask"]
        elif label_type=='all-segmentation':
            self.label_fields = ["archMask","agriMask","naturalMask","modernMask"]
            self.image_class = self.fh["class"]
        elif label_type=='merged-segmentation':
            self.label_fields = ["combinedMask","naturalMask"]
            print(np.array(self.fh.keys()))
            
            available_masks = []
            for key in self.fh.keys():
                if key.endswith('Mask'):
                    print(key)
                    available_masks.append(key)

            self.raw_label_fields = available_masks

            if not 'image_class' in np.array(self.fh.keys()):
                print("calculating image class")
                image_class = []
                for i in range(len(self.fh[available_masks[0]])):
                    if i%1000==0:
                        print(i)
                    mask_sums = []
                    for m in available_masks:
                        mask_sums.append(sum(sum(self.fh[m][i])))
                    image_class.append(available_masks[np.argmin(mask_sums)])

                print(len(image_class)) 
                print(np.unique(image_class))     

                self.fh.create_dataset('image_class', data=image_class, compression="lzf", chunks=True, maxshape=(None,), dtype=h5py.string_dtype()) 
                
            
        self.dataset_size = len(self.fh["images"])
        
        self.crop_ranges = crop_ranges
        self.crop_jitter = crop_jitter
        self.max_white_noise = max_white_noise
        self.translate = translate
        self.augment = augment
        self.ViT_im_size = ViT_im_size
           
        
        
        
        
    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        
        image = self.fh["images"][idx]
        image[np.isnan(image)] = 0
        image = self.clip_and_normalise_data(image)
    
        if 'segmentation' in self.label_type:
            image, label = self.apply_transforms(image,idx)
        else:
            image = self.apply_transforms(image,idx)
            label = self.labels[idx]
        
        sample = [image, label]
        

        return sample
        
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
        
    def apply_transforms(self,image,idx):
        
        image = torch.from_numpy(image)
        
        if self.ViT_im_size:
            crop_size = self.ViT_im_size
        else:
            crop_size = int(round(np.sqrt(image.shape[1]**2/2)))
        
        

        if self.augment:
            rand = np.random.uniform(low=0,high=self.max_white_noise)
            image = image + (rand**0.5)*torch.randn(image.shape)
            image = torch.clamp(image, min=0, max=1)

            transformer = transforms.Compose([
                                            transforms.v2.RandomAffine(degrees=(0, 360),
                                                                       translate=(self.translate,self.translate)),
                                            transforms.v2.RandomHorizontalFlip(),
                                            transforms.v2.CenterCrop((crop_size,crop_size))
                                            ])
        else:
            transformer = transforms.Compose([
                                            transforms.v2.CenterCrop((crop_size,crop_size)),
                                            ])
            
        

        if 'segmentation' in self.label_type:

            label = [self.fh[l][idx] for l in self.label_fields]
            label = np.stack(label)
            label = torch.from_numpy(label)

            t = transformer(torch.cat([image,label],dim=0))

            image = t[:3,:,:]
            label = t[3:,:,:]

            return(image.type(torch.float),label)
        else:
            image = transformer(image)
            return image.type(torch.float)
        
        
def make_weights_for_balanced_classes(classes):

    nclasses = len(np.unique(classes))
    # Turn classes list of strings in to integers
    classes_int = np.array([0] * len(classes) )
    for i,c in enumerate(np.unique(classes)):
        classes_int[np.array(classes)==c] = i
    # Count number intances of each class
    count = [0] * nclasses       
    for item in classes_int:          
        count[item] += 1
    # Calculate weight of each class type                                          
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])
    # Create vector with weight for each sample                         
    weight = [0] * len(classes)                                              
    for idx, val in enumerate(classes_int):                     
        weight[idx] = weight_per_class[val]                                  
    return weight



                                                        
def get_weighted_data_loader(dataset,epoch_size,batch_size,num_workers=40):                                                        
    # For unbalanced dataset we create a weighted sampler   
    weights = make_weights_for_balanced_classes(dataset.image_class)     
                                                            
    weights = torch.DoubleTensor(weights)
    
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, epoch_size)#len(weights))                     
                                                                                    
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,                               
                                             sampler = sampler, pin_memory=True,num_workers=num_workers)      
    return data_loader
                                             

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
                 augment=True, crop_ranges=[[-1,2],[-3,5],[-10,20]], crop_jitter=[0.25,0.5,2], max_white_noise=0.05,label_type='binary',ViT_im_size = False):
        """
        Arguments:
            csv_file (string): Path to the HDF5 file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.hdf5_file = hdf5_file
        self.fh = h5py.File(self.hdf5_file, "r")
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
            
        self.dataset_size = len(self.fh["images"])
        
        self.crop_ranges = crop_ranges
        self.crop_jitter = crop_jitter
        self.max_white_noise = max_white_noise
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
        
        rand = np.random.uniform(low=0,high=self.max_white_noise)
        image = image + (rand**0.5)*torch.randn(image.shape)

        if self.augment:
            transformer = transforms.Compose([
                                            transforms.v2.CenterCrop((crop_size,crop_size)),
                                            transforms.v2.RandomRotation(degrees=(0, 360)),
                                            transforms.v2.RandomHorizontalFlip()
                                            ])
        else:
            transformer = transforms.Compose([
                                            transforms.v2.CenterCrop((crop_size,crop_size)),
                                            ])
            
        image = transformer(image)

        if 'segmentation' in self.label_type:

            label = [self.fh[l][idx] for l in self.label_fields]
            label = np.stack(label)
            label = torch.from_numpy(label)
            label = transformer(label)
            return(image.type(torch.float),label)
        else:
            return image.type(torch.float)
        
        
def make_weights_for_balanced_classes(classes, nclasses):  
    classes = classes.astype(int)
    count = [0] * nclasses       
    for item in classes:          
        count[item] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(classes)                                              
    for idx, val in enumerate(classes):                     
        weight[idx] = weight_per_class[val]                                  
    return weight



                                                        
def get_weighted_data_loader(dataset,epoch_size,batch_size,num_workers=40):                                                        
    # For unbalanced dataset we create a weighted sampler   
    weights = make_weights_for_balanced_classes(dataset.labels, len(dataset.classes))     
                                                            
    weights = torch.DoubleTensor(weights)
    
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, epoch_size)#len(weights))                     
                                                                                    
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,                               
                                             sampler = sampler, pin_memory=True,num_workers=num_workers)      
    return data_loader
                                             
#train_dataset = MagClassDataset(r'F:\test\ml\cg\FirstDataset\train.hdf5')
#train_loader = get_weighted_data_loader(train_dataset) 

#for i,x in train_loader:
#    print(i,x)


#for i in range(20):
#    d = dataset[20100]
#    image = d['image']
#    label = d['label'].decode()
#    project = d['project'].decode()
#    
#    plt.imshow(image[2,:,:],cmap='gray')
#    plt.savefig('20100_'+str(i)+'_'+label+'_'+project.strip()+r'_rotcrop.png')
#    plt.clf()

#print(dataset.__len__())
#print(np.arange(24)*1000)
#
#
#for i in np.arange(24)*1000:
#    d = dataset[i]
#    image = d['image']
#    label = d['label'].decode()
#    project = d['project'].decode()
#    
#    a = np.arange(360)
#    x = np.sin(np.radians(a))*image.shape[1]/2 + image.shape[1]/2
#    y = np.cos(np.radians(a))*image.shape[1]/2 + image.shape[1]/2
#    
#    plt.plot(x,y,'r')
#    plt.plot([83,483,483,83,83],[483,483,83,83,483],'g')
#    #stop
#    
#    plt.imshow(image[2,:,:],cmap='gray')
#    plt.savefig(str(i)+'_'+label+'_'+project.strip()+r'_raw.png')
#    plt.clf()
#    
#    image = torch.from_numpy(image)
#    
#    crop_size = int(round(np.sqrt(image.shape[1]**2/2)))
#    cropper = transforms.v2.CenterCrop((crop_size,crop_size))
#    
#    rotator = transforms.v2.RandomRotation(degrees=(0, 360))
#    
#    image_rot = rotator(image)
#    
#    plt.plot(x,y,'r')
#    plt.plot([83,483,483,83,83],[483,483,83,83,483],'g')
#    
#    plt.imshow(image_rot[2,:,:],cmap='gray')
#    plt.savefig(str(i)+'_'+label+'_'+project.strip()+r'_rot.png')
#    plt.clf()
#    
#    image_rotcrop = cropper(image_rot)
#    
#    plt.imshow(image_rotcrop[2,:,:],cmap='gray')
#    plt.savefig(str(i)+'_'+label+'_'+project.strip()+r'_rotcrop.png')
#    plt.clf()
#    
#    #stop
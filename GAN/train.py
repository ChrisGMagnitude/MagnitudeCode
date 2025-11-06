import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torchmetrics
from torchmetrics.segmentation import MeanIoU
import matplotlib.pyplot as plt
import time
from datetime import datetime
import os
import json
from PIL import Image
from tempfile import TemporaryDirectory
import tqdm
from pynvml import *

def train_model(model, netD, optimizerG, optimizerD, criterion,
                device, dataloaders, log, num_epochs=25,
                real_label = 1, fake_label = 0, label_smoothing = 0.1):
    

    # Create a temporary directory to save training checkpoints

    train_loss_g_epoch = []
    val_loss_g_epoch = []
    train_loss_d_epoch = []
    val_loss_d_epoch = []
    train_real_accuracy_d_epoch = []
    train_fake_accuracy_d_epoch = []
    train_fake_accuracy_g_epoch = []
    val_real_accuracy_d_epoch = []
    val_fake_accuracy_d_epoch = []
    val_fake_accuracy_g_epoch = []
    
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r-a
        print('Start of training epoch')
        print(f'Reserved {r/1000000} / {t/1000000}')
        print(f'Allocated {a/1000000} / {t/1000000}')
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            train_loss_g = []
            val_loss_g = []
            train_loss_d = []
            val_loss_d = []
            train_real_accuracy_d = []
            train_fake_accuracy_d = []
            train_fake_accuracy_g = []
            val_real_accuracy_d = []
            val_fake_accuracy_d = []
            val_fake_accuracy_g = []
            
            # Iterate over data.
            for inputs, labels in tqdm.tqdm(dataloaders[phase],ascii=True):
                
                # Format data and labels and move to GPU
                labels = labels.type(torch.float)
                inputs = inputs.to(device)
                labels = labels.to(device)
                combined = torch.cat((inputs, labels), dim=1).to(device)
                
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                # Segmentation model in eval mode when training netD
                model.eval()
                
                # netD to training mode only for training loop, otherwise eval mode
                if phase == 'train':
                    if log['trainging_mode']=='all' or log['trainging_mode']=='discriminator':
                        netD.train()  # Set model to training mode
                    else:
                        netD.eval()   # Set model to evaluate mode
                else:
                    netD.eval()   # Set model to evaluate mode
                #netD.train()
                
                ## Train with all-real batch
                netD.zero_grad()
                
                # Format batch
                b_size = combined.size(0)
                label = torch.full((b_size,), real_label - label_smoothing, dtype=torch.float, device=device)
                # Forward pass real batch through D
                output_r = netD(combined).view(-1)
                # Calculate loss on all-real batch
                errD_real = criterion(output_r, label)
                output_r = output_r.detach()
                # Calculate gradients for D in backward pass
                if phase == 'train':
                    if log['trainging_mode']=='all' or log['trainging_mode']=='discriminator':
                        errD_real.backward()
                #errD_real.backward()
                
                ## Create all fake batch       
                fake_segmentartion = model(inputs)['out'].detach()
                seg_labels_out = fake_segmentartion>0
                fake_combined = torch.cat((inputs, seg_labels_out), dim=1)
                
                # Create fake label
                label.fill_(fake_label)
                # Classify all fake batch with D
                output_f = netD(fake_combined.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = criterion(output_f, label)
                output_f = output_f.detach()
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                if phase == 'train':
                    if log['trainging_mode']=='all' or log['trainging_mode']=='discriminator':
                       errD_fake.backward()
                #errD_fake.backward()
                        
                # Update D
                #if phase == 'train':
                #    if log['trainging_mode']=='all' or log['trainging_mode']=='discriminator':
                optimizerD.step()
                    
                # Compute error of D as sum over the fake and the real batches
                errD_real = errD_real.detach()
                errD_fake = errD_fake.detach()
                errD = errD_real + errD_fake
                # Update D
                if phase == 'train':
                    train_loss_d.append(errD.cpu())
                    train_real_accuracy_d.append(sum(output_r.cpu().numpy()>0.5)/len(output_r.cpu().numpy()))
                    train_fake_accuracy_d.append(sum(output_f.cpu().numpy()<0.5)/len(output_f.cpu().numpy()))
                else:
                    val_loss_d.append(errD.cpu())
                    val_real_accuracy_d.append(sum(output_r.cpu().numpy()>0.5)/len(output_r.cpu().numpy()))
                    val_fake_accuracy_d.append(sum(output_f.cpu().numpy()<0.5)/len(output_f.cpu().numpy()))
                #continue
                
                #############################
                ## (2) Update G network: maximize log(D(G(z)))
                ############################
                
                if log['trainging_mode']=='discriminator':
                    continue
                if phase == 'train':
                    if log['trainging_mode']=='all' or log['trainging_mode']=='generator':
                        model.train()  # Set model to training mode
                    else:
                        model.eval()   # Set model to evaluate mode
                else:
                    model.eval()
                netD.eval() 
                
                
                # zero the parameter gradients
                optimizerG.zero_grad()
                model.zero_grad()
                label.fill_(real_label)
                
                #continue
                # forward
                # track history if only in train
                outputs = model(inputs)['out']#.detach()
                seg_labels_out = nn.Sigmoid()(outputs)
                fake_combined = torch.cat((inputs, seg_labels_out), dim=1)
                
                #continue
                output = netD(fake_combined).view(-1)
                
                errG = criterion(output, label)
                output = output.detach()
                # backward + optimize only if in training phase
                if phase == 'train':
                    if log['trainging_mode']=='all' or log['trainging_mode']=='generator':
                        errG.backward()
                        optimizerG.step()
                errG = errG.detach()
                
                
                if phase == 'train':
                    train_loss_g.append(errG.detach().cpu())
                    train_fake_accuracy_g.append(sum(output.cpu().numpy()>0.5)/len(output.cpu().numpy()))
                else:
                    val_loss_g.append(errG.detach().cpu())
                    val_fake_accuracy_g.append(sum(output.cpu().numpy()>0.5)/len(output.cpu().numpy()))
                
                
                
            if phase == 'train':
                #scheduler.step()
                if len(train_loss_g)>0:
                    train_loss_g_epoch.append(str(np.mean(train_loss_g)))
                    train_fake_accuracy_g_epoch.append(str(np.mean(train_fake_accuracy_g)))
                    print(f'{phase} Model Loss: {np.mean(train_loss_g):.4f} - Accuracy: {np.mean(train_fake_accuracy_g):.4f}')
                if len(train_loss_d)>0:
                    train_loss_d_epoch.append(str(np.mean(train_loss_d)))
                    train_real_accuracy_d_epoch.append(str(np.mean(train_real_accuracy_d)))
                    train_fake_accuracy_d_epoch.append(str(np.mean(train_fake_accuracy_d)))
                    print(f'{phase} Discriminator Loss: {np.mean(train_loss_d):.4f} - Accuraccy Real Interp: {np.mean(train_real_accuracy_d):.4f} - Accuraccy Seg Interp: {np.mean(train_fake_accuracy_d):.4f}')
                
            else:
                if len(val_loss_g)>0:
                    val_loss_g_epoch.append(str(np.mean(val_loss_g)))
                    val_fake_accuracy_g_epoch.append(str(np.mean(val_fake_accuracy_g)))
                    print(f'{phase} Model Loss: {np.mean(val_loss_g):.4f} - Accuracy: {np.mean(val_fake_accuracy_g):.4f}')
                if len(val_loss_d)>0:
                    val_loss_d_epoch.append(str(np.mean(val_loss_d)))
                    val_real_accuracy_d_epoch.append(str(np.mean(val_real_accuracy_d)))
                    val_fake_accuracy_d_epoch.append(str(np.mean(val_fake_accuracy_d)))
                    print(f'{phase} Discriminator Loss: {np.mean(val_loss_d):.4f} - Accuraccy Real Interp: {np.mean(val_real_accuracy_d):.4f} - Accuraccy Seg Interp: {np.mean(val_fake_accuracy_d):.4f}')
                
            
        
        if epoch%5 == 0:
            torch.save(model.state_dict(), os.path.join(log['model_path'],log['name'],str(epoch)+'_epoch_model_params.pt'))
            torch.save(netD.state_dict(), os.path.join(log['model_path'],log['name'],str(epoch)+'_epoch_netD_params.pt'))
            log2 = log.copy()
            log2['train_loss_g'] = train_loss_g_epoch
            log2['val_loss_g'] = val_loss_g_epoch
            log2['train_loss_d'] = train_loss_d_epoch
            log2['val_loss_d'] = val_loss_d_epoch
            log2['train_real_accuracy_d_epoch'] = train_real_accuracy_d_epoch
            log2['train_fake_accuracy_d_epoch'] = train_fake_accuracy_d_epoch
            log2['val_real_accuracy_d_epoch'] = val_real_accuracy_d_epoch
            log2['val_fake_accuracy_d_epoch'] = val_fake_accuracy_d_epoch
            
            if log2['initial_weights'] == 'default':    
                with open(os.path.join(log2['model_path'],log2['name'],str(epoch)+'_epoch_training_log.json'), 'w') as f:
                    record = {}
                    record[0] = log2
                    json.dump(record, f)
            else:
                with open(os.path.join(log2['initial_weights'],'training_log.json'), 'r') as f:
                    record = json.load(f)
                    record[len(record)] = log2
                with open(os.path.join(log2['model_path'],log2['name'],str(epoch)+'_epoch_training_log.json'), 'w') as f:
                    json.dump(record, f)
                    
            #plt.plot(train_loss_g_epoch)
            #plt.plot(val_loss_g_epoch)
            #plt.legend(['training loss g','validation loss g'])
            #plt.xlabel('Epoch')
            #plt.ylabel('Binary Cross Entropy')
            #plt.grid()
            #plt.savefig(os.path.join(log['model_path'],log['name'],epoch+'loss_g.png'))
            #plt.clf()
            #plt.plot(train_loss_d_epoch)
            #plt.plot(val_loss_d_epoch)
            #plt.legend(['training loss d','validation loss d'])
            #plt.xlabel('Epoch')
            #plt.ylabel('Binary Cross Entropy')
            #plt.grid()
            #plt.savefig(os.path.join(log['model_path'],log['name'],epoch+'loss_d.png'))
            #plt.clf()
    
        print()
 
    return model, log

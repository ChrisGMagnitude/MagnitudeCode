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

class my_round_func2(torch.autograd.Function):
    def forward(ctx, i):
        result = torch.round(i)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

def train_model(model, netD, optimizerG, optimizerD, criterion,
                device, dataloaders, log, num_epochs=25,
                real_label = 1, fake_label = 0, label_smoothing = 0.1, 
                clip_value = 0.01, n_critic = 5, l1_lambda = 100):
    

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
    
    l1 = nn.L1Loss()
    
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
        for phase in ['train']:#, 'val']:
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
            
            first = True
            # Iterate over data.
            batch = -1
            for inputs, labels in tqdm.tqdm(dataloaders[phase],ascii=True):
                batch+=1
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
                model.requires_grad = False
                
                # netD to training mode only for training loop, otherwise eval mode
                if phase == 'train':
                    if log['trainging_mode']=='all' or log['trainging_mode']=='discriminator':
                        netD.train()  # Set model to training mode
                        netD.requires_grad = True
                    else:
                        netD.eval()   # Set model to evaluate mode
                else:
                    netD.eval()   # Set model to evaluate mode
                
                ## Train with all-real batch
                netD.zero_grad()
                optimizerD.zero_grad()
                
                # Format batch
                b_size = combined.size(0)
                label = torch.full((b_size,), real_label - label_smoothing, dtype=torch.float, device=device)
                # Forward pass real batch through D
                output_r = netD(combined).view(-1)
                
                # Calculate loss on all-real batch
                #errD_real = criterion(output_r, label)
                errD_real = output_r.mean().item()
                
                output_r = output_r.detach()
                
                if first:
                    print('Real D')
                    print('label',label)
                    print('output_r',output_r)
                    print('errD_real',errD_real)
                    
                # Calculate gradients for D in backward pass
                if phase == 'train':
                    if log['trainging_mode']=='all' or log['trainging_mode']=='discriminator':
                        errD_real.backward()
                
                
                ## Create all fake batch       
                fake_segmentartion = model(inputs)['out'].detach()
                seg_labels_out = nn.Sigmoid()(fake_segmentartion)
                seg_labels_out = my_round_func2.apply(seg_labels_out)
                fake_combined = torch.cat((inputs, seg_labels_out), dim=1)
                
                # Create fake label
                label.fill_(fake_label)
                # Classify all fake batch with D
                output_f = netD(fake_combined.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                #errD_fake = criterion(output_f, label)
                errD_fake = -torch.mean(output_f)
                output_f = output_f.detach()
                
                if first:
                    print('Fake D')
                    print('label',label)
                    print('output_f',output_f)
                    print('errD_fake',errD_fake)
                    
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                if phase == 'train':
                    if log['trainging_mode']=='all' or log['trainging_mode']=='discriminator':
                       errD_fake.backward()
                
                
                if phase == 'train':
                    if log['trainging_mode']=='all' or log['trainging_mode']=='discriminator':
                        optimizerD.step()
                
                for p in netD.parameters():
                    p.data.clamp_(-clip_value, clip_value)    
                    
                    
                # Compute error of D as sum over the fake and the real batches
                errD_real = errD_real.detach()
                errD_fake = errD_fake.detach()
                errD = errD_real + errD_fake
                
                if first:
                    print('Combined D')
                    print('errD',errD)
                
                # Update D
                if phase == 'train':
                    train_loss_d.append(errD.cpu())
                    train_real_accuracy_d.append(sum(output_r.cpu().numpy()>0.5)/len(output_r.cpu().numpy()))
                    train_fake_accuracy_d.append(sum(output_f.cpu().numpy()<0.5)/len(output_f.cpu().numpy()))
                else:
                    val_loss_d.append(errD.cpu())
                    val_real_accuracy_d.append(sum(output_r.cpu().numpy()>0.5)/len(output_r.cpu().numpy()))
                    val_fake_accuracy_d.append(sum(output_f.cpu().numpy()<0.5)/len(output_f.cpu().numpy()))

                
                #############################
                ## (2) Update G network: maximize log(D(G(z)))
                ############################
                
                if (batch + 1) % n_critic == 0:
                    if phase == 'train':
                        if log['trainging_mode']=='all' or log['trainging_mode']=='generator':
                            model.train()  # Set model to training mode
                            model.requires_grad = True
                        else:
                            model.eval()   # Set model to evaluate mode
                    else:
                        model.eval()
                    netD.eval() 
                    netD.requires_grad = False


                    # zero the parameter gradients
                    optimizerG.zero_grad()
                    model.zero_grad()
                    label.fill_(real_label)

                    # forward
                    # track history if only in train

                    outputs = model(inputs)['out']#.detach()
                    seg_labels_out = nn.Sigmoid()(outputs)
                    seg_labels_out = my_round_func2.apply(seg_labels_out)
                    fake_combined = torch.cat((inputs, seg_labels_out), dim=1)
                    output = netD(fake_combined).view(-1)
                    #errG = criterion(output, label)
                    errG = torch.mean(output)
                    
                    if first:
                        print('Fake G')
                        print('label',label)
                        print('output',output)
                        print('errG',errG)
                        first = False

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
            log2['train_fake_accuracy_g_epoch'] = train_fake_accuracy_g_epoch
            log2['val_fake_accuracy_g_epoch'] = val_fake_accuracy_g_epoch
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

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

def train_model(model, criterion, optimizer, scheduler, device, dataloaders, log, num_epochs=25):
    since = time.time()
    miou = MeanIoU().to(device)
    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0
        best_epoch = 0
        
        train_loss = []
        val_loss = []
        train_acc = []
        val_acc = []
        
        
        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode
                
                running_loss = []
                running_IOU = []

                # Iterate over data.
                for inputs, labels in tqdm.tqdm(dataloaders[phase],ascii=True):
                    #for inputs, labels in dataloaders[phase]:

                    labels = labels.type(torch.float)
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)['out']
                        preds = torch.round(outputs)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss.append(loss.item() * inputs.size(0))
                    
                    #print(preds)
                    #print(torch.round(preds))
                    #print(torch.round(preds).int())
                    #print(torch.round(preds).int().cpu())
                    #print(torch.round(preds).int().cpu().numpy())

                    #print(np.max(torch.round(preds).int().cpu().numpy()))
                    #print(np.max(labels.int().cpu().numpy()))
                    #print(np.min(torch.round(preds).int().cpu().numpy()))
                    #print(np.min(labels.int().cpu().numpy()))
                    #running_IOU.append(0)
                    class_preds = (preds>0.5).int()
                    running_IOU.append(miou(class_preds, labels.int()).item())
                    print(running_IOU)
                    

                if phase == 'train':
                    scheduler.step()
                    
                    epoch_loss = np.mean(running_loss)
                    epoch_acc = np.mean(running_IOU)
                    train_loss.append(epoch_loss)
                    train_acc.append(epoch_acc.tolist())
                else:
                    epoch_loss = np.mean(running_loss)
                    epoch_acc = np.mean(running_IOU)
                    val_loss.append(epoch_loss)
                    val_acc.append(epoch_acc.tolist())
                    
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                print()
                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_epoch = epoch
                    torch.save(model.state_dict(), best_model_params_path)

            print()
        
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path, weights_only=True))
        
        torch.save(model.state_dict(), os.path.join(log['model_path'],log['name'],'best_model_params.pt'))
        
        log['train_loss'] = train_loss
        log['val_loss'] = val_loss
        log['train_acc'] = train_acc
        log['val_acc'] = val_acc
        
        log['best_epoch'] = best_epoch
        
        
        plt.plot(train_loss)
        plt.plot(val_loss)
        plt.legend(['training loss','validation loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Binary Cross Entropy')
        plt.grid()
        
        plt.savefig(os.path.join(log['model_path'],log['name'],'loss.png'))
        plt.clf()
        
        plt.plot(train_acc)
        plt.plot(val_acc)
        plt.legend(['training acc','validation acc'])
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid()
        
        plt.savefig(os.path.join(log['model_path'],log['name'],'accuracy.png'))
        plt.clf()
        
    return model, log

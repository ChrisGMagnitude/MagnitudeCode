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
        train_acc_pc = []
        val_acc_pc = []
        
        
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
                running_IOU_per_class = []

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
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss.append(loss.item() * inputs.size(0)) # * Batch Size
                    

                    class_preds = (outputs>0.5).int()
                    running_IOU.append(miou(class_preds, labels.int()).item())
                    running_IOU_per_class.append(miou(class_preds, labels.int(), per_class=True).item())
                    #print(running_IOU)
                    

                if phase == 'train':
                    scheduler.step()
                    
                    epoch_loss = np.mean(running_loss)
                    epoch_acc = np.mean(running_IOU)
                    train_loss.append(epoch_loss)
                    train_acc.append(epoch_acc.tolist())
                    train_acc_pc.append(np.array(running_IOU_per_class).mean(axis=0))
                else:
                    epoch_loss = np.mean(running_loss)
                    epoch_acc = np.mean(running_IOU)
                    val_loss.append(epoch_loss)
                    val_acc.append(epoch_acc.tolist())
                    val_acc_pc.append(np.array(running_IOU_per_class).mean(axis=0))
                    
                print(f'{phase} Loss: {epoch_loss:.4f} Total IoU: {epoch_acc:.4f}')
                print(f'IoU per class: {val_acc_pc[-1]}')
                print()
                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_epoch = epoch
                    torch.save(model.state_dict(), best_model_params_path)
            
            if epoch%5 == 0:
                torch.save(model.state_dict(), os.path.join(log['model_path'],log['name'],str(epoch)+'_epoch_model_params.pt'))

                log2 = log.copy()
                log2['train_loss'] = train_loss
                log2['val_loss'] = val_loss
                log2['train_acc'] = train_acc
                log2['val_acc'] = val_acc
                log2['train_acc_pc'] = train_acc_pc
                log2['val_acc_pc'] = val_acc_pc
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
        
            print()
        

        torch.save(model.state_dict(), os.path.join(log['model_path'],log['name'],'last_model_params.pt'))


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

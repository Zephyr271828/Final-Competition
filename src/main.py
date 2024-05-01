import os
import random
import argparse
import numpy as np

from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import models, transforms

from models import *
#from configurations import *
from data_utils import CustomDataset

# fix seeds
seed = 2024
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# device setting
device = torch.device("cuda" if torch.cuda.is_available() else "mps") # mps works better than cpu on mac

# hyper parameters
C = 4
input_ch = 1

lr = 1e-3
momentum = 0.9
weight_decay = 1e-3

def train(model, loader, criterion, optimizer, device):
    avg_loss, N = 0, 0
    for idx, (x, label) in enumerate(tqdm(loader)):
        x, label = x.to(device), label.to(device)
        logits = model(x) 
        loss = criterion(logits, label) 
        avg_loss += loss
        N += 1
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        if (idx + 1) % 100 == 0:
            print('loss = {}'.format(avg_loss / N))
    avg_loss /= N
    
    return avg_loss
    
def eval(models, loader, device, predict = False, file_path = None):
    if predict:
        i = 0
        with open(file_path, 'w+') as f:
            f.write('id,category\n')

    with torch.no_grad():
        tot_corr = 0
        tot_num = 0
        
        n = len(models)
        for x, label in tqdm(loader):
            x, label = x.to(device), label.to(device)
            logits = torch.zeros((n, x.shape[0], C))
            for idx, model in enumerate(models):
                logits[idx, :, :] = model(x)

            logits, _ = torch.max(logits, dim = 0)
            pred = logits.argmax(dim=1).to(device)

            if predict:
                with open(file_path, 'a+') as f:
                    for each in pred:
                        f.write(f'{i},{each}\n')
                        i += 1
            
            tot_corr += torch.eq(pred, label).float().sum().item() # using item() to convert tensor to number
            tot_num += x.size(0)
        acc = tot_corr / tot_num
    
        return acc


if __name__ == '__main__':
    # load parameters
    parser = argparse.ArgumentParser(description = 'model and data settings')
    parser.add_argument('--epochs', type = int, default = 10, help = 'epochs to train')
    parser.add_argument('--batch_size', type = int, default = 16, help = 'batch_size of data loader')
    parser.add_argument('--input_size', type = int, default = 64, help = 'size of input images')
    parser.add_argument('--train', type = bool, default = False, help = 'train the model or not. Use None for False.')
    parser.add_argument('--model', type = str, default = 'ResNet', help = 'model to use. options: ResNet, ViT')
    parser.add_argument('--debug', type = bool, default = False, help = 'partially generate dataset')
    parser.add_argument('--ensemble', type = int, default = 1, help = 'number of models for ensemble')

    args = parser.parse_args()

    print('loading datasets...')
    
    train_transforms = [
        transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)), 
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ]),
        # transforms.Compose([
        #     transforms.Resize((args.input_size, args.input_size)), 
        #     transforms.ToTensor(),
        #     transforms.Normalize(0.5, 0.5),
        #     transforms.RandomHorizontalFlip(0.5),
        #     transforms.RandomVerticalFlip(0.5),
        # ]),
        # transforms.Compose([
        #     transforms.RandomCrop((args.input_size, args.input_size)), 
        #     transforms.ToTensor(),
        #     transforms.Normalize(0.5, 0.5),
        # ]), 
    ]
    
    test_transforms = [
        transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)), 
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])
    ]

    if args.train:

        train_set = CustomDataset(
            gz_dir = '../../data/train.gz',
            my_transforms = train_transforms,
            debug = False,
            balance = 0
        )
        train_loader = DataLoader(dataset = train_set, batch_size = args.batch_size, shuffle = True, drop_last = False)
        
        dev_set = CustomDataset(
            gz_dir = '../../data/dev.gz',
            my_transforms = test_transforms,
            debug = False,
            balance = 0
        )
        dev_loader = DataLoader(dataset = dev_set, batch_size = args.batch_size, shuffle = True, drop_last = False)

    test_set = CustomDataset(
        gz_dir = '../../data/test.gz',
        my_transforms = test_transforms,
        debug = False,
        balance = 0
    )
    test_loader = DataLoader(dataset = test_set, batch_size = args.batch_size, shuffle = False, drop_last = False)

    if args.model.lower() == 'resnet':
        models = [CustomResNet(in_channels = input_ch, num_classes = C) for i in range(args.ensemble)]

    elif args.model.lower() == 'vit':
        model = [models.vit_b_16(image_size = args.input_size, num_classes = C) for i in range(args.ensemble)]

    models = [model.to(device) for model in models]

    if args.ensemble > 1:
        model_paths = [f'../../checkpoints/{args.model.lower()}_{i}.pth' for i in range(args.ensemble)]
        acc_path = f'../../checkpoints/{args.model.lower()}_ensemble_best.txt'
    else:
        model_paths = [f'../../checkpoints/{args.model.lower()}.pth']
        acc_path = f'../../checkpoints/{args.model.lower()}_best.txt'
    pred_path = f'../../checkpoints/{args.model.lower()}_pred.csv'

    # for model, model_path in zip(models, model_paths):
    #     model.eval()
    #     model.load_state_dict(torch.load(model_path))

    if args.train:
    
        criterion = nn.CrossEntropyLoss().to(device)
        optimizers = [optim.SGD(model.parameters(), lr = lr, momentum = momentum, weight_decay = weight_decay) for model in models]
        schedulers = [optim.lr_scheduler.MultiStepLR(optimizer, milestones = [args.epochs // 3, args.epochs * 2 // 3], gamma = 0.1, last_epoch = -1) for optimizer in optimizers]
        with open(acc_path, 'r+') as f:
            best_acc = float(f.read().strip())

        for epoch in range(1, args.epochs + 1):

            print('training for epoch {}...'.format(epoch, ))
            loss = 0.0
            for model, optimizer, scheduler in zip(models, optimizers, schedulers):
                model.train()
                loss += train(model, train_loader, criterion, optimizer, device)
                scheduler.step()
            loss /= args.ensemble

            print('evaluating for epoch {}...'.format(epoch, ))
            for model in models:
                model.eval()
            train_acc = eval(models, train_loader, device, predict = False)
            dev_acc = eval(models, dev_loader, device, predict = False)

            lr = schedulers[0].get_last_lr()[0]
            print('epoch = {} | lr = {} | loss = {} | train acc = {}% | dev acc = {}%'.format(epoch, lr, loss, train_acc * 100, dev_acc * 100))

            if dev_acc > best_acc:
                best_acc = dev_acc
                with open(acc_path, 'w+') as f:
                    f.write(str(best_acc))
                print('saving model parameters...')
                for model, model_path in zip(models, model_paths):
                    torch.save(model.state_dict(), model_path)

    print('generating predictions...')

    for model, model_path in zip(models, model_paths):
        model.eval()
        model.load_state_dict(torch.load(model_path))
    _ = eval(models, test_loader, device, predict = True, file_path = pred_path)
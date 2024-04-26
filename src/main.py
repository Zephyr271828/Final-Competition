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
input_ch = 3

lr = 1e-3
momentum = 0.9
weight_decay = 1e-3
model_path = 'model.pth'

def train(model, loader, criterion, optimizer, device, scheduler = None):
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
        if scheduler:
            scheduler.step()

        if (idx + 1) % 100 == 0:
            print('loss = {}'.format(avg_loss / N))
    avg_loss /= N
    
    return avg_loss
    
def eval(model, loader, device, predict = False, file_path = None):
    if predict:
        i = 0
        with open(file_path, 'w+') as f:
            f.write('id,category\n')

    with torch.no_grad():
        tot_corr = 0
        tot_num = 0
        
        for x, label in tqdm(loader):
            x, label = x.to(device), label.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)

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
    parser.add_argument('--data_dir', type = str, default = '../../data', help = 'root directory of data')
    parser.add_argument('--batch_size', type = int, default = 16, help = 'batch_size of data loader')
    parser.add_argument('--input_size', type = int, default = 64, help = 'size of input images')
    args = parser.parse_args()

    print('loading datasets...')
    
    transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)), 
        transforms.ToTensor()
    ])

    train_set = CustomDataset(f'{args.data_dir}/train', label = True, transform = transform)
    train_loader = DataLoader(dataset = train_set, batch_size = args.batch_size, shuffle = True, drop_last = False)
    
    dev_set = CustomDataset(f'{args.data_dir}/dev', label = True, transform = transform)
    dev_loader = DataLoader(dataset = dev_set, batch_size = args.batch_size, shuffle = True, drop_last = False)

    test_set = CustomDataset(f'{args.data_dir}/test_imgs', label = False, transform = transform)
    test_loader = DataLoader(dataset = test_set, batch_size = args.batch_size, shuffle = False, drop_last = False)

    print('loading model...')
    model = models.resnet18(weights = None, num_classes = C)
    model.conv1 = nn.Conv2d(input_ch, model.conv1.weight.shape[0], 3, 1, 1, bias = False)
    model.maxpool = nn.MaxPool2d(kernel_size = 1, stride = 1, padding = 0)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr = lr, momentum = momentum, weight_decay = weight_decay)
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [args.epochs // 3, args.epochs * 2 // 3], gamma = 0.1, last_epoch = -1)
    scheduler = None
    with open('../best_acc.txt', 'r+') as f:
        best_acc = float(f.read().strip())

    for epoch in range(1, args.epochs + 1):
        print('training for epoch {}...'.format(epoch, ))
        model.train()
        loss = train(model, train_loader, criterion, optimizer, device, scheduler)

        print('evaluating for epoch {}...'.format(epoch, ))
        model.eval()
        train_acc = eval(model, train_loader, device, predict = False)
        dev_acc = eval(model, dev_loader, device, predict = False)

        print('epoch = {} | train acc = {}% | dev acc = {}%'.format(epoch, train_acc * 100, dev_acc * 100))

        if dev_acc > best_acc:
            best_acc = dev_acc
            with open('../best_acc.txt', 'w+') as f:
                f.write(str(best_acc))
            print('saving model parameters...')
            torch.save(model.state_dict(), model_path)

    print('generating predictions...')
    model = models.resnet18(weights = None, num_classes = C)
    model.conv1 = nn.Conv2d(input_ch, model.conv1.weight.shape[0], 3, 1, 1, bias = False)
    model.maxpool = nn.MaxPool2d(kernel_size = 1, stride = 1, padding = 0)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    model.eval()
    acc = eval(model, test_loader, device, predict = True, file_path = '../prediction.csv')
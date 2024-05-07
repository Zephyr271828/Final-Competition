import os
import wandb
import random
import argparse
import numpy as np

from tqdm import tqdm
from datetime import datetime
# from sklearn.model_selection import train_test_split
# from transformers import Wav2Vec2ConformerForSequenceClassification

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import v2
from torchvision.ops import sigmoid_focal_loss
from torchvision.models import resnet18, resnet50, resnet101, resnet152, densenet201, vit_b_16, resnext50_32x4d

import torchaudio
from torchaudio.models import Conformer

from models import CustomResNet, LSTMClassifier
from data_utils import ImageDataset, mixup_data, mixup_criterion


# fix seeds
seed = 38
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # mps works better than cpu on mac

# hyper parameters
C = 4

# initialize wandb
def init_wandb(model, ensemble):
    wandb.init(
        project = 'Final-Competition',
        name = '{} | ensemble={}'.format(model, ensemble)
    )

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12349"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def train(model, loader, criterion, optimizer, device, mixup = False):
    avg_loss, N = 0, 0
    for idx, (x, label) in enumerate(tqdm(loader)):
        x, label = x.to(device), label.to(device)
        if mixup:
            x, label1, label2, lam = mixup_data(x, label, 0.2)

        logits = model(x) 

        if mixup:
            loss = mixup_criterion(criterion, logits, label1, label2, lam)
        else:
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
            logits = sum([model(x) for model in models]) / len(models)

            pred = logits.argmax(dim=1).to(device)

            if predict:
                with open(file_path, 'a+') as f:
                    for each in pred:
                        f.write(f'{i},{each}\n')
                        i += 1
            
            tot_corr += torch.eq(pred, label).float().sum().item() 
            tot_num += x.size(0)
        acc = tot_corr / tot_num
    
        return acc

def main(rank, world_size, device, args):
    # ddp_setup(rank, world_size)
    # local_rank = rank - world_size * (rank // world_size)
    # torch.cuda.set_device(local_rank)
    # ddp is not fully implemented considering it may improve efficiency at the price of performance

    if args.wandb: 
        init_wandb(args.model, args.ensemble)

    print('loading datasets...')
    
    audio_transform = transforms.Compose([
        torchaudio.transforms.MelSpectrogram(sample_rate = 22050, n_fft = 2048),
        torchaudio.transforms.AmplitudeToDB(),
    ])
    
    audio_transform2 = transforms.Compose([
        torchaudio.transforms.MelSpectrogram(sample_rate = 22050, n_fft = 2048),
        torchaudio.transforms.AmplitudeToDB(),
        torchaudio.transforms.TimeMasking(time_mask_param = 50, iid_masks = True, p = 1.0)
    ])

    audio_transform3 = transforms.Compose([
        torchaudio.transforms.MelSpectrogram(sample_rate = 22050, n_fft = 2048),
        torchaudio.transforms.AmplitudeToDB(),
        torchaudio.transforms.FrequencyMasking(freq_mask_param = 100, iid_masks = True)
    ])

    image_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
    ])

    image_transform2 = transforms.Compose([
        transforms.RandomCrop((args.input_size * 4, args.input_size * 4)),
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
    ])
    
    image_transform3 = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p = 0.75),
        transforms.RandomVerticalFlip(p = 0.75),
    ])

    if args.extension == '':
        train_transforms = [audio_transform, audio_transform2]
        test_transforms = [audio_transform]
        args.input_ch = 1
    elif args.extension == '_vocal':
        train_transforms = [audio_transform, audio_transform2]
        test_transforms = [audio_transform]
        args.input_ch = 2
    elif args.extension == '_imgs':
        train_transforms = [image_transform, image_transform2, image_transform3]
        test_transforms = [image_transform]
        args.input_ch = 3

    if args.train:

        train_set = CustomDataset(
            gz_dir = f'../../data/train{args.extension}.gz',
            my_transforms = train_transforms,
            expansion = 1,
            balance = 3
        )

        train_loader = DataLoader(
            dataset = train_set, 
            batch_size = args.batch_size, 
            shuffle = True, 
            drop_last = False
        )

        # train_set2 = CustomDataset(
        #     gz_dir = f'../../data/train{args.extension}.gz',
        #     my_transforms = [audio_transform2],
        #     expansion = 1,
        #     balance = 3
        # )

        # train_loader2 = DataLoader(
        #     dataset = train_set2, 
        #     batch_size = args.batch_size, 
        #     shuffle = True, 
        #     drop_last = False)

        # train_loaders = [train_loader, train_loader2]
        # train_loaders = [train_loader for i in range(args.ensemble)]

        dev_set = CustomDataset(
            gz_dir = f'../../data/dev{args.extension}.gz',
            my_transforms = test_transforms,
            balance = 0
        )

        dev_loader = DataLoader(
            dataset = dev_set, 
            batch_size = args.batch_size, 
            shuffle = True, 
            drop_last = False
        )

    test_set = CustomDataset(
        gz_dir = f'../../data/test{args.extension}.gz',
        my_transforms = test_transforms,
        balance = 0
    )

    test_loader = DataLoader(
        dataset = test_set, 
        batch_size = args.batch_size, 
        shuffle = False, 
        drop_last = False
    )

    if 'resnet' in args.model.lower():
        if args.model.lower() == 'resnet':
            models = [CustomResNet(in_channels = args.input_ch, num_classes = C) for i in range(args.ensemble)]
        elif args.model.lower() == 'resnet18':
            models = [resnet18(weights = None, num_classes = C) for i in range(args.ensemble)]
        elif args.model.lower() == 'resnet50':
            models = [resnet50(weights = None, num_classes = C) for i in range(args.ensemble)]
        elif args.model.lower() == 'resnet101':
            models = [resnet101(weights = None, num_classes = C) for i in range(args.ensemble)]
        elif args.model.lower() == 'resnet152':
            models = [resnet152(weights = None, num_classes = C) for i in range(args.ensemble)]

        for model in models:
            model.conv1 = nn.Conv2d(args.input_ch, model.conv1.weight.shape[0], 3, 1, 1, bias = False)
            model.maxpool = nn.MaxPool2d(kernel_size = 1, stride = 1, padding = 0)

    elif 'resnext' in args.model.lower():
        models = [resnext50_32x4d(weights = None, num_classes = C) for i in range(args.ensemble)]
        for model in models:
            model.conv1 = nn.Conv2d(args.input_ch, model.conv1.weight.shape[0], 3, 1, 1, bias = False)
            model.maxpool = nn.MaxPool2d(kernel_size = 1, stride = 1, padding = 0)

    elif 'densenet' in args.model.lower():
        models = [densenet201(weights = None, num_classes = C) for i in range(args.ensemble)]
        for model in models:
            model.features[0] = nn.Conv2d(args.input_ch, 64, 3, 1, 1, bias = False)
            model.features[3] = nn.MaxPool2d(kernel_size = 1, stride = 1, padding = 0)

    elif args.model.lower() == 'vit':
        models = [vit_b_16(image_size = args.input_size, num_classes = C) for i in range(args.ensemble)]

    elif args.model.lower() == 'lstm':
        models = [LSTMClassifier(
            input_size = args.input_size, 
            embed_size = args.input_size,
            hidden_size = args.input_size * 2, 
            num_layers = 2,
            label_size = C,
            batch_size = args.batch_size,
            bidirectional = True
        ) for i in range(args.ensemble)]

    elif args.model.lower() == 'conformer':
        models = [Wav2Vec2ConformerForSequenceClassification.from_pretrained('facebook/wav2vec2-base-100h') for i in range(args.ensemble)]
        for model in models:
            model.classifier = nn.Linear(in_features = 256, out_features = 4, bias = True)

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
    
        criterion = nn.CrossEntropyLoss(label_smoothing = args.label_smoothing).to(device)
        optimizers = [optim.SGD(model.parameters(), lr = args.lr, momentum = args.momentum, weight_decay = args.weight_decay) for model in models]
        schedulers = [optim.lr_scheduler.MultiStepLR(optimizer, milestones = [args.epochs // 4, args.epochs // 2, args.epochs * 3 // 4], gamma = 0.1, last_epoch = -1) for optimizer in optimizers]
        # schedulers = [optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.epochs, eta_min = 0, last_epoch = -1) for optimizer in optimizers]
        # schedulers = [optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = args.epochs // 4, eta_min = 0, last_epoch = -1) for optimizer in optimizers]

        for epoch in range(1, args.epochs + 1):

            print('training for epoch {}...'.format(epoch, ))
            loss = 0.0
            for idx, (model, optimizer, scheduler) in enumerate(zip(models, optimizers, schedulers)):
                model.train()
                if idx == 3:
                    loss += train(model, train_loader, criterion, optimizer, device, mixup = True)
                else:
                    loss += train(model, train_loader, criterion, optimizer, device, mixup = args.mixup)
                scheduler.step()
            loss /= args.ensemble

            print('evaluating for epoch {}...'.format(epoch, ))
            for model in models:
                model.eval()
            train_acc = eval(models, train_loader, device, predict = False)
            dev_acc = eval(models, dev_loader, device, predict = False)

            lr = schedulers[0].get_last_lr()[0]
            print('epoch = {} | lr = {} | loss = {} | train acc = {}% | dev acc = {}%'.format(epoch, lr, loss, train_acc * 100, dev_acc * 100))
            if args.wandb:
                wandb.log({
                    'loss' : loss,
                    'train accuracy' : train_acc,
                    'valiation accuracy' : dev_acc
                })

            try:
                with open(acc_path, 'r+') as f:
                    best_acc = float(f.read().strip())
            except:
                best_acc = 0.0
            if dev_acc > best_acc:
                best_acc = dev_acc
                with open(acc_path, 'w+') as f:
                    f.write(str(best_acc))
                print('saving model parameters...')
                for model, model_path in zip(models, model_paths):
                    torch.save(model.state_dict(), model_path)

    if args.wandb:
        wandb.finish()

    print('generating predictions...')

    for model, model_path in zip(models, model_paths):
        model.eval()
        model.load_state_dict(torch.load(model_path))
    _ = eval(models, test_loader, device, predict = True, file_path = pred_path)


if __name__ == '__main__':
    # load parameters
    parser = argparse.ArgumentParser(description = 'model and data settings')
    parser.add_argument('--epochs', type = int, default = 24, help = 'epochs to train')
    parser.add_argument('--lr', type = float, default = 1e-3, help = 'learning rate')
    parser.add_argument('--momentum', type = float, default = 0.9, help = 'momentum')
    parser.add_argument('--weight_decay', type = float, default = 1e-3, help = 'weight decay')
    parser.add_argument('--batch_size', type = int, default = 16, help = 'batch_size of data loader')
    parser.add_argument('--input_size', type = int, default = 128, help = 'size of input images')
    parser.add_argument('--input_ch', type = int, default = 1, help = 'channels of input images')
    parser.add_argument('--label_smoothing', type = float, default = 0.0, help = 'label_smoothing for cross entropy loss')
    parser.add_argument('--train', type = bool, default = False, help = 'train the model or not. Use None for False.')
    parser.add_argument('--model', type = str, default = 'ResNet18', help = 'model to use')
    parser.add_argument('--debug', type = bool, default = False, help = 'partially generate dataset')
    parser.add_argument('--ensemble', type = int, default = 1, help = 'number of models for ensemble')
    parser.add_argument('--wandb', type = bool, default = False, help = 'use wandb or not')
    parser.add_argument('--extension', type = str, default = '', help = '_imgs or _vocals')
    parser.add_argument('--mixup', type = bool, default = False, help = 'use mixup augmentation or not')

    args = parser.parse_args()

    world_size = torch.cuda.device_count()

    main(rank = None, world_size = world_size, device = device, args = args)
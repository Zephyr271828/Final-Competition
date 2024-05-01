import os
import gzip
import pickle
import argparse
import librosa
from PIL import Image
from tqdm import tqdm
from multiprocess import Pool

import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import models
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, mp3_dir, label_dir = None, transforms = None, debug = False, n_processes = 4, balance = 0):
        self.mp3_dir = mp3_dir
        self.transforms = transforms
        self.balance = balance
        n_files = len([file for file in os.listdir(mp3_dir) if file.endswith('.mp3')])
        if debug:
            n_files = min(n_files, 640)
        # number of files in the directory

        if label_dir:
            self.labels = [int(line.strip()) for line in open(label_dir, 'r+').readlines()]
        else:
            self.labels = [-1 for file in os.listdir(mp3_dir) if file.endswith('.mp3')]
        
        self.data = {}
        for idx in tqdm(range(n_files)):
            self._init_data(idx)
        # with Pool(processes = n_processes) as pool:
        #     pool.starmap(self._init_data, [(idx,) for idx in tqdm(range(n_files))])
        self.data = [pair for pair in self.data[idx] for idx in range(n_files)]

    def _init_data(self, idx):
        copies = []
        mp3_path = os.path.join(self.mp3_dir, f'{idx}.mp3')
        y, sr = librosa.load(mp3_path)
        img = librosa.feature.melspectrogram(y = y, sr = sr)
        # print(img.shape)
        img = Image.fromarray(img)

        label = self.labels[idx]
        for transform in self.transforms:
            tf_img = transform(img)
        
            if label == 0:
                for i in range(self.balance):
                    copies.append((tf_img, label))
            copies.append((tf_img, label))
        self.data[idx] = copies

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def save_data(mp3_dir, label_dir, gz_path):
    n_files = len([file for file in os.listdir(mp3_dir) if file.endswith('.mp3')])
    labels = [int(line.strip()) for line in open(label_dir, 'r+').readlines()]
    with gzip.open(gz_path, 'wb') as f:
        for idx in tqdm(range(n_files)):
            mp3_path = os.path.join(self.mp3_dir, f'{idx}.mp3')
            y, sr = librosa.load(mp3_path)
            img = librosa.feature.melspectrogram(y = y, sr = sr)
            img = Image.fromarray(img)
            label = label[idx]
            pickle.dump((img, label), gz)
            if idx < 3:
                print((img, label))
    
    with gzip.open(gz_path, 'rb') as f:
        data = pickle.load(f)
    print(data[:3])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'model and data settings')
    parser.add_argument('--flag', type = bool, default = False, help = 'True or False. Please use None for False.')
    args = parser.parse_args()

    print(type(args.flag))
    print(args.flag)
    print(bool('False'))
    
    if args.flag:
        print(1)
    else:
        print(2)

    # input_ch = 3
    # model = models.resnet18(weights = None, num_classes = 10)
    # model.conv1 = nn.Conv2d(input_ch, model.conv1.weight.shape[0], 3, 1, 1, bias = False)
    # model.maxpool = nn.MaxPool2d(kernel_size = 1, stride = 1, padding = 0)
    # #print(model.modules)

    mp3_path = f'../../data/train_mp3s/0.mp3'
    y, sr = librosa.load(mp3_path)
    spectrogram = librosa.feature.melspectrogram(y = y, sr = sr)
    print(type(spectrogram))
    print(spectrogram.shape)

    mp3_dir = '../../data/train_mp3s'
    label_dir = '../../data/train_label.txt'

    transforms = [
        transforms.Compose([
            transforms.Resize((128, 128)), 
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])
    ]

    # train_set = CustomDataset(
    #     mp3_dir = mp3_dir, 
    #     label_dir = label_dir, 
    #     transforms = transforms, 
    #     debug = False, 
    #     n_processes = 8,
    #     balance = 3
    # )

    
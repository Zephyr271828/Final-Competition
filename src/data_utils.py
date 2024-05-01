import os
import sys
import gzip
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split

import librosa
import librosa.display

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from configurations import *


np.set_printoptions(threshold=sys.maxsize)

def load_data(mp3_dir, label_dir = None):
    n_files = len([file for file in os.listdir(mp3_dir) if file.endswith('.mp3')])
    if label_dir:
        labels = [int(line.strip()) for line in open(label_dir, 'r+').readlines()]
    else:
        labels = [-1 for file in os.listdir(mp3_dir) if file.endswith('.mp3')]

    data = []
    for idx in tqdm(range(n_files)):
        mp3_path = os.path.join(mp3_dir, f'{idx}.mp3')
        y, sr = librosa.load(mp3_path)
        img = librosa.feature.melspectrogram(y = y, sr = sr)
        img = Image.fromarray(img)
        label = labels[idx]
        data.append((img, label))
    
    return data

class CustomDataset(Dataset):
    def __init__(self, gz_dir, my_transforms = None, debug = False, balance = 0):
        self.raw_data = self._init_data(gz_dir)
        self.processed_data = self._process_data(my_transforms, debug, balance)    

    def _init_data(self, gz_dir):
        with gzip.open(gz_dir, 'rb') as f:
            raw_data = pickle.load(f)
        return raw_data

    def _process_data(self, my_transforms, debug, balance):
        processed_data = []
        for data in tqdm(self.raw_data):
            img, label = data
            for transform in my_transforms:
                tf_img = transform(img)

                if label == 0:
                    for i in range(balance):
                        processed_data.append((tf_img, label))
                processed_data.append((tf_img, label))
        return processed_data

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        return self.processed_data[idx]

if __name__ == '__main__':
    split = 'train'

    
    label_dir = '../../data/train_label.txt'
    gz_dir = '../../data/train.gz'
    print('saving data to .gz file...')
    #init_img(mp3_dir, img_dir)

    # data = load_data(
    #     mp3_dir = '../../data/train_mp3s', 
    #     label_dir = '../../data/train_label.txt'
    # )
    
    # train_idx, dev_idx = train_test_split(, test_size = 0.2, random_state = 42)
    # train = [data[idx] for idx in train_idx]
    # dev = [data[idx] for idx in dev_idx]

    # with gzip.open('../../data/train.gz', 'wb') as f:
    #     pickle.dump(train, f)

    # with gzip.open('../../data/dev.gz', 'wb') as f:
    #     pickle.dump(dev, f)

    # test = load_data(
    #     mp3_dir = '../../data/test_mp3s'
    # )

    # with gzip.open('../../data/test.gz', 'wb') as f:
    #     pickle.dump(test, f)

    my_transforms = [
        transforms.Compose([
            transforms.Resize((128, 128)), 
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])
    ]

    print('reloading the data from .gz file...')
    train_set = CustomDataset(
        gz_dir = gz_dir,
        my_transforms = my_transforms,
        debug = False,
        balance = 0
    )

    # train_loader = DataLoader(dataset = train_set, batch_size = batch_size, shuffle = True)
    # dev_loader = DataLoader(dataset = dev_set, batch_size = batch_size, shuffle = True)
    # test_loader = DataLoader(dataset = test_set, batch_size = batch_size, shuffle = False)

    # for idx, batch in enumerate(test_loader):
    #     img, _ = batch
    #     print(img.shape)
    #     break
    
        

import os
import sys
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

def mp3_to_img(mp3_path, img_path):
    # Load the MP3 file and extract audio data
    y, sr = librosa.load(mp3_path)

    # Convert the audio data into a spectrogram
    spectrogram = librosa.feature.melspectrogram(y = y, sr = sr)

    # Plot the spectrogram
    plt.figure(figsize = (10, 10), dpi = 100)
    librosa.display.specshow(librosa.power_to_db(spectrogram, ref = np.max))
    plt.axis('off')

    # Save the spectrogram as an image
    plt.savefig(img_path, transparent = True)
    plt.close()

def init_img(mp3_dir, img_dir):

    for mp3_name in tqdm(os.listdir(mp3_dir)):
        if '._' in mp3_name:
            mp3_path = os.path.join(mp3_dir, mp3_name)
            os.remove(mp3_path)

    for mp3_name in tqdm(os.listdir(mp3_dir)):
        mp3_path = os.path.join(mp3_dir, mp3_name)

        img_name = mp3_name.replace('.mp3', '.png')
        img_path = os.path.join(img_dir, img_name)
        mp3_to_img(mp3_path, img_path)

def split_data(img_dir, label_dir, train_dir, dev_dir):
    idx = np.arange(len(os.listdir(img_dir)))
    train_idx, dev_idx = train_test_split(idx, test_size = 0.2, random_state = 42)
    labels = [int(line.strip()) for line in open(label_dir, 'r+').readlines()]

    for (new_dir, indices) in zip([train_dir, dev_dir], [train_idx, dev_idx]):
        with open(os.path.join(new_dir, 'labels.txt'), 'w+') as f:
            for new_idx, old_idx in enumerate(tqdm(indices)):

                old_path = os.path.join(img_dir, f'{old_idx}.png')
                img = Image.open(old_path)
                label = labels[old_idx]

                new_path = os.path.join(new_dir, f'{new_idx}.png')
                img.save(new_path)
                f.write(f'{label}\n')

class CustomDataset(Dataset):
    def __init__(self, img_dir, label = False, transform = None, debug = False, balance = 0):
        self.img_dir = img_dir
        self.size = len([file for file in os.listdir(img_dir) if file.endswith('.png')])
        if debug:
            self.size = min(self.size, 640)
        self.transform = transform
        if label:
            self.label_dir = os.path.join(img_dir, 'labels.txt')
            self.labels = [int(line.strip()) for line in open(self.label_dir, 'r+').readlines()]
        else:
            self.label_dir = None
        self.data = self._init_data()
        self.balance = balance

    def _init_data(self):
        data = []
        for idx in tqdm(range(self.size)):
            img_path = os.path.join(self.img_dir, f'{idx}.png')
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            if self.label_dir:
                label = self.labels[idx]
            else:
                label = -1
            if label == 0:
                for i in range(self.balance):
                    data.append((img, label))
                self.size += balance
            data.append((img, label))

        return data

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == '__main__':
    split = 'train'

    mp3_dir = f'../../data/{split}_mp3s'
    img_dir = f'../../data/{split}_imgs'
    label_dir = '../../data/train_label.txt'
    print('initializing image data...')
    #init_img(mp3_dir, img_dir)

    transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor()
    ])

    print('splitting dataset...')
    train_dir = '../../data/train'
    dev_dir = '../../data/dev'
    test_dir = '../../data/test_imgs'
    #split_data(img_dir, label_dir, train_dir, dev_dir)

    print('loading datasets...')
    train_set = CustomDataset(train_dir, label = True, transform = None, debug = True)
    print(np.asarray(train_set[0][0]).shape, train_set[0][1])
    dev_set = CustomDataset(dev_dir, label = True, transform = transform, debug = True)
    test_set = CustomDataset(test_dir, label = False, transform = transform, debug = True)

    train_loader = DataLoader(dataset = train_set, batch_size = batch_size, shuffle = True)
    dev_loader = DataLoader(dataset = dev_set, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(dataset = test_set, batch_size = batch_size, shuffle = False)

    for idx, batch in enumerate(test_loader):
        img, _ = batch
        print(img.shape)
        break
    
        

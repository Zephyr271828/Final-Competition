import os
import sys
import gzip
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from spleeter.separator import Separator
from sklearn.model_selection import train_test_split

import librosa
import librosa.display

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# numpy fully print 
np.set_printoptions(threshold=sys.maxsize)

# fix seeds
seed = 2024
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # mps works better than cpu on mac

def separate(idx, mp3_dir, output_dir):

    audio_file = os.path.join(mp3_dir, f'{idx}.mp3')

    separator.separate_to_file(audio_file, output_dir)

def load_mp3(mp3_dir, label_dir, split = False):

    n_files = len([file for file in os.listdir(mp3_dir)])
    if label_dir:
        labels = [int(line.strip()) for line in open(label_dir, 'r+').readlines()]
    else:
        labels = [-1 for i in range(n_files)]
    data = []
    for idx in tqdm(range(n_files)):
        waveform, sample_rate = torchaudio.load(os.path.join(mp3_dir, f'{idx}/vocals.wav'))
        label = torch.tensor(labels[idx], dtype = torch.long)
        data.append((waveform, label))
    if split:
        idx1, idx2 = train_test_split(list(range(n_files)), test_size = 0.2, random_state = 42)
        data1 = [data[i] for i in idx1]
        data2 = [data[i] for i in idx2]
        return data1, data2
    else:
        return data

def save_data(data, gz_dir):
    
    with gzip.open(gz_dir, 'wb') as f:
        pickle.dump(data, f)

def reload_data(gz_dir):

    with gzip.open(gz_dir, 'rb') as f:
        data = pickle.load(f)

    print(len(data))

def mixup_data(x, y, alpha=1.0, use_cuda=True):

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):

    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class ImageDataset(Dataset):

    def __init__(self, gz_dir, my_transforms = None, expansion = 1, balance = 0):
        self.raw_data = self._init_data(gz_dir)
        self.processed_data = self._process_data(my_transforms, balance, expansion)   

    def _init_data(self, gz_dir):
        with gzip.open(gz_dir, 'rb') as f:
            raw_data = pickle.load(f)
        return raw_data

    def _process_data(self, my_transforms, balance, expansion):
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

    train, dev = load_mp3(
        mp3_dir = 'data/train_output', 
        label_dir = 'data/train_label.txt', 
        split = True
    )

    test = load_mp3(
        mp3_dir = 'data/test_output',
        label_dir = None,
        split = False
    )

    save_data(train, 'data/train_vocal.gz')
    save_data(dev, 'data/dev_vocal.gz')
    save_data(test, 'data/test_vocal.gz')

    reload_data('data/train_vocal.gz')
    
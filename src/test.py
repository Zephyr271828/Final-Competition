import os
import gzip
import pickle
import argparse

from torch.utils.data import Dataset, DataLoader

with open('../best_acc.txt', 'r+') as f:
    best_acc = float(f.read().strip())

print(best_acc)

class CustomDataset(Dataset):
    def __init__(self, img_dir, label = False, transform = None, debug = False):
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
            data.append((img, label))

        return data

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'model and data settings')
    parser.add_argument('--flag', type = bool, default = False, help = 'True or False. Please use None for False.')
    parser.add_argument('--debug', type = bool, default = False, help = 'partially generate dataset')
    args = parser.parse_args()

    print(type(args.flag))
    print(args.flag)
    print(bool('False'))
    
    if args.flag:
        print(1)
    else:
        print(2)

    img_dir = '../../data/train'
    label_dir = os.path.join(img_dir, 'labels.txt')

    size = len([file for file in os.listdir(img_dir) if file.endswith('.png')])
    data = []
    for idx in tqdm(range(size)):
            img_path = os.path.join(self.img_dir, f'{idx}.png')
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            if label_dir:
                label = self.labels[idx]
            else:
                label = -1
            data.append((img, label))

        return data


    

    
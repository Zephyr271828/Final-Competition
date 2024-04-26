import os
import numpy
import random
import argparse

from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from models import *
from configurations import *
from data_utils import load_data



if __name__ == '__main__':

    print('loading datasets...')
    train_dir = '../../data/train'
    dev_dir = '../../data/dev'
    
    train_loader = load_data(train_dir)
    dev_loader = load_data(dev_dir)


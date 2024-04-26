import random
import numpy as np

import torch


# fix seeds
seed = 2024
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# device setting
device = torch.device("cuda" if torch.cuda.is_available() else "mps") # mps works better than cpu on mac

# hyper parameters
batch_size = 16


if __name__ == '__main__':
    pass
import numpy
import random

import torch


# fix seeds
seed = 2024
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# device setting
device = torch.device("cuda" if torch.cuda.is_available() else "mps") # mps works better than cpu on mac

if __name__ == '__main__':
    pass
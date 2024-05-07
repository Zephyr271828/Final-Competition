import random
import numpy as np
from tqdm import tqdm

import torch
import torchaudio
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18, resnet50, resnet101, resnet152, densenet201, vit_b_16, resnext50_32x4d


from main import eval
from models import CustomResNet
from data_utils import ImageDataset

# fix seeds
seed = 2024
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # mps works better than cpu on mac

# hyper parameters
C = 4
input_ch = 2
extension = '_vocal'


audio_transform = transforms.Compose([
    torchaudio.transforms.MelSpectrogram(sample_rate = 22050, n_fft = 2048),
    torchaudio.transforms.AmplitudeToDB(),
])

test_transforms = [audio_transform]

dev_set = ImageDataset(
    gz_dir = f'../../data/dev{extension}.gz',
    my_transforms = test_transforms,
    balance = 0
)
dev_loader = DataLoader(
    dataset = dev_set, 
    batch_size = 16, 
    shuffle = True, 
    drop_last = False
)
test_set = ImageDataset(
    gz_dir = f'../../data/test{extension}.gz',
    my_transforms = test_transforms,
    balance = 0
)
test_loader = DataLoader(
    dataset = test_set, 
    batch_size = 16, 
    shuffle = False, 
    drop_last = False
)

models = [
    CustomResNet(in_channels = input_ch, num_classes = C),
    CustomResNet(in_channels = input_ch, num_classes = C),
    CustomResNet(in_channels = input_ch, num_classes = C),
    CustomResNet(in_channels = input_ch, num_classes = C),
    CustomResNet(in_channels = input_ch, num_classes = C),
    CustomResNet(in_channels = input_ch, num_classes = C),
    CustomResNet(in_channels = input_ch, num_classes = C),
    CustomResNet(in_channels = input_ch, num_classes = C),
    # resnet50(weights = None, num_classes = C),
    # resnet101(weights = None, num_classes = C),
    # densenet201(weights = None, num_classes = C)
]

model_paths = [
    '../../checkpoints/resnet.pth',
    '../../checkpoints/resnet_0.pth',
    '../../checkpoints/resnet_1.pth',
    '../../checkpoints/resnet_2.pth',
    '../../checkpoints/resnet_3.pth',
    '../../checkpoints/resnet_4.pth',
    '../../checkpoints/resnet_5.pth',
    '../../checkpoints/resnet_6.pth',
    # '../../checkpoints/resnet50.pth',
    # '../../checkpoints/resnet101.pth',
    # '../../checkpoints/densenet201.pth'
]

for model, model_path in tqdm(zip(models, model_paths)):
    try:
        model.conv1 = nn.Conv2d(input_ch, model.conv1.weight.shape[0], 3, 1, 1, bias = False)
        model.maxpool = nn.MaxPool2d(kernel_size = 1, stride = 1, padding = 0)
    except:
        model.features[0] = nn.Conv2d(input_ch, 64, 3, 1, 1, bias = False)
        model.features[3] = nn.MaxPool2d(kernel_size = 1, stride = 1, padding = 0)
    model.eval()
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

pred_path = '../prediction.csv'
acc = eval(models, dev_loader, device, predict = False)
print(acc)
_ = eval(models, test_loader, device, predict = True, file_path = pred_path)
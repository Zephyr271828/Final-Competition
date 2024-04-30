import argparse
from torch import nn
from torchvision import models


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

    input_ch = 3
    model = models.resnet18(weights = None, num_classes = 10)
    model.conv1 = nn.Conv2d(input_ch, model.conv1.weight.shape[0], 3, 1, 1, bias = False)
    model.maxpool = nn.MaxPool2d(kernel_size = 1, stride = 1, padding = 0)
    print(model.modules)

    
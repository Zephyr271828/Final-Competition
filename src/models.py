from torch import nn
from torchvision import models
import torch

class BasicBlock(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels,
        stride,
        downsample = False
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size = 3, stride = stride, padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(
            out_channels, 
            eps = 1e-05, momentum = 0.1, affine = True, 
            track_running_stats = True
        )
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size = 3, stride = 1, padding=1,
            bias = False
        )
        self.bn2 = nn.BatchNorm2d(
            out_channels, 
            eps = 1e-05, momentum = 0.1, affine = True, 
            track_running_stats = True
        )
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, 
                    kernel_size = 1, stride = 2, bias = False
                ),
                nn.BatchNorm2d(
                    out_channels, 
                    eps = 1e-05, momentum = 0.1, affine = True, 
                    track_running_stats = True
                )
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return  out



class CustomResNet(nn.Module):

    def __init__(self, input_size = 128, in_channels = 3, num_classes = 4):
        super(CustomResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(64, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 1, stride = 1, padding = 0)

        self.layer1 = nn.Sequential(
            BasicBlock(
                in_channels = 64, out_channels = 64, 
                stride = 1, downsample = False
            ),
            BasicBlock(
                in_channels = 64, out_channels = 64, 
                stride = 1, downsample = False
            )
        )   
        self.layer2 = nn.Sequential(
            BasicBlock(
                in_channels = 64, out_channels = 128, 
                stride = 2, downsample = True
            ),
            BasicBlock(
                in_channels = 128, out_channels = 128, 
                stride = 1, downsample = False
            )
        )
        self.layer3 = nn.Sequential(
            BasicBlock(
                in_channels = 128, out_channels = 256, 
                stride = 2, downsample = True
            ),
            BasicBlock(
                in_channels = 256, out_channels = 256, 
                stride = 1, downsample = False
            )
        )
        self.layer4 = nn.Sequential(
            BasicBlock(
                in_channels = 256, out_channels = 512, 
                stride = 2, downsample = True
            ),
            BasicBlock(
                in_channels = 512, out_channels = 512, 
                stride = 1, downsample = False
            )
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size = (1, 1))
        self.fc = nn.Linear(in_features = 512, out_features = num_classes, bias = True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

if __name__ == '__main__':
    input_ch = 3
    model = models.resnet18(weights = None, num_classes = 4)
    model.conv1 = nn.Conv2d(input_ch, model.conv1.weight.shape[0], 3, 1, 1, bias = False)
    model.maxpool = nn.MaxPool2d(kernel_size = 1, stride = 1, padding = 0)
    print(model.modules)

    model2 = CustomResNet(in_channels = 3, num_classes = 4)
    print(model2.modules)

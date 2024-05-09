from models import CustomResNet

model = CustomResNet(in_channels = 2, num_classes = 4)

print(model.modules)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from collections import OrderedDict

"""
We provide the models, which might be used in the experiments on FedD3, as follows:
    - AlexNet model customized for CIFAR-10 (AlexCifarNet) with 1756426 parameters
    - LeNet model customized for MNIST with 61706 parameters
    - Further ResNet models
    - Further Vgg models
"""


# AlexNet model customized for CIFAR-10 with 1756426 parameters
class AlexCifarNet(nn.Module):
    supported_dims = {32}

    def __init__(self):
        super(AlexCifarNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(4096, 384),
            nn.ReLU(inplace=True),
            nn.Linear(384, 192),
            nn.ReLU(inplace=True),
            nn.Linear(192, 10),
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), 4096)
        out = self.classifier(out)
        return out


# LeNet model customized for MNIST with 61706 parameters
class LeNet(nn.Module):
    supported_dims = {28}

    def __init__(self, num_classes=10, in_channels=1):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x), inplace=True)  # 6 x 28 x 28
        out = F.max_pool2d(out, 2)  # 6 x 14 x 14
        out = F.relu(self.conv2(out), inplace=True)  # 16 x 7 x 7
        out = F.max_pool2d(out, 2)  # 16 x 5 x 5
        out = out.view(out.size(0), -1)  # 16 x 5 x 5
        out = F.relu(self.fc1(out), inplace=True)
        out = F.relu(self.fc2(out), inplace=True)
        out = self.fc3(out)

        return out


# Further ResNet models
def generate_resnet(num_classes=10, in_channels=1, model_name="ResNet18"):
    if model_name == "ResNet18":
        model = models.resnet18(pretrained=True)
    elif model_name == "ResNet34":
        model = models.resnet34(pretrained=True)
    elif model_name == "ResNet50":
        model = models.resnet50(pretrained=True)
    elif model_name == "ResNet101":
        model = models.resnet101(pretrained=True)
    elif model_name == "ResNet152":
        model = models.resnet152(pretrained=True)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, num_classes)

    return model


# Further Vgg models
def generate_vgg(num_classes=10, in_channels=1, model_name="vgg11"):
    if model_name == "VGG11":
        model = models.vgg11(pretrained=False)
    elif model_name == "VGG11_bn":
        model = models.vgg11_bn(pretrained=True)
    elif model_name == "VGG13":
        model = models.vgg11(pretrained=False)
    elif model_name == "VGG13_bn":
        model = models.vgg11_bn(pretrained=True)
    elif model_name == "VGG16":
        model = models.vgg11(pretrained=False)
    elif model_name == "VGG16_bn":
        model = models.vgg11_bn(pretrained=True)
    elif model_name == "VGG19":
        model = models.vgg11(pretrained=False)
    elif model_name == "VGG19_bn":
        model = models.vgg11_bn(pretrained=True)

    # first_conv_layer = [nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)]
    # first_conv_layer.extend(list(model.features))
    # model.features = nn.Sequential(*first_conv_layer)
    # model.conv1 = nn.Conv2d(num_classes, 64, 7, stride=2, padding=3, bias=False)

    fc_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(fc_features, num_classes)

    return model


# class CNN(nn.Module):
#     def __init__(self, num_classes=10, in_channels=1):
#         super(CNN, self).__init__()
#
#         self.fp_con1 = nn.Sequential(OrderedDict([
#             ('con0', nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1)),
#             ('relu0', nn.ReLU(inplace=True)),
#         ]))
#
#         self.ternary_con2 = nn.Sequential(OrderedDict([
#             # Conv Layer block 1
#             ('conv1', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False)),
#             ('norm1', nn.BatchNorm2d(64)),
#             ('relu1', nn.ReLU(inplace=True)),
#             ('pool1', nn.MaxPool2d(kernel_size=2, stride=2)),
#
#             # Conv Layer block 2
#             ('conv2', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False)),
#             ('norm2', nn.BatchNorm2d(128)),
#             ('relu2', nn.ReLU(inplace=True)),
#             ('conv3', nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False)),
#             ('norm3', nn.BatchNorm2d(128)),
#             ('relu3', nn.ReLU(inplace=True)),
#             ('pool2', nn.MaxPool2d(kernel_size=2, stride=2)),
#             # nn.Dropout2d(p=0.05),
#
#             # Conv Layer block 3
#             ('conv3', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, bias=False)),
#             ('norm3', nn.BatchNorm2d(256)),
#             ('relu3', nn.ReLU(inplace=True)),
#             ('conv4', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False)),
#             ('norm4', nn.BatchNorm2d(256)),
#             ('relu4', nn.ReLU(inplace=True)),
#             ('pool4', nn.MaxPool2d(kernel_size=2, stride=2)),
#         ]))
#         self.flatten = nn.Flatten()
#         # self.fp_fc = nn.Linear(4096, num_classes, bias=False)
#         self.fp_fc = nn.Linear(256*3*3, num_classes, bias=False)
#
#     def forward(self, x):
#         x = self.fp_con1(x)
#         x = self.ternary_con2(x)
#         x = self.flatten(x)
#         x = self.fp_fc(x)
#         output = F.log_softmax(x, dim=1)
#         return output

class CNN(nn.Module):
    def __init__(self, num_classes=10, in_channels=1):
        super(CNN, self).__init__()
        # 第一个卷积层:输入通道数为1,输出通道数为16,卷积核大小为3x3,步长为1,填充为1
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()  # ReLU激活函数
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化层,池化核大小为2x2,步长为2

        # 第二个卷积层:输入通道数为16,输出通道数为32,卷积核大小为3x3,步长为1,填充为1
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()  # ReLU激活函数
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化层,池化核大小为2x2,步长为2

        # 全连接层:输入特征向量大小为32*7*7(经过卷积和池化后的特征图大小),输出大小为num_classes
        self.fc = nn.Linear(32 * 7 * 7, num_classes)

    def forward(self, x):
        # 前向传播过程:依次经过卷积层、ReLU激活函数、最大池化层,最后通过全连接层得到输出
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # 将特征图展平成一维向量
        x = self.fc(x)
        return x

if __name__ == "__main__":
    # model_name_list = ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"]
    # for model_name in model_name_list:
    #     model = generate_resnet(num_classes=10, in_channels=1, model_name=model_name)
    #     model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    #     param_len = sum([np.prod(p.size()) for p in model_parameters])
    #     print('Number of model parameters of %s :' % model_name, ' %d ' % param_len)
    # model = models.resnet18(pretrained=False)
    # model_path = "../checkpoints/resnet18-5c106cde.pth"
    # state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    # model.load_state_dict(state_dict)

    # model = AlexCifarNet()
    # state_dict = model.state_dict()
    # list = []
    # for key in state_dict.keys():
    #     list.append(key)
    # print(list)
    model = CNN(10,1)
    for param_name, param in model.named_parameters():
        print(f"Parameter {param_name} has data type: {param.dtype}")
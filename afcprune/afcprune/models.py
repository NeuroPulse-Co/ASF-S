"""
models.py – Conv2Net and Conv6Net for CIFAR-10 (32×32, 10 classes).

Same architectures as cifar100/models.py. With 10 classes they have 183,434 and
815,018 parameters, matching the paper's 183K and 815K (Table 4).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2Net(nn.Module):
    # (conv, bn, next conv) in forward order; the last conv feeds `fc` after one 2× pool
    CONV_SEQ = [("conv1", "bn1", "conv2"), ("conv2", "bn2", None)]
    FC_AFTER_CONV = "fc"
    FC_LAYERS = ["fc"]

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1);  self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1); self.bn2 = nn.BatchNorm2d(64)
        self.pool  = nn.MaxPool2d(2)
        self.fc    = nn.Linear(64 * 16 * 16, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        return self.fc(torch.flatten(x, 1))


class Conv6Net(nn.Module):
    CONV_SEQ = [("conv1", "bn1", "conv2"), ("conv2", "bn2", "conv3"),
                ("conv3", "bn3", "conv4"), ("conv4", "bn4", "conv5"),
                ("conv5", "bn5", "conv6"), ("conv6", "bn6", None)]
    FC_AFTER_CONV = "fc1"
    FC_LAYERS = ["fc1", "fc2"]

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3,   32,  3, padding=1); self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,  32,  3, padding=1); self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32,  64,  3, padding=1); self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64,  64,  3, padding=1); self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64,  128, 3, padding=1); self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1); self.bn6 = nn.BatchNorm2d(128)
        self.pool  = nn.MaxPool2d(2)
        self.fc1   = nn.Linear(128 * 4 * 4, 256)
        self.drop  = nn.Dropout(0.5)
        self.fc2   = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))))
        x = self.pool(F.relu(self.bn4(self.conv4(F.relu(self.bn3(self.conv3(x)))))))
        x = self.pool(F.relu(self.bn6(self.conv6(F.relu(self.bn5(self.conv5(x)))))))
        x = self.drop(F.relu(self.fc1(torch.flatten(x, 1))))
        return self.fc2(x)


MODELS = {"conv2net": Conv2Net, "conv6net": Conv6Net}


def build(name, num_classes=10):
    return MODELS[name](num_classes)

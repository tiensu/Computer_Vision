import torch.nn as tn
import torch.nn.functional as tnf
import torch.utils.data as tud
import torch.utils.data.dataloader as tuddl
import torch.utils.data.dataset as tudds
import torch.autograd.variable as tav
import torchvision
import torchvision.transforms as tvt


class SiameseNetwork(tn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        self.cnn = tn.Sequential(
            tn.Conv2d(1, 96, kernel_size=11, stride=1),
            tn.ReLU(inplace=True),
            tn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
            tn.MaxPool2d(3, stride=2),
            tn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            tn.ReLU(inplace=True),
            tn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
            tn.MaxPool2d(3, stride=2),
            tn.Dropout(p=0.3),
            tn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            tn.ReLU(inplace=True),
            tn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            tn.ReLU(inplace=True),
            tn.MaxPool2d(3, stride=2),
            tn.Dropout(p=0.3)
        )

        self.fc = tn.Sequential(
            tn.Linear(30976, 1024),
            tn.ReLU(inplace=True),
            tn.Dropout(p=0.5),
            tn.Linear(1024, 128),
            tn.ReLU(inplace=True),
            tn.Linear(128, 2)
        )

    def forward_one(self, x):
        output = self.cnn(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2

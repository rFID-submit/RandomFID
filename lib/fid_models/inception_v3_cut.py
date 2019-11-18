import torch
from torch import nn
from torchvision.models import inception_v3


class InceptionV3Cut(nn.Module):
    def __init__(self, pretrained=False, num_outs=10):
        super(InceptionV3Cut, self).__init__()

        self.model = inception_v3()
        self.model.fc = nn.Linear(in_features=2048, out_features=num_outs)
        self.model.add_module('fc', self.model.fc)

        if pretrained:
            self.load_state_dict(torch.load(pretrained))

    def forward(self, x):
        out = self.model(x)
        return out

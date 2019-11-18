import torch
import torch.nn as nn
from fid_models.initialization import fit_bn_to_dataset
from fid_models.model_activations import Activations


class LeNet5(nn.Module):
    """
    Input - 1x32x32
    C1 - 6@28x28 (5x5 kernel)
    tanh
    S2 - 6@14x14 (2x2 kernel, stride 2) Subsampling
    C3 - 16@10x10 (5x5 kernel, complicated shit)
    tanh
    S4 - 16@5x5 (2x2 kernel, stride 2) Subsampling
    C5 - 120@1x1 (5x5 kernel)
    F6 - 84
    tanh
    F7 - 10 (Output)
    """
    def __init__(self):
        super(LeNet5, self).__init__()

        self.convnet = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=(5, 5)),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(6, 16, kernel_size=(5, 5)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(16, 120, kernel_size=(5, 5)),
            nn.BatchNorm2d(120),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.ReLU(),
            nn.Linear(84, 10),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, img):
        if img.shape[1] == 3:
            img = img[:, 0]
            img.unsqueeze_(1)

        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        return output


def make_random_lenet(dataset, steps_for_fit=50, random_init=None, pretrained=False,
                      activations=('fc.1'), no_bias=False):
    random_lenet = LeNet5().cuda()

    if pretrained:
        random_lenet.load_state_dict(torch.load('../pretrained/LeNet5.pt'))
    else:
        fit_bn_to_dataset(random_lenet, dataset, random_init, steps_for_fit, no_bias)

    for param in random_lenet.parameters():
        param.requires_grad = False

    if activations is not None:
        random_lenet = Activations(random_lenet, activations)

    return random_lenet.eval()

import torch.nn as nn


#Define a Fully Convolutional Network
class FCN(nn.Module):

    def __init__(self):

        super(FCN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(5, 5), stride=(1, 1), padding=2), # 64 x 32 x 32, not count the batchsize
            nn.ReLU(),
            nn.BatchNorm2d(64, affine=True),
            nn.MaxPool2d(kernel_size=2), # 64 x 16 x 16
            nn.Conv2d(64, 256, kernel_size=(5, 5), stride=(1, 1),  padding=2), # 256 x 16 x 16
            nn.ReLU(),
            nn.BatchNorm2d(256, affine=True),
            nn.MaxPool2d(kernel_size=2, stride=(1, 1)), # 256 x 15 x 15
            nn.Conv2d(256, 256, kernel_size=(5, 5), stride=(1, 1), padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(256, affine=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(256, 512, kernel_size=(5, 5), stride=(1, 1), padding=2), # 512 x 7 x 7
            nn.ReLU(),
            nn.BatchNorm2d(512, affine=True),
            nn.MaxPool2d(kernel_size=2), # 512 x 2 x 2
            nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=1), # 256 x 3 x 3
            nn.ReLU(),
            nn.BatchNorm2d(256, affine=True),
            nn.MaxPool2d(kernel_size=2), # 256 x 1 x 1
            nn.Conv2d(256, 3, kernel_size=(3, 3), stride=(1, 1), padding=1), # 3 x 1 x 1
            nn.ReLU(),
        )

    def forward(self, x):

        batch = x.shape[0]
        x = self.layer1(x)

        return x.view(batch,3)

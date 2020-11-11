'''
Main Program
'''

from __future__ import print_function, division
import torch
import torchvision
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt

from FCN import FCN
from train import train_model
from test import test_model
import numpy as np
plt.ion()   # interactive mode

# Try use GPU if possible
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 4
filepath = 'data' #the load filepath.

image_datasets = torchvision.datasets.ImageFolder(filepath,
                                            transform=transforms.Compose([
                                            transforms.Resize(32),
                                            transforms.ToTensor()])
                                            )
#splite the train and test dataset, use random_split
train_db, val_db = torch.utils.data.random_split(image_datasets, [int(len(image_datasets)*0.8), len(image_datasets)-int(len(image_datasets)*0.8)])
image_datasets = {'train':train_db, 'test':val_db}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x=='train'), num_workers=4) for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


if __name__ == '__main__':
    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))
    out = torchvision.utils.make_grid(inputs)
    imshow(out)
    print(classes)

    # Define the model
    model_ft = FCN()
    model_ft = model_ft.to(device)
    train_model(model_ft, torch.nn.CrossEntropyLoss(), torch.optim.SGD(params=model_ft.parameters(), lr=0.0001, momentum=0.9), num_epochs=2,dataloaders=dataloaders, device=device,dataset_sizes=dataset_sizes)
    test_model(model_ft, dataloaders, device, dataset_sizes)

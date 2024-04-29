import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import time
import numpy as np
import os

DTYPE = torch.float

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
DATA_PATH = os.getenv('DATA_PATH')
NUM_OUTPUTS = 10 if os.getenv('NUM_OUTPUTS') is None else int(os.getenv('NUM_OUTPUTS'))

loader_g = torch.Generator()
loader_g.manual_seed(2023)


# Define VGG block
def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
        in_channels = out_channels
    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*layers)

# Define the VGG11bn model
class VGG11bn(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG11bn, self).__init__()
        self.features = nn.Sequential(
            vgg_block(1, 3, 64),
            vgg_block(1, 64, 128),
            vgg_block(2, 128, 256),
            vgg_block(2, 256, 512),
            vgg_block(2, 512, 512),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Data loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465), 
        (0.2023, 0.1994, 0.2010))
])

criterion = nn.CrossEntropyLoss()


# Define Network
def load_model(device=None):
    return VGG11bn(num_classes=NUM_OUTPUTS)


def load_train_data(dataset):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=torch.utils.data.RandomSampler(dataset, generator=loader_g),
        drop_last=True
    )

def load_val_data(dataset):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=torch.utils.data.SequentialSampler(dataset),
        drop_last=True
    )

def load_client_data(node_id: int):
    """Load partition CIFAR10 data."""
    with open(DATA_PATH, 'rb') as file:
        # Load the data from the file
        trainsets, valsets, _ = pickle.load(file)

    return load_train_data(trainsets[node_id]), load_val_data(valsets[node_id])

def load_test_data():
    """Load test CIFAR10 data."""
    with open(DATA_PATH, 'rb') as file:
        # Load the data from the file
        _, _, testset = pickle.load(file)

    return load_val_data(testset)



def train(model, optimizer, trainloader, device, epoch):
    model.train()
    train_loss = 0.0
    num_processed_samples = 0
    start_time = time.time()

    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        batch_size = labels.shape[0]
        num_processed_samples += batch_size

    print(f'Train: train_loss={train_loss/len(trainloader):.6f}, samples/s={num_processed_samples / (time.time() - start_time):.3f}')


def test(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    loss = 0

    num_processed_samples = 0
    start_time = time.time()
    with torch.inference_mode():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            loss += criterion(outputs, labels)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            batch_size = labels.shape[0]
            num_processed_samples += batch_size

    test_acc = 100 * correct / total
    test_loss = float(loss / len(data_loader))

    print(f'Test: test_acc={test_acc:.3f}, test_loss={test_loss:.6f}, samples/s={num_processed_samples / (time.time() - start_time):.3f}')
    return test_loss, test_acc

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Centralized PyTorch training")
    print("Load data")
    trainloader, testloader = load_client_data(0)
    net = load_model().to(DEVICE)
    net.eval()
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.0)
    print("Start training")
    train(model=net, optimizer=optimizer, trainloader=trainloader, device=DEVICE, epoch=1)
    print("Evaluate model")
    loss, accuracy = test(model=net, data_loader=testloader, device=DEVICE)
    print("Loss: ", loss)
    print("Accuracy: ", accuracy)

if __name__ == "__main__":
    main()

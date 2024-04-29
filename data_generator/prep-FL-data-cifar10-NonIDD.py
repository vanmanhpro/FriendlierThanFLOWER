from datasets.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader, Subset, random_split
import torchvision
import numpy as np
import pickle
import torch

DATA_PATH='/tmp/data/cifar10'
NUM_CLIENTS = 8
DUMP_FILE_NAME = '/tmp/data/fed-data-NonIDD-8-10.pkl'

transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.4914, 0.4822, 0.4465), 
                (0.2023, 0.1994, 0.2010)),
            ])

cifar10_train = torchvision.datasets.CIFAR10(
    root=DATA_PATH,
    train=True,
    transform=transform,
    download=True
)

cifar10_test = torchvision.datasets.CIFAR10(
    root=DATA_PATH,
    train=False,
    transform=transform,
    download=True
)

def prep_FL_NonIDD_data():
    # Calculate the size of each partition
    total_size = len(cifar10_train)
    indices = list(range(total_size))

    num_classes = 0
    for index in indices:
        num_classes = max(int(cifar10_train[index][1]) + 1, num_classes)

    id_subset_of_class = [[] for i in range(num_classes)]

    for index in indices:
        category = int(cifar10_train[index][1])
        id_subset_of_class[category].append(index)
    
    id_subset_of_client = [[] for i in range(num_classes)]

    for i in range(NUM_CLIENTS):
        i0 = i
        i1 = (i + 1) % num_classes
        i2 = (i + 2) % num_classes
        i3 = (i + 3) % num_classes
        i4 = (i + 4) % num_classes
        s0, e0 = 0, int(len(id_subset_of_class[i0]) / 4)
        s1, e1 = int(len(id_subset_of_class[i1]) / 4), int(len(id_subset_of_class[i1]) / 4) * 2
        s2, e2 = int(len(id_subset_of_class[i2]) / 4) * 2, int(len(id_subset_of_class[i2]) / 4) * 3
        s3, e3 = int(len(id_subset_of_class[i3]) / 4) * 3, int(len(id_subset_of_class[i3]) / 4) * 4
        s4, e4 = int(len(id_subset_of_class[i4]) / 4) * 4, int(len(id_subset_of_class[i4]) / 4) * 5
        id_subset_of_client[i] = id_subset_of_class[i0][s0:e0] + \
                                id_subset_of_class[i1][s1:e1] + \
                                id_subset_of_class[i2][s2:e2] + \
                                id_subset_of_class[i3][s3:e3] + \
                                id_subset_of_class[i4][s4:e4]

    subsets = [Subset(cifar10_train, client_id)
                        for client_id in id_subset_of_client]

    # Create train/val for each partition and wrap it into DataLoader
    trainsets = []
    valsets = []
    for partition_id in range(NUM_CLIENTS):
        partition_train, partition_test = random_split(subsets[partition_id], [0.8, 0.2])
        
        trainsets.append(partition_train)
        valsets.append(partition_test)
    
    testset = cifar10_test
    return trainsets, valsets, testset

def dump_FL_data():
    with open(DUMP_FILE_NAME, 'wb') as file:
        # Use pickle.dump() to dump the data into the file
        pickle.dump(prep_FL_NonIDD_data(), file)


if __name__ == "__main__":
    # test_training()

    dump_FL_data()
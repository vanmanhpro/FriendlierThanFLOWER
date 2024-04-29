from datasets.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader, Subset, random_split
import torchvision
import numpy as np
import pickle

from spikingjelly.activation_based.model.tv_ref_classify import presets, transforms, utils
from torch.utils.data.dataloader import default_collate


DATA_PATH='/tmp/data/cifar100'
NUM_CLIENTS = 8
DUMP_FILE_NAME = '/tmp/data/fed-data-CIFAR100-IID.pkl'

transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.4914, 0.4822, 0.4465), 
                (0.2023, 0.1994, 0.2010)),
            ])

cifar100_train = torchvision.datasets.CIFAR100(
    root=DATA_PATH,
    train=True,
    transform=transform,
    download=True
)

cifar100_test = torchvision.datasets.CIFAR100(
    root=DATA_PATH,
    train=False,
    transform=transform,
    download=True
)

def prep_FL_data():
    # Calculate the size of each partition
    total_size = len(cifar100_train)
    partition_size = total_size // NUM_CLIENTS
    indices = list(range(total_size))

    np.random.shuffle(indices)

    subset_id_lists = [indices[i * partition_size:(i + 1) * partition_size] for i in range(NUM_CLIENTS)]

    subsets = [Subset(cifar100_train, subset_id_list)
                        for subset_id_list in subset_id_lists]

    # Create train/val for each partition and wrap it into DataLoader
    trainsets = []
    valsets = []
    for partition_id in range(NUM_CLIENTS):
        partition_train, partition_test = random_split(subsets[partition_id], [0.8, 0.2])
        
        trainsets.append(partition_train)
        valsets.append(partition_test)
    
    testset = cifar100_test
    return trainsets, valsets, testset

def dump_FL_data():
    with open(DUMP_FILE_NAME, 'wb') as file:
        # Use pickle.dump() to dump the data into the file
        pickle.dump(prep_FL_data(), file)

if __name__ == "__main__":
    dump_FL_data()
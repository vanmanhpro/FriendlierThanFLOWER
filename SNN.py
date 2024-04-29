"""PyTorch CIFAR-10 image classification.

The code is generally adapted from 'PyTorch: A 60 Minute Blitz'. Further
explanations are given in the official PyTorch tutorial:

https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""

# mypy: ignore-errors
# pylint: disable=W0223


from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.activation_based import surrogate, neuron, functional
from spikingjelly.activation_based.model import spiking_vgg
from spikingjelly.activation_based.model.tv_ref_classify import transforms, utils
from torch.utils.data.dataloader import default_collate
import torchvision
import random
import pickle
import time
import numpy as np
import os

DTYPE = torch.float

BATCH_SIZE = 64

# Temporal Dynamics
NUM_STEPS = int(os.getenv('NUM_STEPS'))
DATA_PATH = os.getenv('DATA_PATH')
NUM_OUTPUTS = 10 if os.getenv('NUM_OUTPUTS') is None else int(os.getenv('NUM_OUTPUTS'))

mixup_transforms = []
mixup_alpha=0.2
cutmix_alpha=1.0
if mixup_alpha > 0.0:
    if torch.__version__ >= torch.torch_version.TorchVersion('1.10.0'):
        pass
    else:
        # TODO implement a CrossEntropyLoss to support for probabilities for each class.
        raise NotImplementedError("CrossEntropyLoss in pytorch < 1.11.0 does not support for probabilities for each class."
                                    "Set mixup_alpha=0. to avoid such a problem or update your pytorch.")
    mixup_transforms.append(transforms.RandomMixup(NUM_OUTPUTS, p=1.0, alpha=mixup_alpha))
if cutmix_alpha > 0.0:
    mixup_transforms.append(transforms.RandomCutmix(NUM_OUTPUTS, p=1.0, alpha=cutmix_alpha))
if mixup_transforms:
    mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)
    collate_fn = lambda batch: mixupcutmix(*default_collate(batch))  # noqa: E731

loader_g = torch.Generator()
loader_g.manual_seed(2023)

# Define Network
def load_model(num_classes=NUM_OUTPUTS, device=None):
    net = spiking_vgg.__dict__['spiking_vgg11_bn'](pretrained=False, spiking_neuron=neuron.LIFNode,
                                                    surrogate_function=surrogate.ATan(), 
                                                    detach_reset=True, num_classes=num_classes)
    functional.set_step_mode(net, step_mode='m')
    return net

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def load_train_data(dataset):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=torch.utils.data.RandomSampler(dataset, generator=loader_g),
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        drop_last=True
    )

def load_val_data(dataset):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=torch.utils.data.SequentialSampler(dataset),
        num_workers=0,
        pin_memory=True,
        worker_init_fn=seed_worker,
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

def preprocess_train_sample(x: torch.Tensor):
    # Define how to process training samples before sending them to the model
    return x.unsqueeze(0).repeat(NUM_STEPS, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]

def preprocess_test_sample(x: torch.Tensor):
    # Define how to process test samples before sending them to the model
    return x.unsqueeze(0).repeat(NUM_STEPS, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]

def process_model_output(y: torch.Tensor):
    # Define how to handle y = model(x)
    return y.mean(0)

def cal_acc1_acc5(output, target):
    # Define how to calculate acc1 and acc5
    acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
    return acc1, acc5


def train(model, optimizer, trainloader, device, epoch, model_ema=None, scaler=None):
    model_ema_steps = 32
    lr_warmup_epochs = 0

    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")

    header = f"Epoch: [{epoch}]"

    for image, target in metric_logger.log_every(trainloader, -1, header):
        start_time = time.time()
        image, target = image.to(device), target.to(device)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            image = preprocess_train_sample(image)
            output = process_model_output(model(image))      # Pulse firing frequency
            targets = torch.argmax(target, dim=1)
            label_one_hot = F.one_hot(targets, NUM_OUTPUTS).float()
            loss = F.mse_loss(output, label_one_hot)  # The spike firing frequency of output layer neurons and the MSE of the real category

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:                   # If specified at runtime--disable-amp
            loss.backward()
            optimizer.step()
        functional.reset_net(model)
        

        if model_ema and i % model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < lr_warmup_epochs:
                # Reset ema buffer to maintain copy weights during warm-up
                model_ema.n_averaged.fill_(0)

        acc1, acc5 = cal_acc1_acc5(output, target)
        batch_size = target.shape[0]

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))

        del image, target

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    train_loss, train_acc1, train_acc5 = metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg
    # print(f'Train: train_acc1={train_acc1:.3f}, train_acc5={train_acc5:.3f}, train_loss={train_loss:.6f}, samples/s={metric_logger.meters["img/s"]}')

    return train_loss, train_acc1


def test(model, data_loader, device, log_suffix=""):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    start_time = time.time()
    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, -1, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            image = preprocess_test_sample(image)
            output = process_model_output(model(image))
            label_one_hot = F.one_hot(target, NUM_OUTPUTS).float()
            loss = F.mse_loss(output, label_one_hot)  # The spike firing frequency of output layer neurons and the MSE of the real category

            acc1, acc5 = cal_acc1_acc5(output, target)
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = target.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size
            functional.reset_net(model)

    # Collect statistics for all processes
    num_processed_samples = utils.reduce_across_processes(num_processed_samples)

    metric_logger.synchronize_between_processes()

    test_loss, test_acc1, test_acc5 = metric_logger.loss.global_avg, metric_logger.acc1.global_avg, metric_logger.acc5.global_avg
    print(f'Test: test_acc1={test_acc1:.3f}, test_acc5={test_acc5:.3f}, test_loss={test_loss:.6f}, samples/s={num_processed_samples / (time.time() - start_time):.3f}')
    return test_loss, test_acc1


def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Centralized PyTorch training")
    print("Load data")
    trainloader, testloader = load_client_data(0)
    net = load_model(NUM_OUTPUTS).to(DEVICE)
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

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pickle as pkl
import random

class clientDataset(Dataset):
    def __init__(self, dataset, idxs, n):
        self.dataset = dataset
        N = len(idxs) * n // 100
        idxs = np.random.choice(idxs, size=N)
        self.idxs = list(idxs)
        self.classes = dataset.classes
        self.targets = np.array(dataset.targets)[idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
    
def load_data(n=100, client_id=-1, data_dir='data', dataset_name="MNIST", n_clients=4, alpha=0.1, batch_size=64, res_size=224):

    # Load dataset
    if(dataset_name=="MNIST"):
        transform =transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize((res_size, res_size)),
            transforms.RandomCrop(res_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.MNIST(f"{data_dir}/dataset/MNIST", train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(f"{data_dir}/dataset/MNIST", train=False, download=True, transform=transform)
    elif(dataset_name=="CIFAR10"):
        transform =transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize((res_size, res_size)),
            transforms.RandomCrop(res_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(f"{data_dir}/dataset/CIFAR10", train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(f"{data_dir}/dataset/CIFAR10", train=False, download=True, transform=transform)
    elif(dataset_name=="CIFAR100"):
        transform =transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize((res_size, res_size)),
            transforms.RandomCrop(res_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR100(f"{data_dir}/dataset/CIFAR100", train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR100(f"{data_dir}/dataset/CIFAR100", train=False, download=True, transform=transform)
    else:
        raise Exception("Dataset not recognized")
    

    # Dataloaders for given client
    if(client_id > -1):
        with open(f'{data_dir}/{n_clients}/{alpha}/{dataset_name}/train/' +dataset_name+"_"+str(client_id)+".pkl", "rb") as f:
            train_ids = pkl.load(f).astype(np.int32)
        with open(f'{data_dir}/{n_clients}/{alpha}/{dataset_name}/test/'+dataset_name+"_"+str(client_id)+".pkl", "rb") as f:
            test_ids = pkl.load(f).astype(np.int32)
        # Sanity check
        train_deets, test_deets = np.unique(np.array(trainset.targets)[train_ids], return_counts=True), np.unique(np.array(testset.targets)[test_ids], return_counts=True)

        trainloader = DataLoader(clientDataset(trainset, train_ids, n), batch_size=batch_size, shuffle=True, drop_last=True)            
        testloader = DataLoader(clientDataset(testset, test_ids, n), batch_size=batch_size, shuffle=False, drop_last=True)

        # Sanity check
        print("Client: {c}".format(c=client_id))
        print("Train set details: \n\tClasses: {c} \n\tSamples: {s}".format(c=train_deets[0], s=train_deets[1]))
        print("Test set details: \n\tClasses: {c} \n\tSamples: {s}".format(c=test_deets[0], s=test_deets[1]))
        print("\nTrain set size: {}; Test set size: {} \n".format(len(trainloader.dataset), len(testloader.dataset)))
    return trainloader, testloader


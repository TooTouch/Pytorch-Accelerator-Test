import os
from torch.utils.data import DataLoader
from torchvision import datasets


def create_dataset(datadir: str, aug_name: str = 'default'):
    trainset = datasets.CIFAR100(
        root      = os.path.join(datadir,'CIFAR100'), 
        train     = True, 
        download  = True, 
        transform = __import__('datasets').__dict__[f'{aug_name}_augmentation']()
    )
    testset = datasets.CIFAR100(
        root      = os.path.join(datadir,'CIFAR100'), 
        train     = False, 
        download  = True, 
        transform = __import__('datasets').__dict__['test_augmentation']()
    )

    return trainset, testset


def create_dataloader(dataset, batch_size: int = 4, shuffle: bool = False):

    return DataLoader(
        dataset     = dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = 0
    )

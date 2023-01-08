from torchvision import transforms
from RandAugment import RandAugment

def weak_augmentation():
    transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomCrop(32, fill=128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    return transform

def strong_augmentation():
    transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomCrop(32, fill=128),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    transform.transforms.insert(0, RandAugment(n=3, m=9))

    return transform 


def default_augmentation():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    return transform 

def test_augmentation():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    return transform
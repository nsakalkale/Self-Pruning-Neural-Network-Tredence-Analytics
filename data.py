"""
data.py : CIFAR-10 loaders with reasonable augmentation.
"""

import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader


def get_loaders(data_dir: str = "./data", batch_size: int = 256, num_workers: int = 0):
    train_tf = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, padding=4),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    val_tf = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_ds = torchvision.datasets.CIFAR10(data_dir, train=True,  download=True, transform=train_tf)
    val_ds   = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=512,        shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader
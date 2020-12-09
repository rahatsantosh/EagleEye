import os.path as osp
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def custom_get_dataloaders(batch_size, n_workers, path=""):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    transforms = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]
    )
    train_dataset = torchvision.datasets.CIFAR10(
        train=True,
        download=True,
        transforms=transforms
    )

    test_dataset = torchvision.datasets.CIFAR10(
        train=False,
        download=True,
        transforms=transforms
    )

    dataloader_train = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=True,
    )
    dataloader_test = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True,
    )
    return dataloader_train, dataloader_test

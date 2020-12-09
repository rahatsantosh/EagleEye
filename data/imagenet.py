import os.path as osp
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def custom_get_dataloaders(opt):
    normalize = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(224),
                torchvision.transforms.RandomResizedCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                normalize
            ]
    )
    train_dataset = torchvision.datasets.CIFAR10(
        train=True,
        download=True,
        transform=transforms
    )

    test_dataset = torchvision.datasets.CIFAR10(
        train=False,
        download=True,
        transform=transforms
    )

    dataloader_train = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
    )
    dataloader_test = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
    )
    return dataloader_train, dataloader_test

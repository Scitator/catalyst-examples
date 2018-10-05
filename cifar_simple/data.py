import collections
import torch
import torchvision
from torchvision import transforms
from catalyst.dl.datasource import AbstractDataSource


class DataSource(AbstractDataSource):

    @staticmethod
    def prepare_transforms(*, mode, stage=None):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    @staticmethod
    def prepare_loaders(args, data_params, stage=None):

        loaders = collections.OrderedDict()

        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True,
            download=True,
            transform=DataSource.prepare_transforms(mode="train", stage=stage))
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size,
            shuffle=True, num_workers=args.workers)

        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False,
            download=True,
            transform=DataSource.prepare_transforms(mode="valid", stage=stage))
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size,
            shuffle=False, num_workers=args.workers)

        loaders["train"] = trainloader
        loaders["valid"] = testloader

        return loaders

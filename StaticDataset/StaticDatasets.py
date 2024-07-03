import torch
import torch.utils.data
import torchvision.datasets as datasets



class MNIST:
    """
    MNIST dataset featuring gray-scale 28x28 images of
    hand-written characters belonging to ten different classes.
    Dataset implemented with torchvision.datasets.MNIST.

    Parameters:
        batch_size (int) workers(int).
        is_gpu (bool): True if CUDA is enabled.
            Sets value of pin_memory in DataLoader.

    Attributes:
        train_transforms (torchvision.transforms): Composition of transforms
            including conversion to Tensor, repeating gray-scale image to
            three channel for consistent use with different architectures
            and normalization.
        test_transforms (torchvision.transforms): Composition of transforms
            including conversion to Tensor, repeating gray-scale image to
            three channel for consistent use with different architectures
            and normalization.
        trainset (torch.utils.data.TensorDataset): Training set wrapper.
        testset (torch.utils.data.TensorDataset): Validation set wrapper.
        train_loader (torch.utils.data.DataLoader): Training set loader with shuffling.
        test_loader (torch.utils.data.DataLoader): Validation set loader.
    """

    def __init__(self, batch_size, train_transforms, test_transforms, workers=0, is_gpu=True):
        self.num_classes = 10

        self.train_transforms = train_transforms
        self.test_transforms = test_transforms

        self.trainset, self.testset = self.get_dataset()
        self.train_loader, self.test_loader = self.get_dataset_loader(batch_size, workers, is_gpu)


    def get_dataset(self):
        """
        Uses torchvision.datasets.MNIST to load dataset.
        Downloads dataset if doesn't exist already.

        Returns:
             torch.utils.data.TensorDataset: trainset, testset
        """

        trainset = datasets.MNIST('data/', train=True, transform=self.train_transforms,
                                  target_transform=None, download=True)
        testset = datasets.MNIST('data/', train=False, transform=self.test_transforms,
                                target_transform=None, download=True)

        return trainset, testset

    def get_dataset_loader(self, batch_size, workers, is_gpu):
        """
        Defines the dataset loader for wrapped dataset

        Parameters:
            batch_size (int): Defines the batch size in data loader
            workers (int): Number of parallel threads to be used by data loader
            is_gpu (bool): True if CUDA is enabled so pin_memory is set to True

        Returns:
             torch.utils.data.DataLoader: train_loader, test_loader
        """

        train_loader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=is_gpu, sampler=None)

        test_loader = torch.utils.data.DataLoader(
            self.testset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=is_gpu)

        return train_loader, test_loader





class FashionMNIST:
    """
    FashionMNIST dataset featuring gray-scale 28x28 images of
    Zalando clothing items belonging to ten different classes.
    Dataset implemented with torchvision.datasets.FashionMNIST.

    Parameters:
        batch_size (int) workers(int).
        is_gpu (bool): True if CUDA is enabled.
            Sets value of pin_memory in DataLoader.

    Attributes:
        train_transforms (torchvision.transforms): Composition of transforms
            including conversion to Tensor, repeating gray-scale image to
            three channel for consistent use with different architectures
            and normalization.
        test_transforms (torchvision.transforms): Composition of transforms
            including conversion to Tensor, repeating gray-scale image to
            three channel for consistent use with different architectures
            and normalization.
        trainset (torch.utils.data.TensorDataset): Training set wrapper.
        testset (torch.utils.data.TensorDataset): Validation set wrapper.
        train_loader (torch.utils.data.DataLoader): Training set loader with shuffling.
        test_loader (torch.utils.data.DataLoader): Validation set loader.
    """

    def __init__(self, batch_size, train_transforms, test_transforms, workers=0, is_gpu=True):
        self.num_classes = 10

        self.train_transforms = train_transforms
        self.test_transforms = test_transforms

        self.trainset, self.testset = self.get_dataset()
        self.train_loader, self.test_loader = self.get_dataset_loader(batch_size, workers, is_gpu)

    def get_dataset(self):
        """
        Uses torchvision.datasets.FashionMNIST to load dataset.
        Downloads dataset if doesn't exist already.

        Returns:
             torch.utils.data.TensorDataset: trainset, testset
        """

        trainset = datasets.FashionMNIST('data/', train=True, transform=self.train_transforms,
                                         target_transform=None, download=True)
        testset = datasets.FashionMNIST('data/', train=False, transform=self.test_transforms,
                                       target_transform=None, download=True)

        return trainset, testset

    def get_dataset_loader(self, batch_size, workers, is_gpu):
        """
        Defines the dataset loader for wrapped dataset

        Parameters:
            batch_size (int): Defines the batch size in data loader
            workers (int): Number of parallel threads to be used by data loader
            is_gpu (bool): True if CUDA is enabled so pin_memory is set to True

        Returns:
             torch.utils.data.DataLoader: train_loader, test_loader
        """

        train_loader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=is_gpu, sampler=None)

        test_loader = torch.utils.data.DataLoader(
            self.testset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=is_gpu)

        return train_loader, test_loader






class CIFAR10:
    """
    CIFAR-10 dataset featuring tiny 32x32 color images of
    objects belonging to hundred different classes.
    Dataloader implemented with torchvision.datasets.CIFAR10.

    Parameters:
        batch_size (int) workers(int).
        is_gpu (bool): True if CUDA is enabled.
            Sets value of pin_memory in DataLoader.
    Attributes:
        train_transforms (torchvision.transforms): Composition of transforms
            including conversion to Tensor, horizontal flips, random
            translations of up to 10% in each direction and normalization.
        test_transforms (torchvision.transforms): Composition of transforms
            including conversion to Tensor and normalization.
        trainset (torch.utils.data.TensorDataset): Training set wrapper.
        testset (torch.utils.data.TensorDataset): Validation set wrapper.
        train_loader (torch.utils.data.DataLoader): Training set loader with shuffling
        test_loader (torch.utils.data.DataLoader): Validation set loader.
    """

    def __init__(self, batch_size, train_transforms, test_transforms, workers=0, is_gpu=True):
        self.num_classes = 10

        self.train_transforms = train_transforms
        self.test_transforms = test_transforms

        self.trainset, self.testset = self.get_dataset()
        self.train_loader, self.test_loader = self.get_dataset_loader(batch_size, workers, is_gpu)

    def get_dataset(self):
        """
        Uses torchvision.datasets.CIFAR10 to load dataset.
        Downloads dataset if doesn't exist already.
        Returns:
             torch.utils.data.TensorDataset: trainset, testset
        """

        trainset = datasets.CIFAR10('data', train=True, transform=self.train_transforms,
                                    target_transform=None, download=True)
        testset = datasets.CIFAR10('data', train=False, transform=self.test_transforms,
                                  target_transform=None, download=True)

        return trainset, testset

    def get_dataset_loader(self, batch_size, workers, is_gpu):
        """
        Defines the dataset loader for wrapped dataset
        Parameters:
            batch_size (int): Defines the batch size in data loader
            workers (int): Number of parallel threads to be used by data loader
            is_gpu (bool): True if CUDA is enabled so pin_memory is set to True
        Returns:
             torch.utils.data.TensorDataset: trainset, testset
        """

        train_loader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=is_gpu, sampler=None)

        test_loader = torch.utils.data.DataLoader(
            self.testset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=is_gpu)

        return train_loader, test_loader






class CIFAR100:
    """
    CIFAR-100 dataset featuring tiny 32x32 color images of
    objects belonging to hundred different classes.
    Dataloader implemented with torchvision.datasets.CIFAR100.

    Parameters:
        batch_size (int) workers(int).
        is_gpu (bool): True if CUDA is enabled.
            Sets value of pin_memory in DataLoader.
    Attributes:
        train_transforms (torchvision.transforms): Composition of transforms
            including conversion to Tensor, horizontal flips, random
            translations of up to 10% in each direction and normalization.
        test_transforms (torchvision.transforms): Composition of transforms
            including conversion to Tensor and normalization.
        trainset (torch.utils.data.TensorDataset): Training set wrapper.
        testset (torch.utils.data.TensorDataset): Validation set wrapper.
        train_loader (torch.utils.data.DataLoader): Training set loader with shuffling.
        test_loader (torch.utils.data.DataLoader): Validation set loader.
    """

    def __init__(self, batch_size, train_transforms, test_transforms, workers=0, is_gpu=True):
        self.num_classes = 10

        self.train_transforms = train_transforms
        self.test_transforms = test_transforms

        self.trainset, self.testset = self.get_dataset()
        self.train_loader, self.test_loader = self.get_dataset_loader(batch_size, workers, is_gpu)
    def get_dataset(self):
        """
        Uses torchvision.datasets.CIFAR100 to load dataset.
        Downloads dataset if doesn't exist already.
        Returns:
             torch.utils.data.TensorDataset: trainset, testset
        """

        trainset = datasets.CIFAR100('data', train=True, transform=self.train_transforms,
                                     target_transform=None, download=True)
        testset = datasets.CIFAR100('data', train=False, transform=self.test_transforms,
                                   target_transform=None, download=True)

        return trainset, testset

    def get_dataset_loader(self, batch_size, workers, is_gpu):
        """
        Defines the dataset loader for wrapped dataset
        Parameters:
            batch_size (int): Defines the batch size in data loader
            workers (int): Number of parallel threads to be used by data loader
            is_gpu (bool): True if CUDA is enabled so pin_memory is set to True
        Returns:
             torch.utils.data.TensorDataset: trainset, testset
        """

        train_loader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=is_gpu, sampler=None)

        test_loader = torch.utils.data.DataLoader(
            self.testset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=is_gpu)

        return train_loader, test_loader





import torch
import torchvision
import torchvision.transforms as transforms


class DataLoader:
    """
    Convenience class.

    Creates `Pytorch` `Dataloaders` and returns data samples from it using
    the iterator functionalities.
    """

    def __init__(self):
        self.data_fashion = None
        self.data_mnist = None
        self.test_mnist = None
        self.test_fashion = None
        self.data_fashion_loader = None
        self.data_mnist_loader = None
        self.test_fashion_loader = None
        self.test_mnist_loader = None

    def init_iterators(self, root, batch_size):
        self.data_fashion_loader, self.data_mnist_loader, \
        self.test_fashion_loader, self.test_mnist_loader = self.load_data(
            root, batch_size)
        self.data_mnist = iter(self.data_mnist_loader)
        self.test_mnist = iter(self.test_mnist_loader)
        self.data_fashion = iter(self.data_fashion_loader)
        self.test_fashion = iter(self.test_fashion_loader)

    @staticmethod
    def load_data(root, batch_size):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize([0.5], [0.5])])

        trainset_fashion = torchvision.datasets.FashionMNIST(
            root=root,
            train=True,
            download=True,
            transform=transform)
        trainloader_fashion = torch.utils.data.DataLoader(trainset_fashion,
                                                          batch_size=batch_size,
                                                          shuffle=True,
                                                          num_workers=0)

        testset_fashion = torchvision.datasets.FashionMNIST(root=root,
                                                            train=False,
                                                            download=True,
                                                            transform=transform)
        testload_fashion = torch.utils.data.DataLoader(testset_fashion,
                                                       batch_size=batch_size,
                                                       shuffle=False,
                                                       num_workers=0)
        # load now MNIST dataset
        trainset_mnist = torchvision.datasets.MNIST(root=root,
                                                    train=True,
                                                    download=True,
                                                    transform=transform)
        trainloader_mnist = torch.utils.data.DataLoader(trainset_mnist,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        num_workers=0)

        testset_mnist = torchvision.datasets.MNIST(root=root,
                                                   train=False,
                                                   download=True,
                                                   transform=transform)
        testload_mnist = torch.utils.data.DataLoader(testset_mnist,
                                                     batch_size=batch_size,
                                                     shuffle=False,
                                                     num_workers=0)
        return trainloader_fashion, trainloader_mnist, testload_fashion, testload_mnist

    def dataiter_mnist(self):
        """ MNIST training set iterator """
        return self.data_mnist.next()

    def dataiter_fashion(self):
        """ MNISTFashion training set iterator """
        return self.data_fashion.next()

    def testiter_mnist(self):
        """ MNIST test set iterator """
        return self.test_mnist.next()

    def testiter_fashion(self):
        """ MNISTFashion test set iterator """
        return self.test_fashion.next()
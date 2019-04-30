from typing import Any, Iterator

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

from collections import namedtuple

import numpy as np

from l2l.optimizees.optimizee import Optimizee
from .conv_net import ConvNet
from .mlp_net import MLPNet
from l2l.optimizers.crossentropy import distribution

MNISTOptimizeeParameters = namedtuple('MNISTOptimizeeParameters', ['n_hidden',
                                                                   'seed', 'use_small_mnist',
                                                                   'n_ensembles',
                                                                   'root',
                                                                   'batch_size'])

MnistFashionOptimizeeParameters = namedtuple('MnistFashionOptimizeeParameters',
                                             ['seed',
                                              'n_ensembles', 'root',
                                              'batch_size'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DataLoader:
    """
    Convenience class.

    Dataloaders in Pytorch cannot be pickled and throw a `NotImplementedError`.
    This class creates such Dataloaders but converts them to list-iterators.
    It is not an optimal way but circumvents the problem. A drawback
    with this approach is the increased memory usage and loss of features.
    Note also the loading procedure only works when the `num_workers`
    variables in `load_data` are set to `0`.
    """
    def __init__(self):
        self.data_fashion = None
        self.data_mnist = None
        self.test_mnist = None
        self.test_fashion = None

    def init_iterators(self, root, batch_size):
        data_fashion, data_mnist, test_fashion, test_mnist = self.load_data(
            root, batch_size)
        # here are the expensive operations
        self.data_mnist = iter([i for i in data_mnist])
        self.test_mnist = iter([i for i in test_mnist])
        self.data_fashion = iter([i for i in data_fashion])
        self.test_fashion = iter([i for i in test_fashion])

    def load_data(self, root, batch_size):
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
        """ MNIST training set list iterator """
        return next(self.data_mnist)

    def dataiter_fashion(self):
        """ MNISTFashion training set list iterator """
        return next(self.data_fashion)

    def testiter_mnist(self):
        """ MNIST test set list iterator """
        return next(self.test_mnist)

    def testiter_fashion(self):
        """ MNISTFashion test set list iterator """
        return next(self.test_fashion)


class MnistFashionOptimizee(Optimizee):
    def __init__(self, traj, parameters):
        super().__init__(traj)

        seed = parameters.seed
        seed = np.uint32(seed)
        self.random_state = np.random.RandomState(seed=seed)

        self.n_ensembles = parameters.n_ensembles
        self.batch_size = parameters.batch_size
        self.root = parameters.root

        self.conv_net = ConvNet().to(device)
        self.mlp_net = MLPNet().to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.data_loader = DataLoader()
        self.data_loader.init_iterators(self.root, self.batch_size)
        self.dataiter_fashion = self.data_loader.dataiter_fashion
        self.dataiter_mnist = self.data_loader.dataiter_mnist
        self.testiter_fashion = self.data_loader.testiter_fashion
        self.testiter_mnist = self.data_loader.testiter_mnist

        generation = traj.individual.generation
        if generation % 2 == 0:
            self.inputs, self.labels = self.dataiter_fashion()
        else:
            self.inputs, self.labels = self.dataiter_mnist()

        # create_individual can be called because __init__ is complete except
        # for traj initializtion
        indiv_dict = self.create_individual()
        for key, val in indiv_dict.items():
            traj.individual.f_add_parameter(key, val)
        traj.individual.f_add_parameter('seed', seed)

    def create_individual(self):
        # get weights, biases from networks and flatten them
        # convolutional network parameters ###
        conv_ensembles = []
        with torch.no_grad():
            # weights for layers, conv1, conv2, fc1
            conv1_weights = self.conv_net.state_dict()['conv1.weight'].view(-1).numpy()
            conv2_weights = self.conv_net.state_dict()['conv2.weight'].view(-1).numpy()
            fc1_weights = self.conv_net.state_dict()['fc1.weight'].view(-1).numpy()

            # bias
            conv1_bias = self.conv_net.state_dict()['conv1.bias'].numpy()
            conv2_bias = self.conv_net.state_dict()['conv2.bias'].numpy()
            fc1_bias = self.conv_net.state_dict()['fc1.bias'].numpy()

            # stack everything into a vector of
            # conv1_weights, conv1_bias, conv2_weights, conv2_bias,
            # fc1_weights, fc1_bias
            params = np.hstack((conv1_weights, conv1_bias, conv2_weights,
                                conv2_bias, fc1_weights, fc1_bias))

            conv_ensembles.append(params)
            for _ in range(self.n_ensembles - 1):
                conv_ensembles.append(
                    np.hstack((
                        self._he_init(self.conv_net.state_dict()['conv1.weight']),
                        np.random.normal(self.conv_net.state_dict()['conv1.bias'].view(-1).numpy()),
                        self._he_init(self.conv_net.state_dict()['conv2.weight']),
                        np.random.normal(self.conv_net.state_dict()['conv2.bias'].view(-1).numpy()),
                        self._he_init(self.conv_net.state_dict()['fc1.weight']),
                        np.random.normal(self.conv_net.state_dict()['fc1.bias'].view(-1).numpy()),
                    ))
                )
            # multilayer perceptron parameters ###
            mlp_ensembles = []
            lin1_weights = self.mlp_net.state_dict()['lin1.weight'].view(-1).numpy()
            lin2_weights = self.mlp_net.state_dict()['lin2.weight'].view(-1).numpy()
            lin3_weights = self.mlp_net.state_dict()['lin3.weight'].view(-1).numpy()

            # bias
            lin1_bias = self.mlp_net.state_dict()['lin1.bias'].numpy()
            lin2_bias = self.mlp_net.state_dict()['lin2.bias'].numpy()
            lin3_bias = self.mlp_net.state_dict()['lin3.bias'].numpy()

            # stack everything into a vector of
            # lin1_weights, lin1_bias, lin2_weights, lin2_bias,
            # lin3_weights, lin3_bias
            params = np.hstack((lin1_weights, lin1_bias, lin2_weights,
                                lin2_bias, lin3_weights, lin3_bias))

            mlp_ensembles.append(params)
            for _ in range(self.n_ensembles - 1):
                mlp_ensembles.append(
                    np.hstack((
                        self._he_init(self.mlp_net.state_dict()['lin1.weight']),
                        np.random.normal(self.mlp_net.state_dict()['lin1.bias'].view(-1).numpy()),
                        self._he_init(self.mlp_net.state_dict()['lin2.weight']),
                        np.random.normal(self.mlp_net.state_dict()['lin2.bias'].view(-1).numpy()),
                        self._he_init(self.mlp_net.state_dict()['lin3.weight']),
                        np.random.normal(self.mlp_net.state_dict()['lin3.bias'].view(-1).numpy()),
                    ))
                )
            return dict(conv_params=np.array(conv_ensembles),
                        mlp_params=np.array(mlp_ensembles),
                        targets=self.labels.numpy(),
                        input=self.inputs.squeeze().numpy())

    @staticmethod
    def _he_init(weights, gain=0):
        """
        He- or Kaiming- initialization as in He et al., "Delving deep into
        rectifiers: Surpassing human-level performance on ImageNet
        classification". Values are sampled from
        :math:`\\mathcal{N}(0, \\text{std})` where

        .. math::
        \text{std} = \\sqrt{\\frac{2}{(1 + a^2) \\times \text{fan\\_in}}}

        Note: Only for the case that the non-linearity of the network
            activation is `relu`

        :param weights, tensor
        :param gain, additional scaling factor, Default is 0
        :return: numpy nd array, random array of size `weights`
        """
        fan_in = torch.nn.init._calculate_correct_fan(weights, 'fan_in')
        stddev = np.sqrt(2. / fan_in * (1 + gain ** 2))
        return stddev * np.random.randn(weights.numel())

    def _create_individual_distribution(self, random_state, weights, epsilon=0):
        dist = distribution.Gaussian()
        dist.init_random_state(random_state)
        dist.fit(weights, epsilon)
        new_individuals = dist.sample(self.n_ensembles)
        return new_individuals

    def bounding_func(self, individual):
        """
        Bounds the individual within the required bounds via coordinate clipping
        """
        return individual

    def simulate(self, traj):
        """
        Returns the value of the function chosen during initialization

        :param ~l2l.utils.trajectory.Trajectory traj: Trajectory
        :return: a single element :obj:`tuple` containing the value of the
            chosen function
        """
        # configure_loggers(exactly_once=True)
        # logger configuration is here since this function is paralellised
        # taken care of by jube

        # set the new parameter for the network
        conv_params = np.mean(traj.individual.conv_params, axis=0)
        d = self._shape_parameter_to_conv_net(conv_params)
        self.conv_net.set_parameter(**d)
        mlp_params = np.mean(traj.individual.mlp_params, axis=0)
        d = self._shape_parameter_to_mlp_net(mlp_params)
        self.mlp_net.set_parameter(**d)
        generation = traj.individual.generation
        with torch.no_grad():
            if generation % 1000 != 0:
                inputs = self.inputs
                labels = self.labels
                if generation % 2 == 0:
                    self.inputs, self.labels = self.dataiter_fashion()
                else:
                    self.inputs, self.labels = self.dataiter_mnist()
            elif generation % 1000 == 0 and generation > 0:
                # randomly test from mnist or fashion
                tests = (self.testiter_mnist, self.testiter_fashion)
                rand = np.random.randint(2)
                inputs, labels = tests[rand]()
            outputs = self.conv_net(inputs)
            conv_loss = self.criterion(outputs, labels)
            outputs = self.mlp_net(inputs)
            mlp_loss = self.criterion(outputs, labels)
            conv_params = []
            mlp_params = []
            for c, m in zip(traj.individual.conv_params, traj.individual.mlp_params):
                d = self._shape_parameter_to_conv_net(c)
                self.conv_net.set_parameter(**d)
                conv_params.append(self.conv_net(inputs).numpy().T)
                d = self._shape_parameter_to_mlp_net(m)
                self.mlp_net.set_parameter(**d)
                mlp_params.append(self.mlp_net(inputs).numpy().T)
            out = {
                'conv_params': np.array(conv_params),
                'mlp_params': np.array(mlp_params),
                'conv_loss': float(conv_loss),
                'mlp_loss': float(mlp_loss),
            }
        return out

    def _shape_parameter_to_conv_net(self, params):
        # first we need the shapes of the network parameter to reshape later
        # the new parameter
        conv1_w_shape = self.conv_net.state_dict()['conv1.weight'].shape
        conv1_b_shape = self.conv_net.state_dict()['conv1.bias'].shape
        conv2_w_shape = self.conv_net.state_dict()['conv2.weight'].shape
        conv2_b_shape = self.conv_net.state_dict()['conv2.bias'].shape
        fc1_w_shape = self.conv_net.state_dict()['fc1.weight'].shape
        fc1_b_shape = self.conv_net.state_dict()['fc1.bias'].shape
        # get the lengths
        conv1_w_l = self.conv_net.state_dict()['conv1.weight'].nelement()
        conv1_b_l = conv1_w_l + self.conv_net.state_dict()['conv1.bias'].nelement()
        conv2_w_l = conv1_b_l + self.conv_net.state_dict()['conv2.weight'].nelement()
        conv2_b_l = conv2_w_l + self.conv_net.state_dict()['conv2.bias'].nelement()
        fc1_w_l = conv2_b_l + self.conv_net.state_dict()['fc1.weight'].nelement()
        fc1_b_l = fc1_w_l + self.conv_net.state_dict()['fc1.bias'].nelement()
        # create the parameter dict
        param_dict = {
            'conv1_w': params[:conv1_w_l].reshape(conv1_w_shape),
            'conv1_b': params[conv1_w_l:conv1_b_l].reshape(conv1_b_shape),
            'conv2_w': params[conv1_b_l:conv2_w_l].reshape(conv2_w_shape),
            'conv2_b': params[conv2_w_l:conv2_b_l].reshape(conv2_b_shape),
            'fc1_w': params[conv2_b_l:fc1_w_l].reshape(fc1_w_shape),
            'fc1_b': params[fc1_w_l:fc1_b_l].reshape(fc1_b_shape)
        }
        return param_dict

    def _shape_parameter_to_mlp_net(self, params):
        # first we need the shapes of the network parameter to reshape later
        # the new parameter
        lin1_w_shape = self.mlp_net.state_dict()['lin1.weight'].shape
        lin1_b_shape = self.mlp_net.state_dict()['lin1.bias'].shape
        lin2_w_shape = self.mlp_net.state_dict()['lin2.weight'].shape
        lin2_b_shape = self.mlp_net.state_dict()['lin2.bias'].shape
        lin3_w_shape = self.mlp_net.state_dict()['lin3.weight'].shape
        lin3_b_shape = self.mlp_net.state_dict()['lin3.bias'].shape
        # now get the lengths
        lin1_w_l = self.mlp_net.state_dict()['lin1.weight'].nelement()
        lin1_b_l = lin1_w_l + self.mlp_net.state_dict()['lin1.bias'].nelement()
        lin2_w_l = lin1_b_l + self.mlp_net.state_dict()['lin2.weight'].nelement()
        lin2_b_l = lin2_w_l + self.mlp_net.state_dict()['lin2.bias'].nelement()
        lin3_w_l = lin2_b_l + self.mlp_net.state_dict()['lin3.weight'].nelement()
        lin3_b_l = lin3_w_l + self.mlp_net.state_dict()['lin3.bias'].nelement()
        # create the parameter dict
        param_dict = {
            'lin1_w': params[:lin1_w_l].reshape(lin1_w_shape),
            'lin1_b': params[lin1_w_l:lin1_b_l].reshape(lin1_b_shape),
            'lin2_w': params[lin1_b_l:lin2_w_l].reshape(lin2_w_shape),
            'lin2_b': params[lin2_w_l:lin2_b_l].reshape(lin2_b_shape),
            'lin3_w': params[lin2_b_l:lin3_w_l].reshape(lin3_w_shape),
            'lin3_b': params[lin3_w_l:lin3_b_l].reshape(lin3_b_shape)
        }
        return param_dict
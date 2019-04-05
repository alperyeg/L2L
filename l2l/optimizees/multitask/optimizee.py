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
                                                                   'n_ensembles'])

MnistFashionOptimizeeParameters = namedtuple('MnistFashionOptimizeeParameters',
                                             ['seed',
                                              'n_ensembles'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MnistFashionOptimizee(Optimizee):
    def __init__(self, traj, parameters):
        super().__init__(traj)

        seed = parameters.seed
        seed = np.uint32(seed)
        self.random_state = np.random.RandomState(seed=seed)

        self.n_ensembles = parameters.n_ensembles

        self.conv_net = ConvNet().to(device)
        self.mlp_net = MLPNet().to(device)

        dataiter_fashion, dataiter_mnist, testload_fashion, testload_mnist = self.load_data()
        # TODO alternatively do the change randomly
        if traj.individual.generation % 2 == 0:
            self.inputs, self.labels = dataiter_fashion.next()
        else:
            self.inputs, self.labels = dataiter_mnist.next()

        # create_individual can be called because __init__ is complete except for traj initializtion
        indiv_dict = self.create_individual()
        for key, val in indiv_dict.items():
            traj.individual.f_add_parameter(key, val)
        traj.individual.f_add_parameter('seed', seed)

    def create_individual(self):
        ensembles = []
        # get weights for layers, conv1, conv2, fc1
        # and flatten them
        with torch.no_grad():
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
            weights = np.hstack((conv1_weights, conv1_bias, conv2_weights,
                                 conv2_bias, fc1_weights, fc1_bias))

            ensembles.append(weights)
            for _ in range(self.n_ensembles - 1):
                # ensembles.append(np.hstack((
                #     torch.nn.init.xavier_uniform_(
                #         self.conv_net.state_dict()['conv1.weight']).view(-1).numpy(),
                #     torch.nn.init.uniform_(
                #         self.conv_net.state_dict()['conv1.bias']).view(-1).numpy(),
                #     torch.nn.init.xavier_uniform_(
                #         self.conv_net.state_dict()['conv2.weight']).view(-1).numpy(),
                #     torch.nn.init.uniform_(
                #         self.conv_net.state_dict()['conv2.bias']).view(-1).numpy(),
                #     torch.nn.init.xavier_uniform_(
                #         self.conv_net.state_dict()['fc1.weight']).view(-1).numpy(),
                #     torch.nn.init.uniform_(
                #         self.conv_net.state_dict()['fc1.bias']).view(-1).numpy()
                # )))
                ensembles.append(np.random.uniform(-1, 1, len(weights)))
            ensembles = np.array(ensembles)
            return dict(shift=ensembles,
                        targets=self.labels.numpy(),
                        input=self.inputs.squeeze().numpy())

    def _create_individual_distribution(self, random_state, weights, epsilon=0):
        dist = distribution.Gaussian()
        dist.init_random_state(random_state)
        dist.fit(weights, epsilon)
        new_individuals = dist.sample(self.n_ensembles)
        return new_individuals

    @staticmethod
    def load_data():
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize([0.5], [0.5])])

        # TODO make root as an argument within `parameters`, batch_size too
        root = '/home/yegenoglu/Documents/toolbox/L2L/l2l/optimizees/multitask'

        trainset_fashion = torchvision.datasets.FashionMNIST(root=root,
                                                             train=True,
                                                             download=True,
                                                             transform=transform)
        trainloader_fashion = torch.utils.data.DataLoader(trainset_fashion,
                                                          batch_size=1,
                                                          shuffle=True,
                                                          num_workers=2)

        testset_fashion = torchvision.datasets.FashionMNIST(root=root,
                                                            train=False,
                                                            download=True,
                                                            transform=transform)
        testload_fashion = torch.utils.data.DataLoader(testset_fashion,
                                                       batch_size=1,
                                                       shuffle=False,
                                                       num_workers=2)
        # load now MNIST dataset
        trainset_mnist = torchvision.datasets.MNIST(root=root, train=True,
                                                    download=True,
                                                    transform=transform)
        trainloader_mnist = torch.utils.data.DataLoader(trainset_mnist,
                                                        batch_size=1,
                                                        shuffle=True,
                                                        num_workers=2)

        testset_mnist = torchvision.datasets.MNIST(root=root, train=False,
                                                   download=True,
                                                   transform=transform)
        testload_mnist = torch.utils.data.DataLoader(testset_mnist,
                                                     batch_size=1,
                                                     shuffle=False,
                                                     num_workers=2)
        dataiter_fashion = iter(trainloader_fashion)
        dataiter_mnist = iter(trainloader_mnist)
        return dataiter_fashion, dataiter_mnist, testload_fashion, testload_mnist

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
        params = np.mean(traj.individual.shift, axis=0)
        d = self._shape_parameter_to_conv_net(params)
        self.conv_net.set_parameter(**d)
        with torch.no_grad():
            criterion = nn.CrossEntropyLoss()
            outputs = self.conv_net(self.inputs)
            loss = criterion(outputs, self.labels)
            all_outputs = []
            for s in traj.individual.shift:
                d = self._shape_parameter_to_conv_net(s)
                self.conv_net.set_parameter(**d)
                all_outputs.append(self.conv_net(self.inputs).numpy().T)
        return loss, np.array(all_outputs)

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
        lin3_b_shape = self.mlp_net.state_dict()['lin3.bias'].shape3
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


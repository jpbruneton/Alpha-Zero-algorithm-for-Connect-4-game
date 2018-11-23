#  ================ AlphaZero algorithm for Connect 4 game =================== #
# Name:             ResNet.py
# Description:      Includes both a dense and a resnet.
#                   Almost identical/taken from https://pytorch.org/docs/0.4.0/_modules/torchvision/models/resnet.html
# Authors:          Jean-Philippe Bruneton & Adèle Douin & Vincent Reverdy
# Date:             2018
# License:          BSD 3-Clause License
# ============================================================================ #

# ================================= PREAMBLE ================================= #
# Packages
import torch
import torch.utils
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data
import time
import math
from collections import OrderedDict
import torch.utils.data
import numpy as np
import config
import random

# ================================= CLASS : basic ResNet Block ================================= #

#no bias in conv
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        m = OrderedDict()
        m['conv1'] = conv3x3(inplanes, planes, stride)
        m['bn1'] = nn.BatchNorm2d(planes)
        m['relu1'] = nn.ReLU(inplace=True)
        m['conv2'] = conv3x3(planes, planes)
        m['bn2'] = nn.BatchNorm2d(planes)
        self.group1 = nn.Sequential(m)

        self.relu = nn.Sequential(nn.ReLU(inplace=True))
        self.downsample = downsample

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.group1(x) + residual
        out = self.relu(out)

        return out

# ================================= CLASS : ResNet + two heads ================================= #

class ResNet(nn.Module):
    def __init__(self, block, layers):

        self.input_dim = config.L*config.H
        self.output_dim = config.L
        self.inplanes = config.convsize
        self.convsize=config.convsize
        super(ResNet, self).__init__()

        torch.set_num_threads(1)

        #as a start : the three features are mapped into a conv with 4*4 kernel
        self.ksize = (4, 4)
        self.padding = (1, 1)
        m = OrderedDict()
        m['conv1'] = nn.Conv2d(3, self.convsize, kernel_size=self.ksize, stride=1, padding=self.padding, bias=False)
        m['bn1'] = nn.BatchNorm2d(self.convsize)
        m['relu1'] = nn.ReLU(inplace=True)

        self.group1= nn.Sequential(m)

        #next : entering the resnet tower
        self.layer1 = self._make_layer(block, self.convsize, layers[0])

        #next : entering the policy head
        pol_filters = config.polfilters
        self.policy_entrance = nn.Conv2d(self.convsize, config.polfilters, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnpolicy = nn.BatchNorm2d(config.polfilters)
        self.relu_pol = nn.ReLU(inplace=True)


        #if dense layer in policy head
        if config.usehiddenpol:
            self.hidden_dense_pol = nn.Linear(pol_filters * 30, config.hiddensize)
            self.relu_hidden_pol = nn.ReLU(inplace=True)
            self.fcpol1 = nn.Linear(config.hiddensize, 7)
        else:
            self.fcpol2= nn.Linear(pol_filters*30, 7)

        self.softmaxpol=nn.Softmax(dim=1)
        #end of policy head


        # in parallel: entering the value head
        val_filters = config.valfilters
        self.value_entrance = nn.Conv2d(self.convsize, config.valfilters, kernel_size=1, stride=1, padding=0, bias=False)
        self.bnvalue = nn.BatchNorm2d(config.valfilters)
        self.relu_val = nn.ReLU(inplace=True)

        #entering a dense hidden layer
        self.hidden_dense_value = nn.Linear(val_filters * 30, config.hiddensize)
        self.relu_hidden_val = nn.ReLU(inplace=True)
        self.fcval =  nn.Linear(config.hiddensize, 1)
        self.qval=nn.Tanh()
        #end value head

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / (5*n)))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        if type(x) == np.ndarray :
            x = x.reshape((3, config.H,config.L))
            x = torch.FloatTensor(x)
            x = torch.unsqueeze(x, 0)

        x = self.group1(x)
        x = self.layer1(x)

        x1 = self.policy_entrance(x)
        x1 = self.bnpolicy(x1)
        x1 = self.relu_pol(x1)
        x1 = x1.view(-1, config.polfilters*30)

        if config.usehiddenpol:
            x1 = self.hidden_dense_pol(x1)
            x1 = self.relu_hidden_pol(x1)
            x1 = self.fcpol1(x1)
        else:
            x1 = self.fcpol2(x1)

        x1 = self.softmaxpol(x1)

        x2 = self.value_entrance(x)
        x2 = self.bnvalue(x2)
        x2 = self.relu_val(x2)
        x2 = x2.view(-1, 30*config.valfilters)
        x2 = self.hidden_dense_value(x2)
        x2 = self.relu_hidden_val(x2)
        x2 = self.fcval(x2)
        x2 = self.qval(x2)

        return x2, x1

# -----------------------------------------------------------------#
# builds the model
def resnet18(pretrained=False, model_root=None, **kwargs):
    model = ResNet(BasicBlock, [config.res_tower, 2, 2, 2], **kwargs)
    return model


# ================================= CLASS : ResNet training ================================= #

class ResNet_Training:
    # -----------------------------------------------------------------#
    def __init__(self, net, batch_size, n_epoch, learning_rate, train_set, test_set, num_worker):
        self.net = net
        self.batch_size = batch_size
        self.n_epochs = n_epoch
        self.learning_rate = learning_rate
        self.num_worker = num_worker
        torch.set_num_threads(1)

        if config.use_cuda:
            self.net = self.net.cuda()

        self.train_set = train_set
        self.test_set = test_set

        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True
                                                        , num_workers=self.num_worker, drop_last=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=64, shuffle=True
                                                       , num_workers=self.num_worker)
        self.valid_loader = torch.utils.data.DataLoader(self.train_set, batch_size=128, shuffle=True
                                                        , num_workers=self.num_worker)
        self.net.train()

    # -----------------------------------------------------------------#
    # Losses
    def Loss_value(self):
        loss = torch.nn.MSELoss()
        return loss

    def Loss_policy_bce(self):
        loss = torch.nn.BCELoss()
        return loss

    # -----------------------------------------------------------------#
    # Optimizers
    def Optimizer(self):

        if config.optim == 'sgd':
            optimizer = optim.SGD(self.net.parameters(), lr=self.learning_rate, momentum=config.momentum,
                                  weight_decay=config.wdecay)
        elif config.optim == 'adam':
            optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate, weight_decay=config.wdecay)

        elif config.optim == 'rms':
            optimizer = optim.RMSprop(self.net.parameters(), lr=self.learning_rate, momentum=config.momentum,
                                      weight_decay=config.wdecay)

        return optimizer

    # -----------------------------------------------------------------#
    # training function

    def trainNet(self):

        n_batches = len(self.train_loader)
        print(n_batches, 'batches')
        optimizer = self.Optimizer()

        # Loop for n_epochs
        for epoch in range(self.n_epochs):

            running_loss = 0.0
            print_every = n_batches // 2
            start_time = time.time()
            total_train_loss = 0

            for i, data in enumerate(self.train_loader, 0):

                sboard = config.L * config.H
                preinputs = data[:, 0:3*sboard]
                inputs = preinputs.view(self.batch_size, 3, config.H, config.L)
                probas = data[:, 3*sboard:3*sboard + self.net.output_dim]

                reward = data[:, -1]
                probas = probas.float()
                reward = reward.float()
                reward = reward.view(self.batch_size, 1)

                if config.use_cuda:
                    inputs, probas, reward = inputs.cuda(), probas.cuda(), reward.cuda()

                inputs = Variable(inputs.float())
                probas, reward = Variable(probas), Variable(reward)

                # Set the parameter gradients to zero
                optimizer.zero_grad()

                # Forward pass, backward pass, optimize
                vh, ph = self.net(inputs)
                loss = 0
                loss += self.Loss_value()(vh, reward)
                loss += self.Loss_policy_bce()(ph, probas)

                loss.backward()
                optimizer.step()

                # Print statistics
                running_loss += loss.data.item()
                total_train_loss += loss.data.item()

                if (i + 1) % (print_every + 1) == 0:
                    print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                        epoch+1, int(100 * (i+1) / n_batches), running_loss / print_every, time.time() - start_time))

        #    #Reset running loss and time
                    running_loss = 0.0
                    start_time = time.time()

        # send back the model to cpu for next self play games using forward in parallel cpu
        self.net.cpu()

# ================================= CLASS : DenseNet ================================= #

class DenseNet(nn.Module):
    # -----------------------------------------------------------------#
    def __init__(self):

        self.input_dim = 3*42
        torch.set_num_threads(1)
        self.hiddensize = 1024
        random.seed()

        super(DenseNet, self).__init__()

        self.first = nn.Linear(self.input_dim, self.hiddensize)
        self.second = nn.Linear(self.hiddensize, self.hiddensize)

        self.dense_policy = nn.Linear(self.hiddensize, 7)
        self.dense_value = nn.Linear(self.hiddensize, 1)

    # -----------------------------------------------------------------#
    def forward(self, x):
        if type(x) == np.ndarray:
            x = torch.FloatTensor(x)
            x = x.unsqueeze(0)
        else:
            x = x.squeeze(1)

        x = F.relu(self.first(x))
        x = F.relu(self.second(x))

        x1 = self.dense_policy(x)
        x1 = F.softmax(x1, dim=1)
        x2 = F.torch.tanh(self.dense_value(x))
        return x2, x1


# ================================= CLASS : DenseNet Training ================================= #

class DenseNet_Training:
    # -----------------------------------------------------------------#
    def __init__(self, net, batch_size, n_epoch, learning_rate, train_set, test_set, num_worker):
        self.net = net
        self.batch_size = batch_size
        self.n_epochs = n_epoch
        self.learning_rate = learning_rate
        self.num_worker = num_worker
        torch.set_num_threads(1)
        random.seed()
        self.optim = optim
        self.train_set = train_set
        self.test_set = test_set

        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True
                                                        , num_workers=self.num_worker, drop_last=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=64, shuffle=True
                                                       , num_workers=self.num_worker)
        self.valid_loader = torch.utils.data.DataLoader(self.train_set, batch_size=128, shuffle=True
                                                        , num_workers=self.num_worker)
        self.net.train()

    # -----------------------------------------------------------------#
    # Losses
    def Loss_value(self):
        loss = torch.nn.MSELoss()
        return loss

    def Loss_policy_bce(self):
        loss = torch.nn.BCELoss()
        return loss

    def Optimizer_adam(self):

        optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        return optimizer

    def Optimizer_sgd(self):
        optimizer = optim.SGD(self.net.parameters(), lr=self.learning_rate, momentum=config.momentum)

        return optimizer

    # -----------------------------------------------------------------#
    # Training
    def trainNet(self):

        n_batches = len(self.train_loader)
        print(n_batches, 'batches')

        for epoch in range(self.n_epochs):
            running_loss = 0.0
            print_every = n_batches // 2
            start_time = time.time()
            total_train_loss = 0

            for i, data in enumerate(self.train_loader, 0):

                sboard = config.L * config.H
                preinputs = data[:, 0:3*sboard]
                inputs = preinputs.view(self.batch_size, 3*config.H*config.L)
                probas = data[:, 3*sboard:3*sboard + 7]
                reward = data[:, -1]

                probas = probas.float()
                reward = reward.float()
                probas = probas.view(self.batch_size,7)
                reward = reward.view(self.batch_size, 1)
                inputs = Variable(inputs.float())

                # Set the parameter gradients to zero
                if config.optim == 'sgd':
                    self.Optimizer_sgd().zero_grad()
                elif config.optim == 'adam':
                    self.Optimizer_adam().zero_grad()

                # Forward pass, backward pass, optimize
                vh, ph = self.net(inputs)
                loss = 0
                loss += self.Loss_value()(vh, reward)
                loss += self.Loss_policy_bce()(ph, probas)
                loss.backward()

                if config.optim == 'sgd':
                    self.Optimizer_sgd().step()
                elif config.optim == 'adam':
                    self.Optimizer_adam().step()

                # Print statistics
                running_loss += loss.data.item()
                total_train_loss += loss.data.item()

                if (i + 1) % (print_every + 1) == 0:
                    print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                        epoch + 1, int(100 * (i + 1) / n_batches), running_loss / print_every,
                        time.time() - start_time))

                    #    #Reset running loss and time
                    running_loss = 0.0
                    start_time = time.time()



def densenet(pretrained=False, model_root=None, **kwargs):
    model = DenseNet(**kwargs)
    return model
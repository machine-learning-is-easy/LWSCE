# -*- coding: utf-8 -*-
import torch
import torchvision
import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from loss.partial_loss import MaskedCrossEntropyLoss, AdaptiveMaskedCrossEntropyLoss
from model_define.defined_model import KMNISTNet, CIFARNet
# from model_define.hugging_face_vit import ViTForImageClassification
import torchvision.models as models
import os
import pandas as pd
torch.manual_seed(1000)
torch.cuda.manual_seed(1000)
import random
random.seed(1000)

import argparse

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
parser = argparse.ArgumentParser(description='Train Model')
parser.add_argument('--epoch', '-e', dest='epoch', default=100, help='epoch')
parser.add_argument('--dataset', '-d', dest='dataset', default="CIFAR10", help='dataset', required=False)
parser.add_argument('--opt_alg', '-a', dest='opt_alg', default="SGD", help='opt_alg', required=False)
parser.add_argument('--lossfunction', '-l', dest='lossfunction', default="MASKEDLABEL", help='lossfunction', required=False)
parser.add_argument('--lr', '-lr', dest='lr', type=float, default=1e-4, help='learning rate')


args = parser.parse_args()
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].

########################################################################
# .. note::
#     If running on Windows and you get a BrokenPipeError, try setting
#     the num_worker of torch.utils.data.DataLoader() to 0.


batch_size = 128

current_folder = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_folder, 'data')
if args.dataset == "CIFAR10":
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)
    num_channel = 3
    image_size = trainset.data.shape[1]
    dataclasses_num = len(trainset.classes)

    net = CIFARNet(num_class=dataclasses_num, num_channel=num_channel)
    net = net.to(device)


elif args.dataset == "CIFAR100":
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR100(root=data_path, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR100(root=data_path, train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)
    num_channel = 3
    image_size = trainset.data.shape[1]
    dataclasses_num = len(trainset.classes)

    net = CIFARNet(num_class=dataclasses_num, num_channel=num_channel)
    # net = ViTForImageClassification(num_labels=dataclasses_num)
    net = net.to(device)

elif args.dataset == 'IMAGENET':

    trainset = torchvision.datasets.ImageNet(root=data_path, split="train")
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    testset = torchvision.datasets.EMNIST(root=data_path, split="mnist")
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)

elif args.dataset == "EMNIST":
    transform = transforms.Compose(
        [transforms.ToTensor(), torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))])

    trainset = torchvision.datasets.EMNIST(root=data_path, train=True, split="mnist",
                                             download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    testset = torchvision.datasets.EMNIST(root=data_path, train=False, split="mnist",
                                             download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)

    # define model
    num_channel = 1
    image_size = trainset.data.shape[1]
    dataclasses_num = len(trainset.classes)

    net = KMNISTNet(num_class=dataclasses_num, num_channel=num_channel)
    net = net.to(device)

elif args.dataset == "FashionMNIST":
    transform = transforms.Compose(
        [transforms.ToTensor(), torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))])
    trainset = torchvision.datasets.FashionMNIST(root=data_path, train=True,
                                           download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    testset = torchvision.datasets.FashionMNIST(root=data_path, train=False,
                                          download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)

    # define model
    num_channel = 1
    image_size = trainset.data.shape[1]
    dataclasses_num = len(trainset.classes)

    net = KMNISTNet(num_class=dataclasses_num, num_channel=num_channel)
    net = net.to(device)

elif args.dataset == "MNIST":
    transform = transforms.Compose(
        [transforms.ToTensor(), torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))])
    trainset = torchvision.datasets.MNIST(root=data_path, train=True,
                                           download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    testset = torchvision.datasets.FashionMNIST(root=data_path, train=False,
                                          download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)

    # define model
    num_channel = 1
    image_size = trainset.data.shape[1]
    dataclasses_num = len(trainset.classes)

    net = KMNISTNet(num_class=dataclasses_num, num_channel=num_channel)
    net = net.to(device)

elif args.dataset == "KMNIST":
    transform = transforms.Compose(
        [transforms.ToTensor(), torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))])
    trainset = torchvision.datasets.KMNIST(root=data_path, train=True,
                                           download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    testset = torchvision.datasets.KMNIST(root=data_path, train=False,
                                          download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)

    # define model
    num_channel = 1
    image_size = trainset.data.shape[1]
    dataclasses_num = len(trainset.classes)

    net = KMNISTNet(num_class=dataclasses_num, num_channel=num_channel)
    net = net.to(device)

elif args.dataset == "QMNIST":
    transform = transforms.Compose(
        [transforms.ToTensor(), torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))])
    trainset = torchvision.datasets.QMNIST(root=data_path, train=True,
                                           download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    testset = torchvision.datasets.QMNIST(root=data_path, train=False,
                                          download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)

    # define model
    num_channel = 1
    image_size = trainset.data.shape[1]
    dataclasses_num = len(trainset.classes)
    net = KMNISTNet(num_class=dataclasses_num, num_channel=num_channel)
    net = net.to(device)

else:
    raise Exception("Unable to support the data {}".format(args.dataset))

if args.lossfunction == "LWSCE":
    criterion = MaskedCrossEntropyLoss(alpha=0.2, num_class=dataclasses_num, device=device)
elif args.lossfunction == 'CROSSENTROPY':
    criterion = nn.CrossEntropyLoss()
elif args.lossfunction == 'LABELSMOOTHING':
    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
elif args.lossfunction == "ALWSCE":
    criterion = AdaptiveMaskedCrossEntropyLoss(alpha=0.2, num_class=dataclasses_num, device=device)
else:
    raise Exception("Unaccept loss function {}".format(args.lossfunction))


########################################################################
# 2. Define a Convolutional Neural Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Copy the neural network from the Neural Networks section before and modify it to
# take 3-channel images (instead of 1-channel images as it was defined).


# from model_define.hugging_face_vit import ViTForImageClassification

def defineopt(model):
    if args.opt_alg == "ADAM":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt_alg == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    elif args.opt_alg == "RADAM":
        optimizer = optim.RAdam(model.parameters(), lr=args.lr)
    elif args.opt_alg == "ADADELTA":
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    elif args.opt_alg == "LWADAM":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    else:
        raise Exception("Not accept optimizer of {}".args.opt_alg)
    return optimizer
optimizer = defineopt(net)
########################################################################
def run_test(model_path):
    correct = 0
    total = 0
    net.load_state_dict(torch.load(model_path))
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            images = images.to(device)
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            labels = labels.to(device)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

def L2_reg(parameters):
    L2 = 0
    for p in parameters:
        L2 += torch.sum(torch.square(p))
    return torch.round(L2).item()

def save_model(net, model_path):
    torch.save(net.state_dict(), model_path)

def reinitialization_model(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
# 4. Train the network
for t in range(10): # train model 10 times
    acc = []
    for epoch in range(int(args.epoch)):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # forward + backward + optimize
            inputs = inputs.to(device)
            labels = labels.to(device)
            try:
                outputs = net(inputs)
            except Exception as ex:
                outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # zero the parameter gradients
            optimizer.zero_grad()

            running_loss += loss.item()
            # print("{} step loss is {}".format(i, loss.item()))
        model_path = os.path.join(current_folder, 'model', '{}_{}_{}_net.pth'.format(args.dataset, args.opt_alg, args.lossfunction))
        save_model(net, model_path)
        acc_epoch = run_test(model_path)
        acc_epoch = round(acc_epoch, 2)
        L2 = L2_reg(net.parameters())
        acc.append([epoch, acc_epoch, round(running_loss, 2), L2])
        print("{} epoch acc is {}, L2 is {}".format(epoch, acc_epoch, L2))
    print('Finished Training')
    result_file = os.path.join(os.path.join(current_folder, 'result', 'result_direct_{}_{}_{}'.format(args.dataset, args.opt_alg, args.lossfunction), "{}.csv".format(str(t))))
    if not os.path.exists(os.path.dirname(result_file)):
        os.makedirs(os.path.dirname(result_file))
    pd.DataFrame(acc).to_csv(result_file, header=["epoch", "training_acc", "training_loss", "L2"], index=False)
    del net
    del optimizer
    if 'MNIST' in args.dataset:
        net = KMNISTNet(num_class=dataclasses_num, num_channel=num_channel)
        net = net.to(device)
        optimizer = defineopt(net)
    else:
        net = CIFARNet(num_class=dataclasses_num, num_channel=num_channel)
        net = net.to(device)
        optimizer = defineopt(net)

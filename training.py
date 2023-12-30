# -*- coding: utf-8 -*-
import torch
import torchvision
import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
from torch.utils.data import random_split
import torch.optim as optim
import torch.nn as nn

from loss.lwsce import LabelWiseSignificanceCrossEntropy

from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.models as models
import os
from model_define.googlenet import googlenet
from model_define.resnet import resnet18
from model_define.densenet import densenet121
from model_define.mobilenet import mobilenet
from model_define.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from model_define.wideresidual import wideresnet
import pandas as pd
torch.manual_seed(1000)
torch.cuda.manual_seed(1000)
import random
random.seed(1000)
import argparse

TINY_IMAGENET_DATASET = 'Maysee/tiny-imagenet'
# IMAGENET1k_DATASET = 'imagenet-1k'
POKEMON_DATASET = "keremberke/pokemon-classification"
OXFORD_FLOWER_DATASET = "nelorth/oxford-flowers"

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
parser = argparse.ArgumentParser(description='Train Model')
parser.add_argument('--epoch', '-e', dest='epoch', default=50, help='epoch', required=False)
parser.add_argument('--dataset', '-d', dest='dataset', default="CIFAR10", help='dataset', required=False)
parser.add_argument('--opt_alg', '-a', dest='opt_alg', default="ADAM", help='opt_alg', required=False)
parser.add_argument('--lossfunction', '-l', dest='lossfunction', default="LWSCE", help='LWSCE|CROSSENTROPY|LABELSMOOTHING', 
                    required=False)
parser.add_argument('--lr', '-lr', dest='lr', type=float, default=1e-3, help='learning rate', required=False)
parser.add_argument('--model', '-m', dest='model', default="googlenet", help='model', required="googlenet|resnet|vit")
parser.add_argument('--policy', '-p', dest='policy', default="NO", help="NO|ALL|ALC", required=False)


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

def reinitialization_model(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


def define_model(model_type, num_class, image_size):
    if model_type == "googlenet":
        net = googlenet(num_class)
    elif model_type == "vit":
        from vit.vit import ViTForImageClassification
        net = ViTForImageClassification(num_labels=num_class, image_size=image_size)
    elif model_type == "densenet":
        net = densenet121(num_class)
    elif model_type == 'mobilenet':
        net = mobilenet(class_num=num_class)
    elif model_type == 'vgg':
        net = vgg19_bn(num_class)
    elif model_type == 'wideresnet':
        net = wideresnet(num_class)
    elif model_type == 'resnet':
        net = resnet18(num_class)
    else:
        raise Exception("Unable to support model type of {}".model_type)

    return net

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

elif args.dataset == "CIFAR100":
    transform = transforms.Compose([transforms.ToTensor()])

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
else:
    raise Exception("Unable to support the data {}".format(args.dataset))

net = define_model(args.model, dataclasses_num, image_size=image_size)
net = net.to(device)

CROSSENTROPY_LOSS = nn.CrossEntropyLoss()
LABELSOOMTHING_LOSS = nn.CrossEntropyLoss(label_smoothing=0.2)
LWSCE_LOSS = LabelWiseSignificanceCrossEntropy(alpha=0.2, num_class=dataclasses_num, device=device, input_type="log")
if args.lossfunction == "LWSCE":
    criterion = LWSCE_LOSS
elif args.lossfunction == 'CROSSENTROPY':
    criterion = CROSSENTROPY_LOSS
elif args.lossfunction == 'LABELSMOOTHING':
    criterion = LABELSOOMTHING_LOSS
else:
    raise Exception("Unaccept loss function {}".format(args.lossfunction))
def get_loss(outputs, labels, step):
    if args.policy == "NO":
        return criterion(outputs, labels)
    elif args.policy == "ALC":
        # alternative LWSCE and CrossEntropy
        if step % 2 == 0:
            return LWSCE_LOSS(outputs, labels)
        else:
            return CROSSENTROPY_LOSS(outputs, labels)
    elif args.policy == "ALL":
        # alternative LWSCE and LabelSmoothing
        if step % 2 == 0:
            return LWSCE_LOSS(outputs, labels)
        else:
            return LABELSOOMTHING_LOSS(outputs, labels)
    elif args.policy == "LFL":
        # alternative LWSCE and LabelSmoothing
        if step <= 10:
            return LWSCE_LOSS(outputs, labels)
        else:
            return LABELSOOMTHING_LOSS(outputs, labels)

    elif args.policy == "LFC":
        # alternative LWSCE and LabelSmoothing
        if step <= 10:
            return LWSCE_LOSS(outputs, labels)
        else:
            return CROSSENTROPY_LOSS(outputs, labels)
    else:
        raise Exception("Unable to accept policy parameter {}".format(args.policy))


########################################################################
# 2. Define a Convolutional Neural Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Copy the neural network from the Neural Networks section before and modify it to
# take 3-channel images (instead of 1-channel images as it was defined).


# from model_define.hugging_face_vit import ViTForImageClassification

def defineopt(model):
    if args.opt_alg == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.opt_alg == "ADAM":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt_alg == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    elif args.opt_alg == "RADAM":
        optimizer = optim.RAdam(model.parameters(), lr=args.lr)
    elif args.opt_alg == "ADADELTA":
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    else:
        raise Exception("Not accept optimizer of {}".args.opt_alg)
    return optimizer
optimizer = defineopt(net)
def define_scheduler(optimizer):
    return ReduceLROnPlateau(optimizer, patience=10)
scheduler = define_scheduler(optimizer)
########################################################################
def run_test(net):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            # calculate outputs by running images through the network
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            predicted = torch.argmax(outputs.data, 1)
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

# 4. Train the network
for t in range(10):  # train model 10 times
    acc = []
    for epoch in range(1,int(args.epoch)):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]

            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            try:
                outputs = net(inputs)
            except Exception as ex:
                raise Exception("Inference model encounter Exceptions")

            loss = get_loss(outputs, labels, step=epoch)
            loss.backward()
            optimizer.step()
            # zero the parameter gradients
            optimizer.zero_grad()
            running_loss += loss.item()
            # print("{} step loss is {}".format(i, loss.item()))
        model_path = os.path.join(current_folder, 'model', '{}_{}_{}_net.pth'.format(args.dataset, args.opt_alg, args.lossfunction))
        # save_model(net, model_path)
        acc_epoch = run_test(net)
        scheduler.step(metrics=acc_epoch)
        acc_epoch = round(acc_epoch, 2)
        L2 = L2_reg(net.parameters())
        acc.append([epoch, acc_epoch, round(running_loss, 2), L2])
        print("{} epoch acc is {}, L2 is {}".format(epoch, acc_epoch, L2))
    print('Finished Training')
    if args.policy == "NO":
        folder_name = '{}_{}_{}_result'.format(args.dataset, args.model, args.lossfunction)
    else:
        folder_name = '{}_{}_{}_result'.format(args.dataset, args.model, args.policy)
    result_file = os.path.join(os.path.join(current_folder, 'result', folder_name, "{}.csv".format(str(t))))
    if not os.path.exists(os.path.dirname(result_file)):
        os.makedirs(os.path.dirname(result_file))
    pd.DataFrame(acc).to_csv(result_file, header=["epoch", "training_acc", "training_loss", "L2"], index=False)

    # reinitialize the model parameters and optimizer
    del net
    del optimizer
    net = define_model(args.model, dataclasses_num, image_size=image_size)
    net = net.to(device)
    optimizer = defineopt(net)
    scheduler = define_scheduler(optimizer)

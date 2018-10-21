'''
this script is for reproducing the result of Part3 in Project 1.

-------------------------------------------
INTRO:
You should write your codes or modify codes between the 
two '#####' lines. The codes between two '=======' lines 
are used for logging or printing, please do not change.

You need to debug and run your codes in somewhere else (e.g. Jupyter 
Notebook). This file is only used for your submission and marks. 

In this file, you do not have to split training set to training 
and validation or draw curves. The training and evaluation functions
are already written for successful logs, please do not change them. 
Please also do not copy the functions.

-------------------------------------------
USAGE:
In your final update, please rename the file as 'python1.py'.

>> python project1_part3_template.py
This will run the program on CPU, train for one epoch and test on trained nets.

>> python project1_part3_template.py --pretrained
This will run the program on CPU, train for one epoch and test using pretrained
network parameters 'baseline.pth' and 'modified.pth'.

>> python project1_part3_template.py --cuda
This will run the program on GPU. You can ignore this if you do not have GPU or
CUDA installed.

-------------------------------------------
NOTE:
this file might be incomplete, feel free to contact us
if you found any bugs or any stuff should be improved.
Thanks :)

Email: geng.zhan@sydney.edu.au, dwan3342@uni.sydney.edu.au
'''

# import the packages
import argparse
import logging
import sys
import time
import os
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# ==================================
# control input options. DO NOT CHANGE THIS PART.
def parse_args():
    parser = argparse.ArgumentParser(description= \
        'scipt for part 3 of project 1')
    parser.add_argument('--cuda', action='store_true', default=False,
        help='Used when there are cuda installed.')
    parser.add_argument('--output_path', default='./', type=str,
        help='The path that stores the log files.')
    parser.add_argument('--pretrained', action='store_true', default=False,
        help='When using this option, only run the test functions.')

    pargs = parser.parse_args()
    return pargs

# Creat logs. DO NOT CHANGE THIS PART.
def create_logger(final_output_path):
    log_file = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=os.path.join(final_output_path, log_file),
                        format=head)
    clogger = logging.getLogger()
    clogger.setLevel(logging.INFO)
    # add handler
    # print to stdout and log file
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    clogger.addHandler(ch)
    return clogger

# training process. The training process will only run one epoch 
# for the convenience of marking. DO NOT CHANGE THIS PART.
def train_net(net, trainloader, logging, criterion, optimizer, scheduler, epochs=1):
    net = net.train()
    for epoch in range(epochs):  # loop over the dataset multiple times, only 1 time by default
        if type(scheduler).__name__ != 'NoneType':
            scheduler.step()

        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            if args.cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if args.cuda:
                loss = loss.cpu()
            # print statistics
            running_loss += loss.item()
            
            if i % 2000 == 1999:    # print every 2000 mini-batches
                logging.info('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        
    # save network
    # torch.save(net.state_dict(), args.output_path + 'modified.pth')
    # write finish to the flie
    logging.info('Finished Training')
    
    
# evaluation process. DO NOT CHANGE THIS PART.
def eval_net(net, loader, logging, mode="baseline"):
    net = net.eval()
    if args.cuda:
        net = net.cuda()

    if args.pretrained:
        if args.cuda:
            net.load_state_dict(torch.load(args.output_path + mode + '.pth', map_location='cuda'))
        else:
            net.load_state_dict(torch.load(args.output_path + mode + '.pth', map_location='cpu'))

    correct = 0
    total = 0
    for data in loader:
        images, labels = data
        if args.cuda:
            images, labels = images.cuda(), labels.cuda()
        outputs = net(images)
        if args.cuda:
            outputs = outputs.cpu()
            labels = labels.cpu()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # print and write to log
    logging.info('=' * 55)
    logging.info('SUMMARY of '+ mode)
    logging.info('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    logging.info('=' * 55)

# Prepare for writing logs and setting GPU. 
# DO NOT CHANGE THIS PART.
args = parse_args()
if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)
# print('using args:\n', args)

logger = create_logger(args.output_path)
logger.info('using args:')
logger.info(args)

# DO NOT change codes above this line
# ==================================


####################################
# Transformation definition
# NOTE:
# Write the assigned transformation method here.
# Your modification of transformation should be performed on training
# set, i.e. train_transform. You can keep test_transform untouched or
# the same as the train_transform or using something else.

train_transform = transforms.Compose(
    [transforms.Resize(45),
     transforms.ToTensor(), 
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

test_transform =  transforms.Compose(
    [transforms.Resize(45),
     transforms.ToTensor()])

####################################

####################################
# Define training and test dataset. 
# You can make some modifications, e.g. batch_size, adding other hyperparameters

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

####################################

####################################
# Define your baseline network
# NOTE:
# define your new baseline network.
# use 3conv + 3pool + 3fc + 1out layers

class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(32 * 3 * 3, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 32 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

####################################

####################################
# Define your modified network
# NOTE:
# define your modified network that will generate the best results

####################################
class Modified(nn.Module):
    def __init__(self):
        super(Modified, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 10, 3)
        self.conv3 = nn.Conv2d(10, 16, 3)
        self.fc1 = nn.Linear(16 * 3 * 3, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 16 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
# ==================================
# use cuda if called with '--cuda'. DO NOT CHANGE THIS PART.
baseline = Baseline()
modified = Modified()
if args.cuda:
    baseline = baseline.cuda()
    modified = modified.cuda()
# ==================================

####################################
# Define Optimizer or Scheduler
# NOTE:
# define your optimizer or scheduler below.

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(modified.parameters(), lr=0.001, momentum=0.9)
scheduler = None
####################################

# ==================================
# finish training and test the network 
# and write to logs. DO NOT CHANGE THIS PART.
if __name__ == '__main__':
# train modified network
  
    train_net(modified, trainloader, logging, criterion, optimizer, scheduler)

    # test the baseline network and modified network
    eval_net(baseline, testloader, logging, mode="baseline")
    eval_net(modified, testloader, logging, mode="modified")
# ==================================




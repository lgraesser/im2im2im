from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
from torch.autograd import Variable

import tensorboard_logger

from tools import *
from common import get_data_loader
from vgg_model import *
#from focal_loss import FocalLoss
from sklearn.metrics import f1_score

parser = argparse.ArgumentParser(description='')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--base_path', type=str, default='.')
parser.add_argument('--exp_version', type=str, default='test')
parser.add_argument('--config', type=str, help="net configuration", default="../exps/unit/blond_brunette_smiling_eyeglass_clf.yaml")
args = parser.parse_args()

base_path = args.base_path
data_path = os.path.join(base_path, 'data')
exp_path = os.path.join(base_path, args.exp_version)
ckpt_path = os.path.join(exp_path, 'checkpoint')
log_path = os.path.join(exp_path, 'logs')
print("Experiment Name: {}".format(args.exp_version))
print(ckpt_path)
for path in [base_path, exp_path, data_path, base_path, ckpt_path, log_path]:
    if not os.path.exists(path):
        os.makedirs(path)

use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.set_device(args.gpu_id)

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
config = NetConfig(args.config)
trainloader = get_data_loader(config.datasets['traindata'], args.batch_size)
testloader = get_data_loader(config.datasets['valdata'], args.batch_size)
# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    #assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('%s/ckpt.t7'%ckpt_path)
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    net = VGG('VGG11', 4)
    #from torchvision.models.vgg import *
    #net = vgg11_bn(num_classes=4)
tensorboard_logger.configure(log_path)
if use_cuda:
    net.cuda()
    #net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
#criterion = FocalLoss(4, use_cuda)
optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=1e-6)

def train(epoch):
    global optimizer
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if batch_idx > 10:
           break
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        net.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
         
        optimizer.step()
        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        #progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #    %(train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        if batch_idx % 100 == 0:
            print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
            %(train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    tensorboard_logger.log_value('train_loss', train_loss/(batch_idx+1), epoch)
    tensorboard_logger.log_value('train_accuracy', 100.*correct/total, epoch)

def test(epoch):
    global best_acc
    f1 = 0
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if batch_idx > 10:
            break
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        f1 += f1_score(predicted.cpu().numpy(), targets.data.cpu().numpy(), average='micro')
    print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        #if not os.path.isdir('checkpoint'):
        #    os.mkdir('checkpoint')
        torch.save(state, '%s/ckpt.t7'%ckpt_path)
        best_acc = acc
    tensorboard_logger.log_value('test_loss', test_loss/(batch_idx+1), epoch)
    tensorboard_logger.log_value('test_accuracy', 100.*correct/total, epoch)
    print(predicted[:10].cpu().numpy())




for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)

    lr = args.lr*(0.9**int(epoch/10))
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-6)
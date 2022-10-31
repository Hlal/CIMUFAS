import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as torchmodels
import os
import time
import math
import numpy as np
from numpy import linalg as LA 
import models as mymodels
from quantization import *
import argparse
from config import Config

config_file = "config.yaml.txt"
config = Config(config_file)

arch = config.arch
dataset = config.dataset
data_path = config.data
load_model = config.load_path


lr = 0.001
momentum = 0.9
weight_decay=5e-4
bit_wt = config.bit_wt
bit_act = config.bit_act

modelname = '{}_{}'.format(arch, dataset)
model = mymodels.__dict__[modelname]()
model = torch.nn.DataParallel(model).cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)
if dataset == 'cifar10':
    transform_test = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)), ])
    testset = datasets.CIFAR10(root=data_path, train=False, download=False, transform=transform_test)
    # val_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)
    val_loader = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False, num_workers=8)
    # val_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=8)
elif dataset == 'mnist':
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    testset = datasets.MNIST(root=data_path, train=False, download=True, transform=transform_test)
    val_loader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=8)
    # val_loader = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False, num_workers=8)
    # val_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=8)
elif dataset== 'ImageNet' or dataset == 'imagenet':
    valdir = os.path.join(data_path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                                             transforms.ToTensor(), normalize, ])), batch_size=100,
            shuffle=False, num_workers=8,
            pin_memory=True)
checkpoint = torch.load(load_model)
if 'state_dict' in checkpoint:
    checkpoint = checkpoint['state_dict']
	
bits_activations = {}
bits_weights = {}
bits_bias = {}
input_bit = bit_act

for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        bits_activations[name] = bit_act
        bits_weights[name] = bit_wt
        bits_bias[name] = bit_wt
    else:
        bits_weights[name] = None
        bits_activations[name] = None
        bits_bias[name] = None
		
# quantizer = QuantAwareTrainConvLinearQuantizer(model, optimizer, bits_activations, bits_weights, bits_bias, num_bits_inputs=input_bit)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # print(correct[:k])
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

#Quantizer is the quantization function we use, which is used to modify the network structure to the structure required for quantization; 
# we have modified the quantized convolution and fully connected operations to simulate on-chip operations; the difference in quantization functions will not affect the simulation process. accuracy.
quantizer = QuantAwareTrainConvLinearQuantizer(model, optimizer, bits_activations, bits_weights, bits_bias, quantize_inputs=False, config = config)
quantizer.prepare_model()
model.load_state_dict(checkpoint)
model.eval()

criterion = nn.CrossEntropyLoss()
batch_time = AverageMeter()
losses = AverageMeter()
top1 = AverageMeter()
top5 = AverageMeter()


def runmodel():
    with torch.no_grad():
        end = time.time()
        alltime = time.time()
        print(alltime)
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if i % 100 == 0:
            if True:
                print('Test: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time, loss=losses,
                        top1=top1, top5=top5))
            # break
            # if( i == 0):
                # break
        print(time.time()-alltime)
        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
                .format(top1=top1, top5=top5))
        # print('*******************************************')
        # print(sigmaratio, '{top1.avg:.3f}'.format(top1=top1))
        # print('*******************************************')

runmodel()

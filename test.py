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

# dataset = 'mnist'
# dataset = 'cifar10'
# data_path = './data/mnist'
# data_path = './data'
arch = 'resnet18'
# # arch = 'lenets2'
# arch = 'lenet'
# arch = 'vgg8'
# load_model = './checkpoint_lenets2_mnist/exp1/quanted_real_pre0001_ckpt_b44_cosine_warmup.t7'
# load_model = './checkpoint_lenets_mnist/exp0/quanted_pruned_ckpt_b42_cosine_warmup.t7"'
# load_model = './checkpoint_lenet_mnist/exp1/quanted_b88_3_cosine_warmup.t7'
# load_model = './checkpoint_lenet_mnist/exp1/quanted_b44_cosine_warmup.t7'
# load_model = './checkpoint_lenet_mnist/exp1/one_third_pruned_b44_cosine_warmup.t7'
# load_model = './checkpoint_lenet_mnist/exp1/nopruned_b44_cosine_warmup.t7'
# load_model = './checkpoint_vgg8_cifar10/exp1/vgg8_b44_cosine_warmup.t7'


dataset = 'imagenet'
data_path = '/home/data/ImageNet/'


# load_model = './checkpoint_resnet18_cifar10/exp0/quanted_b44_cosine_warmup.t7'
load_model = './checkpoint_resnet18_imagenet/exp6/quanted_b44_cosine_warmup.t7'
# dataset = 'cifar10'
# data_path = './data/'#cifar10
# arch = 'reshalf18'
# load_model = './checkpoint_reshalf18_cifar10/exp0/quanted_b44_cosine_warmup.t7'

# parser = argparse.ArgumentParser(description='test')
# parser.add_argument('--sig', type=float, default=1, help='Random seed.')
# args = parser.parse_args()

lr = 0.001
momentum = 0.9
weight_decay=5e-4
bit_wt = 4
bit_act = 4
# bit_wt = 8
# bit_act = 8    

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
#flag = 5
flag = 0
#注释掉的是lenet的
# for name, module in model.named_modules():
#     flag = flag + 1
#     if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
#         if flag == 15:
#             bits_activations[name] = 1
#             bits_weights[name] = 1
#             bits_bias[name] = 1
#         else:
#             bits_activations[name] = bit_act
#             bits_weights[name] = bit_wt
#             bits_bias[name] = bit_wt
#     else:
#         bits_weights[name] = None
#         bits_activations[name] = None
#         bits_bias[name] = None
for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        #if flag > 0:
            #bits_activations[name] = 4
            #flag -= 1
        #else:
            #bits_activations[name] = bit_act
        # if flag == 0:
        #     bits_activations[name] = None
        #     flag += 1
        # else:
        #     bits_activations[name] = bit_act
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


quantizer = QuantAwareTrainConvLinearQuantizer(model, optimizer, bits_activations, bits_weights, bits_bias, quantize_inputs=False)
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
            # print('第{}组数据----------------------------------------------------------------------'.format(i))
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
            if( i == 0):
                break
        print(time.time()-alltime)
        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
                .format(top1=top1, top5=top5))
        # print('*******************************************')
        # print(sigmaratio, '{top1.avg:.3f}'.format(top1=top1))
        # print('*******************************************')

runmodel()

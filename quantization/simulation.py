from collections import OrderedDict, namedtuple
import torch
import torch.nn as nn
from torch.nn import functional as F
from .quantizer import Quantizer
from .q_utils import *
from .range_linear import *
from .clipped_linear import *
import logging
import numpy as np
import math
import heapq

from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack
import torch.distributions as diss
import pytorch_categorical

torch.set_printoptions(profile="full")
np.set_printoptions(threshold=np.inf)

msglogger = logging.getLogger()

def get_hn_r(loc = 2050, scale = 59.646):#high condu
        # print('begin')
        number = np.random.normal(loc = loc, scale = scale)
        return number

def get_ln_r(loc = 706.25, scale = 39.878):#low condu
    number = np.random.normal(loc = loc, scale = scale)
    return number

def file_check(file_name):
    temp_file_name = file_name
    i = 1
    while i:
        # print(temp_file_name)
        # print('cun zai?',os.path.exists(temp_file_name+'.npz'))
        if os.path.exists(temp_file_name+'.npz'):
            temp_file_name = file_name + str(i)
            i = i+1
        else:
            return temp_file_name

class ConvQuantizationWrapper(nn.Module):
    def __init__(self, wrapped_module, num_bits, init_act_clip_val, mode, ema_decay, config):
        super(ConvQuantizationWrapper, self).__init__()
        #self.conv_q = ConvClippedLinearQuantization(num_bits, init_act_clip_val, ema_decay, dequantize=True,
                                             #inplace=getattr(wrapped_module, 'inplace', False))
        self.conv_q = FakeLinearQuantization(num_bits, mode, ema_decay, dequantize=True,
                                             inplace=getattr(wrapped_module, 'inplace', False))
        self.wrapped_module = wrapped_module
        self.weight_scale = 0
        self.activ_scale = 0
        self.config = config
        # self.locarray = locarray
        # self.sigmaratio = sigmaratio
        
    def change_bit(self, bit):
        self.conv_q.num_bits = bit
        self.conv_q.iter_count = torch.zeros(1).cuda()
        self.conv_q.tracked_min_biased.data[0] = 0
        self.conv_q.tracked_max_biased.data[0] = 0
        self.conv_q.scale = torch.zeros(1).cuda()
        self.conv_q.zero_point = torch.zeros(1).cuda()
    

    def forward(self, *input):
        mode = self.config.mode
        # torch.set_printoptions(profile="full")
        # mode = "bit_slice"
        # mode = "bit_kernel"
        # mode = "sparate_pn_kernel"
        # mode = "4bit_rram"
        # mode = "varia_pn_kernel"
        # mode = "fast_2mod"
        # mode = "fast_4bit"
        # mode = "normal"
        #noise_type = "LSB*LSB"
        #RRAM
        # R_on = True
        # R_SAF = True
        R_on = self.config.R_on
        R_SAF = self.config.R_SAF
        ADC_error = self.config.ADC_error

        if mode == "normal":
            res = self.conv_q(*input)
            res = self.wrapped_module(res)
            return res
        elif mode == "bit_slice":
            torch.backends.cudnn.deterministic = True
            thre = 7 
            ad_limit = 15
            bit_weight = self.config.bit_wt
            bit_activ = self.config.bit_act
            # para = 18
            paranumber = self.config.para
            print('paranumber', paranumber)

            scale_w = self.wrapped_module.weight_scale
            scale_a = self.conv_q.scale
            scale_wa = scale_w * scale_a
            input_tmp = torch.round(input[0] * scale_a - self.conv_q.zero_point)
            input_tmp = torch.clamp(input_tmp, 0, (2**bit_activ)-1)

            weight_tmp = torch.round(self.wrapped_module.weight*scale_w)
            
            ce = self.wrapped_module.weight*scale_w
            wnz = (weight_tmp.view(-1) == 0).sum()*1.0/weight_tmp.numel()
            in_a = self.conv_q(*input)
            res = self.wrapped_module(in_a)
            res_sliced = torch.cuda.FloatTensor(res.shape).fill_(0) 
            res_sliced_tmp = torch.cuda.FloatTensor(res.shape).fill_(0)

            if hasattr(self.wrapped_module, 'kernel_size'):
                kernel_size = self.wrapped_module.kernel_size[0] * self.wrapped_module.kernel_size[1]
                para = paranumber // kernel_size
                paralevel = (self.wrapped_module.in_channels * kernel_size) // paranumber
                last_number = (self.wrapped_module.in_channels * kernel_size) % paranumber
                if(last_number != 0): paralevel += 1
                remain_number = 0
                begin_channel = 0
                end_channel = 0
                tmp_channel = 0
                for cib in range(0, paralevel):
                    res_sliced_tmp = torch.cuda.FloatTensor(res.shape).fill_(0)
                    arrayrow = min(paranumber, kernel_size * self.wrapped_module.in_channels - cib*paranumber)
                    for numcell in range(0, arrayrow):
                        number_all = cib * paranumber + numcell
                        channel = number_all // kernel_size
                        remain_number = number_all % kernel_size
                        krow = remain_number // self.wrapped_module.kernel_size[1]
                        kcol = remain_number % self.wrapped_module.kernel_size[0]
                        in_cis = input_tmp[:, channel:channel+1, :, :]
                        w_cis = weight_tmp[:, channel:channel+1, :, :]
                        m = nn.ZeroPad2d((self.wrapped_module.padding[0] - kcol, self.wrapped_module.padding[0]-(self.wrapped_module.kernel_size[1]-kcol-1), self.wrapped_module.padding[1] - krow, self.wrapped_module.padding[1]-(self.wrapped_module.kernel_size[0]-krow-1)))
                        a_pad = m(in_cis)

                        for ba in range(0, bit_activ//2):#��֧��2������������
                            for bw in range(0, bit_weight):
                                w_tmp = ((w_cis[:,:,krow:krow+1,kcol:kcol+1]+(1<<(bit_weight))).int() & (1<<bw)) >> bw
                                #ȡ��1����λ��weight,���ֱ�ȡ��1bit
                                w_tmp = w_tmp.float()
                                

                                a_tmp = (a_pad.int() & (3<<(ba*2))) >> (ba*2)#����λȡ�����ֵ

                                if(R_on):
                                    for i in range(0, w_tmp.size(0)):
                                        if(w_tmp[i,:,:,:] == 0): w_tmp[i,:,:,:] = get_ln_r(loc = self.config.rram_conductance[0], scale= self.config.rram_variation[0])
                                        if(w_tmp[i,:,:,:] == 1): w_tmp[i,:,:,:] = get_hn_r(loc = self.config.rram_conductance[1], scale= self.config.rram_variation[1])

                                res_tmp = F.conv2d(a_tmp.float(), w_tmp.float(), bias=None, stride=self.wrapped_module.stride, padding=0)

                                if bw == bit_weight-1:
                                    res_tmp = -res_tmp

                                res_sliced_tmp = res_sliced_tmp + (res_tmp<<(ba*2+bw)).float()#!!!!!!!
                    if(R_on):
                        res_sliced_tmp /= 2050
                        res_sliced_tmp = torch.round(res_sliced_tmp)

                    # exit(0)
                    res_sliced = res_sliced + res_sliced_tmp


                if self.conv_q.zero_point != 0:
                    in_tmp_b = torch.ones_like(in_a)*self.conv_q.zero_point
                    res_tmp_b = F.conv2d(in_tmp_b, weight_tmp, stride = self.wrapped_module.stride, padding = self.wrapped_module.padding)
                    res_sliced = res_sliced + res_tmp_b
                
                res_sliced = res_sliced / scale_wa
                if self.wrapped_module.bias is not None:
                    res_sliced = res_sliced + self.wrapped_module.bias.reshape(1, -1, 1, 1)
                return res_sliced
            else:#����kernel
                para = paranumber
                paralevel = self.wrapped_module.in_features // para
                last_element = self.wrapped_module.in_features % para
                for cib in range(0, paralevel):
                    res_sliced_tmp = torch.cuda.FloatTensor(res.shape).fill_(0)
                    in_cis = input_tmp[:, cib*para:cib*para+para]
                    w_cis = weight_tmp[:, cib*para:cib*para+para] 
                    for ba in range(0, bit_activ // 2):
                        for bw in range(0, bit_weight):
                            w_tmp = ((w_cis+(1<<(bit_weight))).int() & (1<<bw)) >> bw
                            w_tmp = w_tmp.float()

                            if(R_on):
                                for i in range(0, w_tmp.size(0)):
                                    for j in range(0, w_tmp.size(1)):
                                        if(w_tmp[i,j] == 0): w_tmp[i,j] = get_ln_r(loc = self.config.rram_conductance[0], scale= self.config.rram_variation[0])
                                        if(w_tmp[i,j] == 1): w_tmp[i,j] = get_hn_r(loc = self.config.rram_conductance[1], scale= self.config.rram_variation[1])

                            a_tmp = (in_cis.int() & (3<<(ba*2))) >> (ba*2)
                            
                            res_tmp = F.linear(a_tmp.float(), w_tmp.float(), bias=None)



                            if bw == bit_weight-1:
                                res_tmp = -res_tmp

                            res_sliced_tmp = res_sliced_tmp + (res_tmp<<(2*ba+bw)).float()

                    if(R_on):
                        res_sliced_tmp /= 2050
                        res_sliced_tmp = torch.round(res_sliced_tmp)
                    res_sliced = res_sliced + res_sliced_tmp
                
                if(last_element != 0):
                    res_sliced_tmp = torch.cuda.FloatTensor(res.shape).fill_(0)
                    in_cis = input_tmp[:, paralevel*para:paralevel*para+last_element]
                    w_cis = weight_tmp[:, paralevel*para:paralevel*para+last_element]
                    for ba in range(0, bit_activ // 2):
                        for bw in range(0, bit_weight):
                            w_tmp = ((w_cis+(1<<(bit_weight))).int() & (1<<bw)) >> bw
                            w_tmp = w_tmp.float()

                            if(R_on):
                                for i in range(0, w_tmp.size(0)):
                                    for j in range(0, w_tmp.size(1)):
                                        if(w_tmp[i,j] == 0): w_tmp[i,j] = get_ln_r(loc = self.config.rram_conductance[0], scale= self.config.rram_variation[0])
                                        if(w_tmp[i,j] == 1): w_tmp[i,j] = get_hn_r(loc = self.config.rram_conductance[1], scale= self.config.rram_variation[1])

                            a_tmp = (in_cis.int() & (3<<(ba*2))) >> (ba*2)
                            a_nz = (a_tmp!=0).sum(dim=1, keepdim=True) 
                            
                            res_tmp = F.linear(a_tmp.float(), w_tmp.float(), bias=None)
                            if bw == bit_weight-1:
                                res_tmp = -res_tmp

                            res_sliced_tmp = res_sliced_tmp+(res_tmp<<(2*ba+bw)).float()

                    if(R_on):
                        res_sliced_tmp /= 2050
                        # res_sliced_tmp /= 0.001
                        res_sliced_tmp = torch.round(res_sliced_tmp)
                    res_sliced = res_sliced + res_sliced_tmp
                
                res_sliced = res_sliced / scale_wa
                res_sliced = res_sliced + self.wrapped_module.bias
                return res_sliced    
           
        elif mode == "bit_kernel":
            torch.backends.cudnn.deterministic = True
            bit_weight = self.config.bit_wt
            bit_activ = self.config.bit_act
            paranumber = self.config.para
            # loc=34.16666667
            # scale=1.929166667
            loc=self.config.rram_conductance[0]
            scale=self.config.rram_variation[0]

            scale_w = self.wrapped_module.weight_scale
            scale_a = self.conv_q.scale
            scale_wa = scale_w * scale_a

            input_ori = torch.round(input[0] * scale_a - self.conv_q.zero_point)
            input_ori = torch.clamp(input_ori, 0, (2**bit_activ)-1)
            weight_ori = torch.round(self.wrapped_module.weight*scale_w)
            
            ce = self.wrapped_module.weight*scale_w

            wnz = (weight_ori.view(-1) == 0).sum()*1.0/weight_ori.numel()

            in_a = self.conv_q(*input)

            res = self.wrapped_module(in_a)
            res_sliced = torch.cuda.FloatTensor(res.shape).fill_(0) 

            if hasattr(self.wrapped_module, 'kernel_size'):
                for ba in range(0, bit_activ//2):#��֧��2������������
                    for bw in range(0, bit_weight):
                        input_tmp = (input_ori.int() & (3<<(ba*2))) >> (ba*2)#����λȡ�����ֵ
                        weight_tmp = ((weight_ori+(1<<(bit_weight))).int() & (1<<bw)) >> bw

                        kernel_size = self.wrapped_module.kernel_size[0] * self.wrapped_module.kernel_size[1]
                        para = paranumber // kernel_size

                        paralevel = (self.wrapped_module.in_channels * kernel_size) // paranumber
                        last_number = (self.wrapped_module.in_channels * kernel_size) % paranumber
                        if(last_number != 0): paralevel += 1

                        remain_number = 0
                        for cib in range(0, paralevel):
                            res_sliced_tmp = torch.cuda.FloatTensor(res.shape).fill_(0)
                            arrayrow = min(paranumber, kernel_size * self.wrapped_module.in_channels - cib*paranumber)
                            for numcell in range(0, arrayrow):
                                number_all = cib * paranumber + numcell
                                channel = number_all // kernel_size
                                remain_number = number_all % kernel_size
                                krow = remain_number // self.wrapped_module.kernel_size[1]
                                kcol = remain_number % self.wrapped_module.kernel_size[0]
                                in_cis = input_tmp[:, channel:channel+1, :, :]
                                w_cis = weight_tmp[:, channel:channel+1, :, :]
                                m = nn.ZeroPad2d((self.wrapped_module.padding[0] - kcol, self.wrapped_module.padding[0]-(self.wrapped_module.kernel_size[1]-kcol-1), self.wrapped_module.padding[1] - krow, self.wrapped_module.padding[1]-(self.wrapped_module.kernel_size[0]-krow-1)))
                                a_pad = m(in_cis)

                                w_tmp = w_cis[:,:,krow:krow+1,kcol:kcol+1]
                                #ȡ��1����λ��weight,���ֱ�ȡ��1bit
                                w_tmp = w_tmp.float()
                                a_tmp = a_pad.int()
                                if(R_on):

                                    rram_error_1 = torch.cuda.FloatTensor(w_tmp.shape).normal_(self.config.rram_conductance[1], self.config.rram_variation[1])

                                    rram_error_0 = torch.cuda.FloatTensor(w_tmp.shape).normal_(loc, scale)


                                    w_tmp = torch.where(w_tmp==1, rram_error_1, w_tmp)
                                    w_tmp = torch.where(w_tmp==0, rram_error_0, w_tmp)

                                res_tmp = F.conv2d(a_tmp.float(), w_tmp.float(), bias=None, stride=self.wrapped_module.stride, padding=0)

                                if bw == bit_weight-1:
                                    res_tmp = -res_tmp

                                res_sliced_tmp = res_sliced_tmp + (res_tmp<<(ba*2+bw)).float()#!!!!!!!
                            if(R_on):
                                res_sliced_tmp /= (self.config.rram_conductance[1])
                                res_sliced_tmp = torch.round(res_sliced_tmp)

                            res_sliced = res_sliced + res_sliced_tmp
                        

                if self.conv_q.zero_point != 0:
                    in_tmp_b = torch.ones_like(in_a)*self.conv_q.zero_point
                    res_tmp_b = F.conv2d(in_tmp_b, weight_ori, stride = self.wrapped_module.stride, padding = self.wrapped_module.padding)
                    res_sliced = res_sliced + res_tmp_b

                
                res_sliced = res_sliced / scale_wa

                if self.wrapped_module.bias is not None:
                    res_sliced = res_sliced + self.wrapped_module.bias.reshape(1, -1, 1, 1)
                return res_sliced
            else:#����kernel
                para = paranumber

                for ba in range(0, bit_activ//2):#��֧��2������������
                    for bw in range(0, bit_weight):
                        input_tmp = (input_ori.int() & (3<<(ba*2))) >> (ba*2)#����λȡ�����ֵ
                        weight_tmp = ((weight_ori+(1<<(bit_weight))).int() & (1<<bw)) >> bw

                        paralevel = self.wrapped_module.in_features // para
                        last_element = self.wrapped_module.in_features % para
                        for cib in range(0, paralevel):
                            res_sliced_tmp = torch.cuda.FloatTensor(res.shape).fill_(0)
                            in_cis = input_tmp[:, cib*para:cib*para+para]
                            w_cis = weight_tmp[:, cib*para:cib*para+para] 

                            w_tmp = w_cis
                            w_tmp = w_tmp.float()

                            if(R_on):
                                rram_error_1 = torch.cuda.FloatTensor(w_tmp.shape).normal_(self.config.rram_conductance[1], self.config.rram_variation[1])
                                rram_error_0 = torch.cuda.FloatTensor(w_tmp.shape).normal_(loc, scale)

                                w_tmp = torch.where(w_tmp==1, rram_error_1, w_tmp)
                                w_tmp = torch.where(w_tmp==0, rram_error_0, w_tmp)

                            a_tmp = in_cis.int()
                            
                            res_tmp = F.linear(a_tmp.float(), w_tmp.float(), bias=None)

                            if bw == bit_weight-1:
                                res_tmp = -res_tmp

                            res_sliced_tmp = res_sliced_tmp+(res_tmp.int()<<(2*ba+bw)).float()
                            if(R_on):
                                res_sliced_tmp /= (2050)
                                # res_sliced_tmp /= 0.001
                                res_sliced_tmp = torch.round(res_sliced_tmp)

                            res_sliced = res_sliced + res_sliced_tmp
                        
                        if(last_element != 0):
                            res_sliced_tmp = torch.cuda.FloatTensor(res.shape).fill_(0)
                            in_cis = input_tmp[:, paralevel*para:paralevel*para+last_element]
                            w_cis = weight_tmp[:, paralevel*para:paralevel*para+last_element]

                            w_tmp = w_cis

                            if(R_on):
                                for i in range(0, w_tmp.size(0)):
                                    for j in range(0, w_tmp.size(1)):
                                        if(w_tmp[i,j] == 0): w_tmp[i,j] = get_ln_r(loc=loc, scale=scale)
                                        if(w_tmp[i,j] == 1): w_tmp[i,j] = get_hn_r(loc = self.config.rram_conductance[1], scale= self.config.rram_variation[1])

                            a_tmp = in_cis.int()
                            
                            res_tmp = F.linear(a_tmp.float(), w_tmp.float(), bias=None)

                            if bw == bit_weight-1:
                                res_tmp = -res_tmp

                            res_sliced_tmp = res_sliced_tmp+(res_tmp.int()<<(2*ba+bw)).float()
                            if(R_on):
                                res_sliced_tmp /= self.config.rram_conductance[1]
                                res_sliced_tmp = torch.round(res_sliced_tmp)
                            res_sliced = res_sliced + res_sliced_tmp
                            
                res_sliced = res_sliced / scale_wa
                res_sliced = res_sliced + self.wrapped_module.bias
                return res_sliced
            exit(0)
        elif mode == "sparate_pn_kernel":
            torch.backends.cudnn.deterministic = True
            bit_weight = self.config.bit_wt
            bit_activ = self.config.bit_act
            paranumber = self.config.para
            # loc = 10.25
            # scale = 0.57875
            loc=self.config.rram_conductance[0]
            scale=self.config.rram_variation[0]
            delta = self.config.ADC_delta


            scale_w = self.wrapped_module.weight_scale
            scale_a = self.conv_q.scale
            scale_wa = scale_w * scale_a
            input_ori = torch.round(input[0] * scale_a - self.conv_q.zero_point)
            input_ori = torch.clamp(input_ori, 0, (2**bit_activ)-1)
            weight_ori = torch.round(self.wrapped_module.weight*scale_w)
            w_p_ori = torch.where(weight_ori > 0,weight_ori, torch.zeros_like(weight_ori) )#����
            w_n_ori = torch.where(weight_ori < 0,-(weight_ori), torch.zeros_like(weight_ori) )#����

            in_a = self.conv_q(*input)
            res = self.wrapped_module(in_a)
            res_sliced = torch.cuda.FloatTensor(res.shape).fill_(0) 

            if hasattr(self.wrapped_module, 'kernel_size'):
                for ba in range(0, bit_activ//2):#��֧��2������������
                    for bw in range(0, bit_weight):
                        input_tmp = (input_ori.int() & (3<<(ba*2))) >> (ba*2)#����λȡ�����ֵ
                        weight_tmp_p = ((w_p_ori+(1<<(bit_weight))).int() & (1<<bw)) >> bw
                        weight_tmp_n = ((w_n_ori+(1<<(bit_weight))).int() & (1<<bw)) >> bw

                        kernel_size = self.wrapped_module.kernel_size[0] * self.wrapped_module.kernel_size[1]
                        para = paranumber // kernel_size
                        para_remain = paranumber % kernel_size

                        paralevel = (self.wrapped_module.in_channels * kernel_size) // paranumber
                        last_number = (self.wrapped_module.in_channels * kernel_size) % paranumber
                        if(last_number != 0): paralevel += 1


                        remain_number = 0
                        for cib in range(0, paralevel):
                            res_sliced_tmp = torch.cuda.FloatTensor(res.shape).fill_(0)
                            arrayrow = min(paranumber, kernel_size * self.wrapped_module.in_channels - cib*paranumber)
                            for numcell in range(0, arrayrow):
                                number_all = cib * paranumber + numcell
                                channel = number_all // kernel_size
                                remain_number = number_all % kernel_size
                                krow = remain_number // self.wrapped_module.kernel_size[1]
                                kcol = remain_number % self.wrapped_module.kernel_size[0]
                                in_cis = input_tmp[:, channel:channel+1, :, :]
                                w_cis_p = weight_tmp_p[:, channel:channel+1, :, :]
                                w_cis_n = weight_tmp_n[:, channel:channel+1, :, :]
                                m = nn.ZeroPad2d((self.wrapped_module.padding[0] - kcol, self.wrapped_module.padding[0]-(self.wrapped_module.kernel_size[1]-kcol-1), self.wrapped_module.padding[1] - krow, self.wrapped_module.padding[1]-(self.wrapped_module.kernel_size[0]-krow-1)))
                                a_pad = m(in_cis)

                                w_tmp_p = w_cis_p[:,:,krow:krow+1,kcol:kcol+1]
                                w_tmp_n = w_cis_n[:,:,krow:krow+1,kcol:kcol+1]
                                #ȡ��1����λ��weight,���ֱ�ȡ��1bit
                                w_tmp_p = w_tmp_p.float()
                                w_tmp_n = w_tmp_n.float()
                                a_tmp = a_pad.int()
                                if(R_on):
                                    rram_error_1 = torch.cuda.FloatTensor(w_tmp_p.shape).normal_(self.config.rram_conductance[1],self.config.rram_variation[1])
                                    rram_error_0_p = torch.cuda.FloatTensor(w_tmp_p.shape).normal_(loc, scale)
                                    rram_error_0_n = torch.cuda.FloatTensor(w_tmp_n.shape).normal_(loc, scale)

                                    w_tmp_p = torch.where(w_tmp_p==1, rram_error_1, w_tmp_p)
                                    w_tmp_p = torch.where(w_tmp_p==0, rram_error_0_p, w_tmp_p)
                                    w_tmp_n = torch.where(w_tmp_n==1, rram_error_1, w_tmp_n)
                                    w_tmp_n = torch.where(w_tmp_n==0, rram_error_0_n, w_tmp_n)

                                res_tmp_p = F.conv2d(a_tmp.float(), w_tmp_p.float(), bias=None, stride=self.wrapped_module.stride, padding=0)
                                res_tmp_n = F.conv2d(a_tmp.float(), w_tmp_n.float(), bias=None, stride=self.wrapped_module.stride, padding=0)
                                res_tmp = res_tmp_p - res_tmp_n

                                res_sliced_tmp = res_sliced_tmp + (res_tmp<<(ba*2+bw)).float()#!!!!!!!

                            if(R_on):
                                if(ADC_error):
                                    adc_error = torch.normal(mean=res_sliced_tmp, std = delta)
                                    res_sliced_tmp = adc_error / (self.config.rram_conductance[1]-loc)
                                else: res_sliced_tmp /= self.config.rram_conductance[1] - loc
                                res_sliced_tmp = torch.round(res_sliced_tmp)

                            res_sliced = res_sliced + res_sliced_tmp
                        

                if self.conv_q.zero_point != 0:
                    in_tmp_b = torch.ones_like(in_a)*self.conv_q.zero_point
                    res_tmp_b = F.conv2d(in_tmp_b, weight_ori, stride = self.wrapped_module.stride, padding = self.wrapped_module.padding)
                    res_sliced = res_sliced + res_tmp_b

                
                res_sliced = res_sliced / scale_wa

                if self.wrapped_module.bias is not None:
                    res_sliced = res_sliced + self.wrapped_module.bias.reshape(1, -1, 1, 1)

                return res_sliced
            else:#����kernel
                para = paranumber

                for ba in range(0, bit_activ//2):#��֧��2������������
                    for bw in range(0, bit_weight):
                        input_tmp = (input_ori.int() & (3<<(ba*2))) >> (ba*2)#����λȡ�����ֵ
                        weight_tmp_p = ((w_p_ori+(1<<(bit_weight))).int() & (1<<bw)) >> bw
                        weight_tmp_n = ((w_n_ori+(1<<(bit_weight))).int() & (1<<bw)) >> bw

                        paralevel = self.wrapped_module.in_features // para
                        last_element = self.wrapped_module.in_features % para
                        for cib in range(0, paralevel):
                            res_sliced_tmp = torch.cuda.FloatTensor(res.shape).fill_(0)
                            in_cis = input_tmp[:, cib*para:cib*para+para]
                            w_cis_p = weight_tmp_p[:, cib*para:cib*para+para] 
                            w_cis_n = weight_tmp_n[:, cib*para:cib*para+para] 

                            w_tmp_p = w_cis_p.float()
                            w_tmp_n = w_cis_n.float()

                            if(R_on):
                                rram_error_1 = torch.cuda.FloatTensor(w_tmp_p.shape).normal_(self.config.rram_conductance[1],self.config.rram_variation[1])
                                rram_error_0_p = torch.cuda.FloatTensor(w_tmp_p.shape).normal_(loc, scale)
                                rram_error_0_n = torch.cuda.FloatTensor(w_tmp_n.shape).normal_(loc, scale)


                                w_tmp_p = torch.where(w_tmp_p==1, rram_error_1, w_tmp_p)
                                w_tmp_p = torch.where(w_tmp_p==0, rram_error_0_p, w_tmp_p)
                                w_tmp_n = torch.where(w_tmp_n==1, rram_error_1, w_tmp_n)
                                w_tmp_n = torch.where(w_tmp_n==0, rram_error_0_n, w_tmp_n)

                            a_tmp = in_cis.int()
                            
                            res_tmp_p = F.linear(a_tmp.float(), w_tmp_p.float(), bias=None)
                            res_tmp_n = F.linear(a_tmp.float(), w_tmp_n.float(), bias=None)
                            res_tmp = res_tmp_p - res_tmp_n


                            res_sliced_tmp = res_sliced_tmp+(res_tmp.int()<<(2*ba+bw)).float()
                            if(R_on):
                                if(ADC_error):
                                    adc_error = torch.normal(mean=res_sliced_tmp, std = delta)
                                    res_sliced_tmp = adc_error / (self.config.rram_conductance[1]-loc)
                                else: res_sliced_tmp /= (self.config.rram_conductance[1]-loc)
                                res_sliced_tmp = torch.round(res_sliced_tmp)
                            res_sliced = res_sliced + res_sliced_tmp
                        
                        if(last_element != 0):
                            res_sliced_tmp = torch.cuda.FloatTensor(res.shape).fill_(0)

                            in_cis = input_tmp[:, paralevel*para:paralevel*para+last_element]
                            w_cis_p = weight_tmp_p[:, paralevel*para:paralevel*para+last_element]
                            w_cis_n = weight_tmp_n[:, paralevel*para:paralevel*para+last_element]


                            w_tmp_p = w_cis_p.float()
                            w_tmp_n = w_cis_n.float()

                            if(R_on):

                                rram_error_1 = torch.cuda.FloatTensor(w_tmp_p.shape).normal_(self.config.rram_conductance[1],self.config.rram_variation[1])
                                rram_error_0_p = torch.cuda.FloatTensor(w_tmp_p.shape).normal_(loc, scale)
                                rram_error_0_n = torch.cuda.FloatTensor(w_tmp_n.shape).normal_(loc, scale)

                                w_tmp_p = torch.where(w_tmp_p==1, rram_error_1, w_tmp_p)
                                w_tmp_p = torch.where(w_tmp_p==0, rram_error_0_p, w_tmp_p)
                                w_tmp_n = torch.where(w_tmp_n==1, rram_error_1, w_tmp_n)
                                w_tmp_n = torch.where(w_tmp_n==0, rram_error_0_n, w_tmp_n)
                            a_tmp = in_cis.int()
                            
                            res_tmp_p = F.linear(a_tmp.float(), w_tmp_p.float(), bias=None)
                            res_tmp_n = F.linear(a_tmp.float(), w_tmp_n.float(), bias=None)
                            res_tmp = res_tmp_p - res_tmp_n

                            res_sliced_tmp = res_sliced_tmp+(res_tmp.int()<<(2*ba+bw)).float()
                            if(R_on):
                                if(ADC_error):
                                    adc_error = torch.normal(mean=res_sliced_tmp, std = delta)
                                    res_sliced_tmp = adc_error / (self.config.rram_conductance[1]-loc)
                                else: res_sliced_tmp /= (self.config.rram_conductance[1]-loc)
                                res_sliced_tmp = torch.round(res_sliced_tmp)
                              
                            res_sliced = res_sliced + res_sliced_tmp
                            
                    
                res_sliced = res_sliced / scale_wa
                res_sliced = res_sliced + self.wrapped_module.bias

                return res_sliced
        elif mode == "varia_pn_kernel":
            torch.backends.cudnn.deterministic = True
            bit_weight = self.config.bit_wt
            bit_activ = self.config.bit_act
            paranumber = self.config.para
            # loc = 10.25
            # scale = 0.57875
            loc=self.config.rram_conductance[0]
            scale=self.config.rram_variation[0]
            delta = self.config.ADC_delta
            # thre = 7 
            # ad_limit = 15
            # bit_weight = 4
            # bit_activ = 4
            # paranumber = 16
            # loc = 10.25
            # scale = 0.57875
            # delta = 400

            scale_w = self.wrapped_module.weight_scale
            scale_a = self.conv_q.scale
            scale_wa = scale_w * scale_a
            input_ori = torch.round(input[0] * scale_a - self.conv_q.zero_point)
            input_ori = torch.clamp(input_ori, 0, (2**bit_activ)-1)
            weight_ori = torch.round(self.wrapped_module.weight*scale_w)
            w_p_ori = torch.where(weight_ori > 0,weight_ori, torch.zeros_like(weight_ori) )#����
            w_n_ori = torch.where(weight_ori < 0,-(weight_ori), torch.zeros_like(weight_ori) )#����

            in_a = self.conv_q(*input)
            res = self.wrapped_module(in_a)

            res_sliced = torch.cuda.FloatTensor(res.shape).fill_(0) 

            if hasattr(self.wrapped_module, 'kernel_size'):
                for ba in range(0, bit_activ//2):#��֧��2������������
                    for bw in range(0, bit_weight):
                        res_sliced_tmp = torch.cuda.FloatTensor(res.shape).fill_(0)
                        input_tmp = (input_ori.int() & (3<<(ba*2))) >> (ba*2)#����λȡ�����ֵ
                        weight_tmp_p = ((w_p_ori+(1<<(bit_weight))).int() & (1<<bw)) >> bw
                        weight_tmp_n = ((w_n_ori+(1<<(bit_weight))).int() & (1<<bw)) >> bw

                        kernel_size = self.wrapped_module.kernel_size[0] * self.wrapped_module.kernel_size[1]

                        paralevel = (self.wrapped_module.in_channels * kernel_size) // paranumber
                        last_number = (self.wrapped_module.in_channels * kernel_size) % paranumber
                        if(last_number != 0): paralevel += 1
                        
                        

                        input_tmp = input_tmp.float()
                        weight_tmp_p = weight_tmp_p.float()
                        weight_tmp_n = weight_tmp_n.float()

                        if(R_SAF):
                            saf_p = torch.cuda.FloatTensor(weight_tmp_p.shape).fill_(0)
                            saf_p = torch.rand(size=weight_tmp_p.shape, out=saf_p)

                            zero = torch.cuda.FloatTensor(weight_tmp_p.shape).fill_(0)
                            one = torch.cuda.FloatTensor(weight_tmp_p.shape).fill_(1)
                            saf_n = torch.cuda.FloatTensor(weight_tmp_n.shape).fill_(0)
                            saf_n = torch.rand(size=weight_tmp_n.shape, out=saf_n)
                            weight_tmp_p = torch.where(saf_p<self.config.SA1, zero, weight_tmp_p)
                            weight_tmp_p = torch.where(saf_p>self.config.SA0, one, weight_tmp_p)
                            weight_tmp_n = torch.where(saf_n<self.config.SA1, zero, weight_tmp_n)
                            weight_tmp_n = torch.where(saf_n>self.config.SA0, one, weight_tmp_n)

                        if(R_on):

                            rram_error_1 = torch.cuda.FloatTensor(weight_tmp_p.shape).normal_(self.config.rram_conductance[1],self.config.rram_variation[1])

                            rram_error_0_p = torch.cuda.FloatTensor(weight_tmp_p.shape).normal_(loc, scale)
                            rram_error_0_n = torch.cuda.FloatTensor(weight_tmp_n.shape).normal_(loc, scale)


                            weight_tmp_p = torch.where(weight_tmp_p==1, rram_error_1, weight_tmp_p)
                            weight_tmp_p = torch.where(weight_tmp_p==0, rram_error_0_p, weight_tmp_p)
                            weight_tmp_n = torch.where(weight_tmp_n==1, rram_error_1, weight_tmp_n)
                            weight_tmp_n = torch.where(weight_tmp_n==0, rram_error_0_n, weight_tmp_n)

                        res_tmp_p = F.conv2d(input_tmp.float(), weight_tmp_p.float(), bias=None, stride=self.wrapped_module.stride, padding=self.wrapped_module.padding)
                        res_tmp_n = F.conv2d(input_tmp.float(), weight_tmp_n.float(), bias=None, stride=self.wrapped_module.stride, padding=self.wrapped_module.padding)
                        res_tmp = res_tmp_p - res_tmp_n

                        res_sliced_tmp = res_sliced_tmp + (res_tmp<<(ba*2+bw)).float()#!!!!!!!

                        if(R_on):
                            if(ADC_error):
                                adc_error = torch.cuda.FloatTensor(res_sliced_tmp.shape).fill_(0)
                                for i in range(0, paralevel):
                                    adc_error += torch.normal(mean=torch.cuda.FloatTensor(res_sliced_tmp.shape).fill_(0), std = delta)
                                res_sliced_tmp += adc_error
                            res_sliced_tmp = res_sliced_tmp / (self.config.rram_conductance[1]-loc)
                            res_sliced_tmp = torch.round(res_sliced_tmp)

                        res_sliced = res_sliced + res_sliced_tmp
                        

                if self.conv_q.zero_point != 0:
                    in_tmp_b = torch.ones_like(in_a)*self.conv_q.zero_point
                    res_tmp_b = F.conv2d(in_tmp_b, weight_ori, stride = self.wrapped_module.stride, padding = self.wrapped_module.padding)
                    res_sliced = res_sliced + res_tmp_b

                
                res_sliced = res_sliced / scale_wa

                if self.wrapped_module.bias is not None:
                    res_sliced = res_sliced + self.wrapped_module.bias.reshape(1, -1, 1, 1)
                return res_sliced
            else:#����kernel
                para = paranumber

                for ba in range(0, bit_activ//2):#��֧��2������������
                    for bw in range(0, bit_weight):
                        res_sliced_tmp = torch.cuda.FloatTensor(res.shape).fill_(0)
                        input_tmp = (input_ori.int() & (3<<(ba*2))) >> (ba*2)#����λȡ�����ֵ
                        weight_tmp_p = ((w_p_ori+(1<<(bit_weight))).int() & (1<<bw)) >> bw
                        weight_tmp_n = ((w_n_ori+(1<<(bit_weight))).int() & (1<<bw)) >> bw

                        paralevel = self.wrapped_module.in_features // para
                        last_element = self.wrapped_module.in_features % para
                        if(last_element != 0): paralevel += 1

                        input_tmp = input_tmp.float()
                        weight_tmp_p = weight_tmp_p.float()
                        weight_tmp_n = weight_tmp_n.float()
                        if(R_on):
                                
                            rram_error_1 = torch.cuda.FloatTensor(weight_tmp_p.shape).normal_(self.config.rram_conductance[1],self.config.rram_variation[1])
                            rram_error_0_p = torch.cuda.FloatTensor(weight_tmp_p.shape).normal_(loc, scale)
                            rram_error_0_n = torch.cuda.FloatTensor(weight_tmp_p.shape).normal_(loc, scale)

                            weight_tmp_p = torch.where(weight_tmp_p==1, rram_error_1, weight_tmp_p)
                            weight_tmp_p = torch.where(weight_tmp_p==0, rram_error_0_p, weight_tmp_p)
                            weight_tmp_n = torch.where(weight_tmp_n==1, rram_error_1, weight_tmp_n)
                            weight_tmp_n = torch.where(weight_tmp_n==0, rram_error_0_n, weight_tmp_n)
                            
                        res_tmp_p = F.linear(input_tmp.float(), weight_tmp_p.float(), bias=None)
                        res_tmp_n = F.linear(input_tmp.float(), weight_tmp_n.float(), bias=None)
                        res_tmp = res_tmp_p - res_tmp_n


                        res_sliced_tmp = res_sliced_tmp+(res_tmp.int()<<(2*ba+bw)).float()

                        
                        if(R_on):
                            if(ADC_error):
                                adc_error = torch.cuda.FloatTensor(res_sliced_tmp.shape).fill_(0)
                                for i in range(0, paralevel):
                                    adc_error += torch.normal(mean=torch.cuda.FloatTensor(res_sliced_tmp.shape).fill_(0), std = delta)
                                res_sliced_tmp += adc_error
                            res_sliced_tmp = res_sliced_tmp / (self.config.rram_conductance[1]-loc)

                            res_sliced_tmp = torch.round(res_sliced_tmp)

                        res_sliced = res_sliced + res_sliced_tmp
                            
                    
                res_sliced = res_sliced / scale_wa
                res_sliced = res_sliced + self.wrapped_module.bias
                return res_sliced
        elif mode == "fast_2mod":
            torch.backends.cudnn.deterministic = True
            bit_weight = self.config.bit_wt
            bit_activ = self.config.bit_act
            paranumber = self.config.para
            # loc = 10.25
            # scale = 0.57875
            loc=self.config.rram_conductance[0]
            scale=self.config.rram_variation[0]
            delta = self.config.ADC_delta

            # paranumber = 16
            # loc = 10.25
            # scale = 0.57875
            # delta = 2400

            scale_w = self.wrapped_module.weight_scale
            scale_a = self.conv_q.scale
            scale_wa = scale_w * scale_a
            input_ori = torch.round(input[0] * scale_a - self.conv_q.zero_point)
            input_ori = torch.clamp(input_ori, 0, (2**bit_activ)-1)

            weight_ori = torch.round(self.wrapped_module.weight*scale_w)

            in_a = self.conv_q(*input)

            res = self.wrapped_module(in_a)

            res_sliced = torch.cuda.FloatTensor(res.shape).fill_(0) 

            if hasattr(self.wrapped_module, 'kernel_size'):
                for ba in range(0, bit_activ//2):#��֧��2������������
                    for bw in range(0, bit_weight):
                        res_sliced_tmp = torch.cuda.FloatTensor(res.shape).fill_(0)
                        input_tmp = (input_ori.int() & (3<<(ba*2))) >> (ba*2)#����λȡ�����ֵ
                        weight_tmp = ((weight_ori+(1<<(bit_weight))).int() & (1<<bw)) >> bw

                        
                        

                        input_tmp = input_tmp.float()
                        weight_tmp = weight_tmp.float()

                        if(R_SAF):
                            saf = torch.cuda.FloatTensor(weight_tmp.shape).fill_(0)
                            saf = torch.rand(size=weight_tmp.shape, out=saf)

                            zero = torch.cuda.FloatTensor(weight_tmp.shape).fill_(0)
                            one = torch.cuda.FloatTensor(weight_tmp.shape).fill_(1)
                            weight_tmp = torch.where(saf_p<self.config.SA1, zero, weight_tmp)
                            weight_tmp = torch.where(saf_p>self.config.SA0, one, weight_tmp)

                        if(R_on):

                            rram_error_1 = torch.cuda.FloatTensor(weight_tmp_p.shape).normal_(self.config.rram_conductance[1],self.config.rram_variation[1])

                            rram_error_0 = torch.cuda.FloatTensor(weight_tmp.shape).normal_(loc, scale)


                            weight_tmp = torch.where(weight_tmp_p==1, rram_error_1, weight_tmp)
                            weight_tmp = torch.where(weight_tmp_p==0, rram_error_0_p, weight_tmp)

                        res_tmp = F.conv2d(input_tmp.float(), weight_tmp.float(), bias=None, stride=self.wrapped_module.stride, padding=self.wrapped_module.padding)


                        res_sliced_tmp = res_sliced_tmp + (res_tmp<<(ba*2+bw)).float()#!!!!!!!

                        if(R_on):
                            res_sliced_tmp = res_sliced_tmp / self.config.rram_conductance[1]
                            res_sliced_tmp = torch.round(res_sliced_tmp)


                        res_sliced = res_sliced + res_sliced_tmp
                        

                if self.conv_q.zero_point != 0:

                    in_tmp_b = torch.ones_like(in_a)*self.conv_q.zero_point
                    res_tmp_b = F.conv2d(in_tmp_b, weight_ori, stride = self.wrapped_module.stride, padding = self.wrapped_module.padding)
                    res_sliced = res_sliced + res_tmp_b

                
                res_sliced = res_sliced / scale_wa

                if self.wrapped_module.bias is not None:
                    res_sliced = res_sliced + self.wrapped_module.bias.reshape(1, -1, 1, 1)

                return res_sliced
            else:#����kernel
                #return res
                para = paranumber

                for ba in range(0, bit_activ//2):#��֧��2������������
                    for bw in range(0, bit_weight):
                        res_sliced_tmp = torch.cuda.FloatTensor(res.shape).fill_(0)
                        input_tmp = (input_ori.int() & (3<<(ba*2))) >> (ba*2)#����λȡ�����ֵ

                        weight_tmp = ((weight_ori+(1<<(bit_weight))).int() & (1<<bw)) >> bw


                        input_tmp = input_tmp.float()
                        weight_tmp = weight_tmp.float()

                        if(R_on):
                                
                            rram_error_1 = torch.cuda.FloatTensor(weight_tmp.shape).normal_(self.config.rram_conductance[1],self.config.rram_variation[1])
                            rram_error_0 = torch.cuda.FloatTensor(weight_tmp.shape).normal_(loc, scale)


                            weight_tmp = torch.where(weight_tmp_p==1, rram_error_1, weight_tmp)
                            weight_tmp = torch.where(weight_tmp_p==0, rram_error_0, weight_tmp)
                            
                        res_tmp = F.linear(input_tmp.float(), weight_tmp.float(), bias=None)

                        res_sliced_tmp = res_sliced_tmp+(res_tmp.int()<<(2*ba+bw)).float()

                        
                        if(R_on):

                            res_sliced_tmp = res_sliced_tmp / self.config.rram_conductance[1]
                            res_sliced_tmp = torch.round(res_sliced_tmp)

                        res_sliced = res_sliced + res_sliced_tmp
                            
                    
                res_sliced = res_sliced / scale_wa
                res_sliced = res_sliced + self.wrapped_module.bias

                return res_sliced
        elif mode == "4bit_rram":
            self.locarray = torch.cuda.FloatTensor([0.001, 0.006667, 0.01333, 0.02, 0.027, 0.03335, 0.042, 0.04667, 0.05335, 0.06, 0.0667, 0.07337, 0.08, 0.08667, 0.09333, 0.1005])
            self.sigmaratio = 1
            torch.backends.cudnn.deterministic = True
            thre = 7 
            ad_limit = 15
            bit_weight = 4
            bit_activ = 4
            # para = 18
            paranumber = 16

            loc_0, scale_0 = self.locarray[0], 0.005*self.sigmaratio
            loc_1, scale_1 = self.locarray[1], 0.005*self.sigmaratio
            loc_2, scale_2 = self.locarray[2], 0.005*self.sigmaratio
            loc_3, scale_3 = self.locarray[3], 0.005*self.sigmaratio
            loc_4, scale_4 = self.locarray[4], 0.005*self.sigmaratio
            loc_5, scale_5 = self.locarray[5], 0.005*self.sigmaratio
            loc_6, scale_6 = self.locarray[6], 0.005*self.sigmaratio
            loc_7, scale_7 = self.locarray[7], 0.005*self.sigmaratio
            loc_8, scale_8 = self.locarray[8], 0.005*self.sigmaratio
            loc_9, scale_9 = self.locarray[9], 0.005*self.sigmaratio
            loc_10, scale_10 = self.locarray[10], 0.005*self.sigmaratio
            loc_11, scale_11 = self.locarray[11], 0.005*self.sigmaratio
            loc_12, scale_12 = self.locarray[12], 0.005*self.sigmaratio
            loc_13, scale_13 = self.locarray[13], 0.005*self.sigmaratio
            loc_14, scale_14 = self.locarray[14], 0.005*self.sigmaratio
            loc_15, scale_15 = self.locarray[15], 0.005*self.sigmaratio


            # loc_0, scale_0 = 0.001, 0.005
            # loc_1, scale_1 = 0.006667, 0.005
            # loc_2, scale_2 = 0.01333, 0.005
            # loc_3, scale_3 = 0.02, 0.005
            # loc_4, scale_4 = 0.027, 0.005
            # loc_5, scale_5 = 0.03335, 0.005
            # loc_6, scale_6 = 0.042, 0.005
            # loc_7, scale_7 = 0.04667, 0.005
            # loc_8, scale_8 = 0.05335, 0.005
            # loc_9, scale_9 = 0.06, 0.005
            # loc_10, scale_10 = 0.0667, 0.005
            # loc_11, scale_11 = 0.07337, 0.005
            # loc_12, scale_12 = 0.08, 0.005
            # loc_13, scale_13 = 0.08667, 0.005
            # loc_14, scale_14 = 0.09333, 0.005
            # loc_15, scale_15 = 0.1005, 0.005
            # print('paranumber', paranumber)

            scale_w = self.wrapped_module.weight_scale
            scale_a = self.conv_q.scale
            scale_wa = scale_w * scale_a

            # print(self.wrapped_module)

            input_tmp = torch.round(input[0] * scale_a - self.conv_q.zero_point)
            # print('input',input_tmp)
            input_tmp = torch.clamp(input_tmp, 0, (2**bit_activ)-1)
            weight_tmp = torch.round(self.wrapped_module.weight*scale_w)
            
            # print('weight', self.wrapped_module.weight)
            ce = self.wrapped_module.weight*scale_w

            wnz = (weight_tmp.view(-1) == 0).sum()*1.0/weight_tmp.numel()
            #print("wnz:",wnz)
            in_a = self.conv_q(*input)
            # print('in_a',in_a)
            res = self.wrapped_module(in_a)
            # print('res',res.size(),res)
            #return res`    `
            res_sliced = torch.cuda.FloatTensor(res.shape).fill_(0) 
            res_sliced_tmp = torch.cuda.FloatTensor(res.shape).fill_(0)



            if hasattr(self.wrapped_module, 'kernel_size'):
                # if (self.wrapped_module.in_channels == 31):
                #return res
                kernel_size = self.wrapped_module.kernel_size[0] * self.wrapped_module.kernel_size[1]
                para = paranumber // kernel_size
                para_remain = paranumber % kernel_size
                # print('channel', self.wrapped_module.in_channels)
                paralevel = (self.wrapped_module.in_channels * kernel_size) // paranumber
                last_number = (self.wrapped_module.in_channels * kernel_size) % paranumber
                if(last_number != 0): paralevel += 1

                first_number = 0
                middle_number = 0
                remain_number = 0
                begin_channel = 0
                end_channel = 0
                tmp_channel = 0
                for cib in range(0, paralevel):
                    res_sliced_tmp = torch.cuda.FloatTensor(res.shape).fill_(0)
                    arrayrow = min(paranumber, kernel_size * self.wrapped_module.in_channels - cib*paranumber)
                    for numcell in range(0, arrayrow):
                        number_all = cib * paranumber + numcell
                        channel = number_all // kernel_size
                        remain_number = number_all % kernel_size
                        krow = remain_number // self.wrapped_module.kernel_size[1]
                        kcol = remain_number % self.wrapped_module.kernel_size[0]
                        in_cis = input_tmp[:, channel:channel+1, :, :]
                        w_cis = weight_tmp[:, channel:channel+1, :, :]
                        m = nn.ZeroPad2d((self.wrapped_module.padding[0] - kcol, self.wrapped_module.padding[0]-(self.wrapped_module.kernel_size[1]-kcol-1), self.wrapped_module.padding[1] - krow, self.wrapped_module.padding[1]-(self.wrapped_module.kernel_size[0]-krow-1)))
                        a_pad = m(in_cis)

                        for ba in range(0, bit_activ//2):#��֧��2������������
                                w_tmp = w_cis[:,:,krow:krow+1,kcol:kcol+1]
                                w_tmp = w_tmp.float()                               

                                a_tmp = (a_pad.int() & (3<<(ba*2))) >> (ba*2)#����λȡ�����ֵ

                                if(R_on):
                                    rram_error_0 = torch.cuda.FloatTensor(w_tmp.shape).normal_(loc_0,scale_0)
                                    rram_error_1 = torch.cuda.FloatTensor(w_tmp.shape).normal_(loc_1,scale_1)
                                    rram_error_2 = torch.cuda.FloatTensor(w_tmp.shape).normal_(loc_2,scale_2)
                                    rram_error_3 = torch.cuda.FloatTensor(w_tmp.shape).normal_(loc_3,scale_3)
                                    rram_error_4 = torch.cuda.FloatTensor(w_tmp.shape).normal_(loc_4,scale_4)
                                    rram_error_5 = torch.cuda.FloatTensor(w_tmp.shape).normal_(loc_5,scale_5)
                                    rram_error_6 = torch.cuda.FloatTensor(w_tmp.shape).normal_(loc_6,scale_6)
                                    rram_error_7 = torch.cuda.FloatTensor(w_tmp.shape).normal_(loc_7,scale_7)
                                    rram_error_8 = torch.cuda.FloatTensor(w_tmp.shape).normal_(loc_8,scale_8)
                                    rram_error_9 = torch.cuda.FloatTensor(w_tmp.shape).normal_(loc_9,scale_9)
                                    rram_error_10 = torch.cuda.FloatTensor(w_tmp.shape).normal_(loc_10,scale_10)
                                    rram_error_11 = torch.cuda.FloatTensor(w_tmp.shape).normal_(loc_11,scale_11)
                                    rram_error_12 = torch.cuda.FloatTensor(w_tmp.shape).normal_(loc_12,scale_12)
                                    rram_error_13 = torch.cuda.FloatTensor(w_tmp.shape).normal_(loc_13,scale_13)
                                    rram_error_14 = torch.cuda.FloatTensor(w_tmp.shape).normal_(loc_14,scale_14)
                                    rram_error_15 = torch.cuda.FloatTensor(w_tmp.shape).normal_(loc_15,scale_15)
                                    # rram_error_0_p = torch.cuda.FloatTensor(w_tmp_p.shape).normal_(loc, scale)
                                    # rram_error_0_n = torch.cuda.FloatTensor(w_tmp_n.shape).normal_(loc, scale)
                                        
                                    w_tmp = torch.where(w_tmp==0, rram_error_0, w_tmp)
                                    w_tmp = torch.where(w_tmp==1, rram_error_1, w_tmp)
                                    w_tmp = torch.where(w_tmp==2, rram_error_2, w_tmp)
                                    w_tmp = torch.where(w_tmp==3, rram_error_3, w_tmp)
                                    w_tmp = torch.where(w_tmp==4, rram_error_4, w_tmp)
                                    w_tmp = torch.where(w_tmp==5, rram_error_5, w_tmp)
                                    w_tmp = torch.where(w_tmp==6, rram_error_6, w_tmp)
                                    w_tmp = torch.where(w_tmp==7, rram_error_7, w_tmp)
                                    w_tmp = torch.where(w_tmp==8, rram_error_8, w_tmp)
                                    w_tmp = torch.where(w_tmp==9, rram_error_9, w_tmp)
                                    w_tmp = torch.where(w_tmp==10, rram_error_10, w_tmp)
                                    w_tmp = torch.where(w_tmp==11, rram_error_11, w_tmp)
                                    w_tmp = torch.where(w_tmp==12, rram_error_12, w_tmp)
                                    w_tmp = torch.where(w_tmp==13, rram_error_13, w_tmp)
                                    w_tmp = torch.where(w_tmp==14, rram_error_14, w_tmp)
                                    w_tmp = torch.where(w_tmp==15, rram_error_15, w_tmp)

                                    w_tmp = torch.where(w_tmp==-1, -rram_error_1, w_tmp)
                                    w_tmp = torch.where(w_tmp==-2, -rram_error_2, w_tmp)
                                    w_tmp = torch.where(w_tmp==-3, -rram_error_3, w_tmp)
                                    w_tmp = torch.where(w_tmp==-4, -rram_error_4, w_tmp)
                                    w_tmp = torch.where(w_tmp==-5, -rram_error_5, w_tmp)
                                    w_tmp = torch.where(w_tmp==-6, -rram_error_6, w_tmp)
                                    w_tmp = torch.where(w_tmp==-7, -rram_error_7, w_tmp)
                                    w_tmp = torch.where(w_tmp==-8, -rram_error_8, w_tmp)
                                    w_tmp = torch.where(w_tmp==-9, -rram_error_9, w_tmp)
                                    w_tmp = torch.where(w_tmp==-10, -rram_error_10, w_tmp)
                                    w_tmp = torch.where(w_tmp==-11, -rram_error_11, w_tmp)
                                    w_tmp = torch.where(w_tmp==-12, -rram_error_12, w_tmp)
                                    w_tmp = torch.where(w_tmp==-13, -rram_error_13, w_tmp)
                                    w_tmp = torch.where(w_tmp==-14, -rram_error_14, w_tmp)
                                    w_tmp = torch.where(w_tmp==-15, -rram_error_15, w_tmp)

                                res_tmp = F.conv2d(a_tmp.float(), w_tmp.float(), bias=None, stride=self.wrapped_module.stride, padding=0)


                                res_sliced_tmp = res_sliced_tmp + (res_tmp<<(ba*2)).float()#!!!!!!!


                    if(R_on):

                        res_sliced_tmp /= loc_1
                        res_sliced_tmp = torch.round(res_sliced_tmp)

                    res_sliced = res_sliced + res_sliced_tmp


                if self.conv_q.zero_point != 0:

                    in_tmp_b = torch.ones_like(in_a)*self.conv_q.zero_point
                    res_tmp_b = F.conv2d(in_tmp_b, weight_tmp, stride = self.wrapped_module.stride, padding = self.wrapped_module.padding)
                    res_sliced = res_sliced + res_tmp_b
                
                res_sliced = res_sliced / scale_wa
                if self.wrapped_module.bias is not None:
                    res_sliced = res_sliced + self.wrapped_module.bias.reshape(1, -1, 1, 1)

                return res_sliced
            else:#����kernel
                para = paranumber
                res2 = F.linear(input_tmp, weight_tmp, bias=None)
                paralevel = self.wrapped_module.in_features // para
                last_element = self.wrapped_module.in_features % para
                for cib in range(0, paralevel):
                    res_sliced_tmp = torch.cuda.FloatTensor(res.shape).fill_(0)

                    in_cis = input_tmp[:, cib*para:cib*para+para]
                    w_cis = weight_tmp[:, cib*para:cib*para+para] 

                    for ba in range(0, bit_activ // 2):
                        w_tmp = w_cis
                        w_tmp = w_tmp.float()

                        # print('w_tmp',w_tmp)

                        if(R_on):
                            rram_error_0 = torch.cuda.FloatTensor(w_tmp.shape).normal_(loc_0,scale_0)
                            rram_error_1 = torch.cuda.FloatTensor(w_tmp.shape).normal_(loc_1,scale_1)
                            rram_error_2 = torch.cuda.FloatTensor(w_tmp.shape).normal_(loc_2,scale_2)
                            rram_error_3 = torch.cuda.FloatTensor(w_tmp.shape).normal_(loc_3,scale_3)
                            rram_error_4 = torch.cuda.FloatTensor(w_tmp.shape).normal_(loc_4,scale_4)
                            rram_error_5 = torch.cuda.FloatTensor(w_tmp.shape).normal_(loc_5,scale_5)
                            rram_error_6 = torch.cuda.FloatTensor(w_tmp.shape).normal_(loc_6,scale_6)
                            rram_error_7 = torch.cuda.FloatTensor(w_tmp.shape).normal_(loc_7,scale_7)
                            rram_error_8 = torch.cuda.FloatTensor(w_tmp.shape).normal_(loc_8,scale_8)
                            rram_error_9 = torch.cuda.FloatTensor(w_tmp.shape).normal_(loc_9,scale_9)
                            rram_error_10 = torch.cuda.FloatTensor(w_tmp.shape).normal_(loc_10,scale_10)
                            rram_error_11 = torch.cuda.FloatTensor(w_tmp.shape).normal_(loc_11,scale_11)
                            rram_error_12 = torch.cuda.FloatTensor(w_tmp.shape).normal_(loc_12,scale_12)
                            rram_error_13 = torch.cuda.FloatTensor(w_tmp.shape).normal_(loc_13,scale_13)
                            rram_error_14 = torch.cuda.FloatTensor(w_tmp.shape).normal_(loc_14,scale_14)
                            rram_error_15 = torch.cuda.FloatTensor(w_tmp.shape).normal_(loc_15,scale_15)

                                
                            w_tmp = torch.where(w_tmp==0, rram_error_0, w_tmp)
                            w_tmp = torch.where(w_tmp==1, rram_error_1, w_tmp)
                            w_tmp = torch.where(w_tmp==2, rram_error_2, w_tmp)
                            w_tmp = torch.where(w_tmp==3, rram_error_3, w_tmp)
                            w_tmp = torch.where(w_tmp==4, rram_error_4, w_tmp)
                            w_tmp = torch.where(w_tmp==5, rram_error_5, w_tmp)
                            w_tmp = torch.where(w_tmp==6, rram_error_6, w_tmp)
                            w_tmp = torch.where(w_tmp==7, rram_error_7, w_tmp)
                            w_tmp = torch.where(w_tmp==8, rram_error_8, w_tmp)
                            w_tmp = torch.where(w_tmp==9, rram_error_9, w_tmp)
                            w_tmp = torch.where(w_tmp==10, rram_error_10, w_tmp)
                            w_tmp = torch.where(w_tmp==11, rram_error_11, w_tmp)
                            w_tmp = torch.where(w_tmp==12, rram_error_12, w_tmp)
                            w_tmp = torch.where(w_tmp==13, rram_error_13, w_tmp)
                            w_tmp = torch.where(w_tmp==14, rram_error_14, w_tmp)
                            w_tmp = torch.where(w_tmp==15, rram_error_15, w_tmp)

                            w_tmp = torch.where(w_tmp==-1, -rram_error_1, w_tmp)
                            w_tmp = torch.where(w_tmp==-2, -rram_error_2, w_tmp)
                            w_tmp = torch.where(w_tmp==-3, -rram_error_3, w_tmp)
                            w_tmp = torch.where(w_tmp==-4, -rram_error_4, w_tmp)
                            w_tmp = torch.where(w_tmp==-5, -rram_error_5, w_tmp)
                            w_tmp = torch.where(w_tmp==-6, -rram_error_6, w_tmp)
                            w_tmp = torch.where(w_tmp==-7, -rram_error_7, w_tmp)
                            w_tmp = torch.where(w_tmp==-8, -rram_error_8, w_tmp)
                            w_tmp = torch.where(w_tmp==-9, -rram_error_9, w_tmp)
                            w_tmp = torch.where(w_tmp==-10, -rram_error_10, w_tmp)
                            w_tmp = torch.where(w_tmp==-11, -rram_error_11, w_tmp)
                            w_tmp = torch.where(w_tmp==-12, -rram_error_12, w_tmp)
                            w_tmp = torch.where(w_tmp==-13, -rram_error_13, w_tmp)
                            w_tmp = torch.where(w_tmp==-14, -rram_error_14, w_tmp)
                            w_tmp = torch.where(w_tmp==-15, -rram_error_15, w_tmp)

                        a_tmp = (in_cis.int() & (3<<(ba*2))) >> (ba*2)
                        a_nz = (a_tmp!=0).sum(dim=1, keepdim=True) 
                        
                        res_tmp = F.linear(a_tmp.float(), w_tmp.float(), bias=None)


                        res_sliced_tmp = res_sliced_tmp + (res_tmp<<(2*ba)).float()

                    if(R_on):
                        res_sliced_tmp /= loc_1

                        res_sliced_tmp = torch.round(res_sliced_tmp)
                    res_sliced = res_sliced + res_sliced_tmp
                
                if(last_element != 0):
                    res_sliced_tmp = torch.cuda.FloatTensor(res.shape).fill_(0)

                    in_cis = input_tmp[:, paralevel*para:paralevel*para+last_element]
                    w_cis = weight_tmp[:, paralevel*para:paralevel*para+last_element]
                    for ba in range(0, bit_activ // 2):
                        w_tmp = w_cis
                        w_tmp = w_tmp.float()

                        # print('w_tmp',w_tmp)

                        if(R_on):
                            rram_error_0 = torch.cuda.FloatTensor(w_tmp.shape).normal_(loc_0,scale_0)
                            rram_error_1 = torch.cuda.FloatTensor(w_tmp.shape).normal_(loc_1,scale_1)
                            rram_error_2 = torch.cuda.FloatTensor(w_tmp.shape).normal_(loc_2,scale_2)
                            rram_error_3 = torch.cuda.FloatTensor(w_tmp.shape).normal_(loc_3,scale_3)
                            rram_error_4 = torch.cuda.FloatTensor(w_tmp.shape).normal_(loc_4,scale_4)
                            rram_error_5 = torch.cuda.FloatTensor(w_tmp.shape).normal_(loc_5,scale_5)
                            rram_error_6 = torch.cuda.FloatTensor(w_tmp.shape).normal_(loc_6,scale_6)
                            rram_error_7 = torch.cuda.FloatTensor(w_tmp.shape).normal_(loc_7,scale_7)
                            rram_error_8 = torch.cuda.FloatTensor(w_tmp.shape).normal_(loc_8,scale_8)
                            rram_error_9 = torch.cuda.FloatTensor(w_tmp.shape).normal_(loc_9,scale_9)
                            rram_error_10 = torch.cuda.FloatTensor(w_tmp.shape).normal_(loc_10,scale_10)
                            rram_error_11 = torch.cuda.FloatTensor(w_tmp.shape).normal_(loc_11,scale_11)
                            rram_error_12 = torch.cuda.FloatTensor(w_tmp.shape).normal_(loc_12,scale_12)
                            rram_error_13 = torch.cuda.FloatTensor(w_tmp.shape).normal_(loc_13,scale_13)
                            rram_error_14 = torch.cuda.FloatTensor(w_tmp.shape).normal_(loc_14,scale_14)
                            rram_error_15 = torch.cuda.FloatTensor(w_tmp.shape).normal_(loc_15,scale_15)
                            # rram_error_0_p = torch.cuda.FloatTensor(w_tmp_p.shape).normal_(loc, scale)
                            # rram_error_0_n = torch.cuda.FloatTensor(w_tmp_n.shape).normal_(loc, scale)
                                
                            w_tmp = torch.where(w_tmp==0, rram_error_0, w_tmp)
                            w_tmp = torch.where(w_tmp==1, rram_error_1, w_tmp)
                            w_tmp = torch.where(w_tmp==2, rram_error_2, w_tmp)
                            w_tmp = torch.where(w_tmp==3, rram_error_3, w_tmp)
                            w_tmp = torch.where(w_tmp==4, rram_error_4, w_tmp)
                            w_tmp = torch.where(w_tmp==5, rram_error_5, w_tmp)
                            w_tmp = torch.where(w_tmp==6, rram_error_6, w_tmp)
                            w_tmp = torch.where(w_tmp==7, rram_error_7, w_tmp)
                            w_tmp = torch.where(w_tmp==8, rram_error_8, w_tmp)
                            w_tmp = torch.where(w_tmp==9, rram_error_9, w_tmp)
                            w_tmp = torch.where(w_tmp==10, rram_error_10, w_tmp)
                            w_tmp = torch.where(w_tmp==11, rram_error_11, w_tmp)
                            w_tmp = torch.where(w_tmp==12, rram_error_12, w_tmp)
                            w_tmp = torch.where(w_tmp==13, rram_error_13, w_tmp)
                            w_tmp = torch.where(w_tmp==14, rram_error_14, w_tmp)
                            w_tmp = torch.where(w_tmp==15, rram_error_15, w_tmp)

                            w_tmp = torch.where(w_tmp==-1, -rram_error_1, w_tmp)
                            w_tmp = torch.where(w_tmp==-2, -rram_error_2, w_tmp)
                            w_tmp = torch.where(w_tmp==-3, -rram_error_3, w_tmp)
                            w_tmp = torch.where(w_tmp==-4, -rram_error_4, w_tmp)
                            w_tmp = torch.where(w_tmp==-5, -rram_error_5, w_tmp)
                            w_tmp = torch.where(w_tmp==-6, -rram_error_6, w_tmp)
                            w_tmp = torch.where(w_tmp==-7, -rram_error_7, w_tmp)
                            w_tmp = torch.where(w_tmp==-8, -rram_error_8, w_tmp)
                            w_tmp = torch.where(w_tmp==-9, -rram_error_9, w_tmp)
                            w_tmp = torch.where(w_tmp==-10, -rram_error_10, w_tmp)
                            w_tmp = torch.where(w_tmp==-11, -rram_error_11, w_tmp)
                            w_tmp = torch.where(w_tmp==-12, -rram_error_12, w_tmp)
                            w_tmp = torch.where(w_tmp==-13, -rram_error_13, w_tmp)
                            w_tmp = torch.where(w_tmp==-14, -rram_error_14, w_tmp)
                            w_tmp = torch.where(w_tmp==-15, -rram_error_15, w_tmp)

                        a_tmp = (in_cis.int() & (3<<(ba*2))) >> (ba*2)
                        a_nz = (a_tmp!=0).sum(dim=1, keepdim=True) 
                        
                        res_tmp = F.linear(a_tmp.float(), w_tmp.float(), bias=None)

                        res_sliced_tmp = res_sliced_tmp+(res_tmp<<(2*ba)).float()

                    if(R_on):
                        res_sliced_tmp /= loc_1

                        res_sliced_tmp = torch.round(res_sliced_tmp)
                    res_sliced = res_sliced + res_sliced_tmp
                res_sliced = res_sliced / scale_wa
                res_sliced = res_sliced + self.wrapped_module.bias

                return res_sliced
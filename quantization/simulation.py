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
    def __init__(self, wrapped_module, num_bits, init_act_clip_val, mode, ema_decay):
        super(ConvQuantizationWrapper, self).__init__()
        #self.conv_q = ConvClippedLinearQuantization(num_bits, init_act_clip_val, ema_decay, dequantize=True,
                                             #inplace=getattr(wrapped_module, 'inplace', False))
        self.conv_q = FakeLinearQuantization(num_bits, mode, ema_decay, dequantize=True,
                                             inplace=getattr(wrapped_module, 'inplace', False))
        self.wrapped_module = wrapped_module
        self.weight_scale = 0
        self.activ_scale = 0
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
        # torch.set_printoptions(profile="full")
        # mode = "bit_slice"
        # mode = "bit_kernel"
        # mode = "sparate_pn_kernel"
        # mode = "4bit_rram"
        # mode = "varia_pn_kernel"
        # mode = "fast_2mod"
        # mode = "fast_4bit"
        mode = "normal"
        # print(mode)
        # mode = "lenet11"
        #noise_type = "LSB*LSB"
        #RRAM
        r_ratio = 1000
        high_condu = 1/1000 #1
        low_condu =  high_condu/r_ratio #0
        low_condu =  0 #0
        # R_on = True
        # R_SAF = True
        R_on = False
        R_SAF = False
        R_off = False


        #hrs_noise = 0.005
        #hrs_mean = 1/32.3
        #lrs_noise = 0.23

        if mode == "normal":
            res = self.conv_q(*input)
            res = self.wrapped_module(res)
            return res
        elif mode == "lenet11":
            if hasattr(self.wrapped_module, 'kernel_size'):
                print(self.wrapped_module)
                res = self.conv_q(*input)
                res = self.wrapped_module(res)
                return res
            elif (self.wrapped_module.in_features == 64):
                a = get_hn_r()
                print('111',a)
                scale_w = self.wrapped_module.weight_scale
                scale_a = self.conv_q.scale
                scale_wa = scale_w * scale_a
                bit_weight = 1
                bit_activ = 1
                input_tmp = torch.round(input[0] * scale_a - self.conv_q.zero_point)
                input_tmp = torch.clamp(input_tmp, 0, (2**bit_activ)-1)
                weight_tmp = torch.round(self.wrapped_module.weight*scale_w)
                ce = self.wrapped_module.weight*scale_w
                in_a = self.conv_q(*input)
                res = self.wrapped_module(in_a)
                res_sliced = torch.cuda.FloatTensor(res.shape).fill_(0) 
                res_sliced_tmp = torch.cuda.FloatTensor(res.shape).fill_(0)

                para = 4
                res2 = F.linear(input_tmp, weight_tmp, bias=None)
                paralevel = self.wrapped_module.in_features // para
                last_element = self.wrapped_module.in_features % para
                print(last_element)
                for cib in range(0, paralevel):
                    in_cis = input_tmp[:, cib*para:cib*para+para]
                    w_cis = weight_tmp[:, cib*para:cib*para+para] 
                    if(R_on):
                    # if(R_off):
                        for i in range(0, w_cis.size(0)):
                            for j in range(0, w_cis.size(1)):
                                if(w_cis[i,j] == 0): w_cis[i,j] = get_ln_r()
                                if(w_cis[i,j] == 1): w_cis[i,j] = get_hn_r()
                                if(w_cis[i,j] == -1): w_cis[i,j] = -(get_hn_r())
                    res_tmp = F.linear(in_cis.float(), w_cis.float(), bias=None)
                    res_sliced = res_sliced+res_tmp.float()
                
                if(last_element != 0):
                    # print('c & r:', paralevel*para, paralevel*para+last_element)
                    in_cis = input_tmp[:, paralevel*para:paralevel*para+last_element]
                    w_cis = weight_tmp[:, paralevel*para:paralevel*para+last_element]
                    if(R_on):
                    # if(R_off):
                        print(w_cis)
                        print(w_cis.size(0), w_cis.size(1))
                        for i in range(0, w_cis.size(0)):
                            for j in range(0, w_cis.size(1)):
                                if(w_cis[i,j] == 0): w_cis[i,j] = get_ln_r()
                                if(w_cis[i,j] == 1): w_cis[i,j] = get_hn_r()
                                if(w_cis[i,j] == -1): w_cis[i,j] = -(get_hn_r())
                        print(w_cis)
                    res_tmp = F.linear(in_cis.float(), w_cis.float(), bias=None)
                    res_sliced = res_sliced+res_tmp.float()
                
                print('res_sliced_befor_adc', res_sliced)
                bit_answer = (2**bit_activ-1)*(2**(bit_weight-1)-1)
                # if(R_off):
                if(R_on):
                    res_sliced /= 2050
                    res_sliced = torch.round(res_sliced)
                    print('res_sliced_R_ON', res_sliced)
                    
                     

                #print((res_sliced-res2).std())
                #exit(0)
                res_sliced = res_sliced / scale_wa
                res_sliced = res_sliced + self.wrapped_module.bias
                print('END3')
                print('res_sliced', res_sliced.size())
                print(res_sliced)
                return res_sliced
            else:
                print(self.wrapped_module)
                res = self.conv_q(*input)
                res = self.wrapped_module(res)
                return res

            return res
        elif mode == "bit_slice":
            torch.backends.cudnn.deterministic = True
            thre = 7 
            ad_limit = 15
            bit_weight = 4
            bit_activ = 4
            # para = 18
            paranumber = 2
            print('paranumber', paranumber)

            scale_w = self.wrapped_module.weight_scale
            scale_a = self.conv_q.scale
            scale_wa = scale_w * scale_a
            # print('****************************************************************************************')
            # print('w,a,wa',scale_w, scale_a, scale_wa)
            print(self.wrapped_module)
            ber = [0,8.61527E-13,4.18694E-07,6.1348E-05,0.000751256,0.004207439,0.012610766,0.024916498,0.045734124,0.072046802,0.104019711,0.140023727,0.174666477,0.201189703,0.22912297,0.112170035,1]
            # ber = [i * 2 for i in ber]
            # ber = [0.0] * 16
            #ber = [0.25*1e-4]+[1e-4]*16
            ber = torch.tensor(ber).cuda()
            # print('ber',ber)

            # print('input0',input[0])
            # print('input_before_round',input[0] * scale_a - self.conv_q.zero_point)
            input_tmp = torch.round(input[0] * scale_a - self.conv_q.zero_point)
            # print('input',input_tmp)
            input_tmp = torch.clamp(input_tmp, 0, (2**bit_activ)-1)
            # input_t2 = torch.cat((input_tmp.int() & 3, input_tmp.int()>>2), dim = 0)
            # restt = self.conv_q(*input)
            # print('input_tmp', input_tmp)
            # print('input_tmp.size', input_tmp.size())
            # print('restt', restt)

            # nz = (input_t2.view(-1) == 0).sum()*1.0/input_t2.numel()
            # nz2 = (input_tmp.view(-1) == 0).sum()*1.0/input_tmp.numel()
            # nz3 = ((input_tmp.int()&3).view(-1) == 0).sum()*1.0/input_tmp.numel()
            # print("nz3:", nz2)
            #print("nz:", nz)
            #exit(0)
            #print(input_tmp.unique())
            #sexit(0)
            weight_tmp = torch.round(self.wrapped_module.weight*scale_w)
            # print('weight_tmp', weight_tmp.size())
            # print(weight_tmp)
            # a = input_tmp.cpu().detach().numpy()
            # print(a)
            # np.savez('/harddisk/yuejs/train+prune+quant/testdata/input',input_tmp.cpu().detach().numpy())
            # np.savez('/harddisk/yuejs/train+prune+quant/testdata/weight',weight_tmp.cpu().detach().numpy())
            # exit(0)
            # a = torch.cuda.FloatTensor(weight_tmp.shape).fill_(0.0000001)
            # print('a',type(a),a)
            # print('weight_tmp',type(weight_tmp),weight_tmp)
            # exit(0)
            
            # print('weight', self.wrapped_module.weight)
            ce = self.wrapped_module.weight*scale_w
            # print('max',torch.max(ce))
            # print('weight*scale',self.wrapped_module.weight*scale_w)
            # print('weight_tmp',weight_tmp.size())
            wnz = (weight_tmp.view(-1) == 0).sum()*1.0/weight_tmp.numel()
            #print("wnz:",wnz)
            in_a = self.conv_q(*input)
            # print('in_a',in_a)
            res = self.wrapped_module(in_a)
            # print('res',res.size(),res)
            #return res`    `
            res_sliced = torch.cuda.FloatTensor(res.shape).fill_(0) 
            res_sliced_tmp = torch.cuda.FloatTensor(res.shape).fill_(0)
            # print('res_sliced_ideal',res_sliced.size())
            # exit(0)
            # print(res_sliced)
            # print(self.wrapped_module)
            #exit(0)


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
                # print('last_number', last_number)
                # print('paralevel', paralevel)
                # print('lastchannel', last_channel)
                # if(paralevel==0): paralevel += 1
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
                            for bw in range(0, bit_weight):
                                # print('ba & bw', ba, bw)
                                # print('w_tmp_size',)
                                w_tmp = ((w_cis[:,:,krow:krow+1,kcol:kcol+1]+(1<<(bit_weight))).int() & (1<<bw)) >> bw
                                #ȡ��1����λ��weight,���ֱ�ȡ��1bit
                                w_tmp = w_tmp.float()
                                

                                a_tmp = (a_pad.int() & (3<<(ba*2))) >> (ba*2)#����λȡ�����ֵ
                                # res_tmp = F.conv2d(a_tmp.float(), w_tmp.float(), bias=None, stride=self.wrapped_module.stride, padding=0)
                                # print('before', res_tmp)
                                # print(w_tmp.size())
                                # exit(0)
                                if(R_on):
                                    # w_tt = w_tmp
                                    # w_tmp[w_tmp==0] = low_condu
                                    # w_tmp[w_tmp==1] = high_condu
                                    # w_tt[w_tt==0] = 0.000001
                                    # w_tt[w_tt==1] = 0.001
                                    # print(w_tmp[0,:,:,:])
                                    for i in range(0, w_tmp.size(0)):
                                    #     # for j in range(0, w_tmp.size(1)):
                                    #         # print(w_tmp[i,j,:,:])
                                    #     # print(w_tmp)
                                    #     # print('1',w_tmp[i,:,:,:].size(),w_tmp[i,:,:,:])
                                    #     # print('2',w_tmp[i,0,0,0].size(),w_tmp[i,0,0,0])
                                        if(w_tmp[i,:,:,:] == 0): w_tmp[i,:,:,:] = get_ln_r()
                                        if(w_tmp[i,:,:,:] == 1): w_tmp[i,:,:,:] = get_hn_r()
                                        # print(w_tmp[i,:,:,:])
                                    #         # if(w_cis[i,j] == -1): w_cis[i,j] = -(get_hn_r())
                                    # print('w_tt', w_tt)
                                    # print('w_tmp',w_tmp)
                                    # print(w_tt.size(), w_tmp.size())
                                    # exit(0)
                                            

                                
                                # print(a_tmp)
                                # print(w_tmp)
                                # exit(0)
                                res_tmp = F.conv2d(a_tmp.float(), w_tmp.float(), bias=None, stride=self.wrapped_module.stride, padding=0)
                                # print('after', res_tmp)

                                if bw == bit_weight-1:
                                    # print('-res',res_tmp)
                                    res_tmp = -res_tmp

                                res_sliced_tmp = res_sliced_tmp + (res_tmp<<(ba*2+bw)).float()#!!!!!!!
                    # bit_answer = (2**bit_activ-1)*(2**(bit_weight-1)-1)
                    # if(R_off):
                    if(R_on):
                        # print(res_sliced_tmp)
                        res_sliced_tmp /= 2050
                        # res_sliced_tmp /= 0.001
                        res_sliced_tmp = torch.round(res_sliced_tmp)
                        # print(res_sliced_tmp)
                    # exit(0)
                    res_sliced = res_sliced + res_sliced_tmp

                # for cib in range(0, paralevel):
                #     res_sliced_tmp = torch.cuda.FloatTensor(res.shape).fill_(0)
                #     # print('��{}��'.format(cib))
                #     # print('in_size & w_size',input_tmp.size(), weight_tmp.size())
                #     # print('c & r:', cib*para, cib*para+para)
                #     arrayrow = min(paranumber, kernel_size * self.wrapped_module.in_channels - cib*paranumber)
                #     print('arrayrow', arrayrow)
                #     first_number = (kernel_size - remain_number) % kernel_size 
                #     middle_number = ((arrayrow - first_number) // kernel_size) * kernel_size
                #     remain_number = arrayrow - first_number - middle_number
                #     # print('f,m,r',first_number, middle_number, remain_number)

                #     # in_cis = input_tmp[:, cib*para:cib*para+para, :, :]
                #     # w_cis = weight_tmp[:, cib*para:cib*para+para, :, :]
                #     # print('in_cis_size & w_cis_size',in_cis.size(), w_cis.size())
                #     # print('in_cis', in_cis.size())
                #     # print('kenel_size',self.wrapped_module.kernel_size)
                #     if(first_number != 0):
                #         print('f_n', first_number)
                #         print(tmp_channel)
                #         # print('begin_first_channel')
                #         in_cis = input_tmp[:, tmp_channel:tmp_channel+1, :, :]
                #         w_cis = weight_tmp[:, tmp_channel:tmp_channel+1, :, :]
                #         # print('first_channel', tmp_channel)
                #         print(in_cis.size(), w_cis.size())
                #         tmp_channel += 1

                #         # for num_tmp in range(kernel_size-first_number, kernel_size):
                #         for num_tmp in range(first_number, kernel_size):
                #             # print(num_tmp)
                #             krow = num_tmp // self.wrapped_module.kernel_size[1]
                #             kcol = num_tmp % self.wrapped_module.kernel_size[0]
                #             # print('padding',self.wrapped_module.padding)
                #             m = nn.ZeroPad2d((self.wrapped_module.padding[0] - kcol, self.wrapped_module.padding[0]-(self.wrapped_module.kernel_size[1]-kcol-1), self.wrapped_module.padding[1] - krow, self.wrapped_module.padding[1]-(self.wrapped_module.kernel_size[0]-krow-1)))
                #             a_pad = m(in_cis)

                #             for ba in range(0, bit_activ//2):#��֧��2������������
                #                 for bw in range(0, bit_weight):
                #                     # print('ba & bw', ba, bw)
                #                     # print('w_tmp_size',)
                #                     w_tmp = ((w_cis[:,:,krow:krow+1,kcol:kcol+1]+(1<<(bit_weight))).int() & (1<<bw)) >> bw
                #                     #ȡ��1����λ��weight,���ֱ�ȡ��1bit
                #                     w_tmp = w_tmp.float()
                #                     # print('w_tmp',type(w_tmp),w_tmp)
                #                     # w_tmp[w_tmp==0] = 0.000001
                #                     # w_tmp[w_tmp==1] = 0.001
                #                     # print('w_tmp',type(w_tmp),w_tmp)
                #                     # exit(0)
                #                     # if(R_off):
                                    

                #                     a_tmp = (a_pad.int() & (3<<(ba*2))) >> (ba*2)#����λȡ�����ֵ
                #                     # res_tmp = F.conv2d(a_tmp.float(), w_tmp.float(), bias=None, stride=self.wrapped_module.stride, padding=0)
                #                     # print('before', res_tmp)
                #                     if(R_on):
                #                         w_tmp[w_tmp==0] = low_condu
                #                         w_tmp[w_tmp==1] = high_condu
                                    
                #                     # print(a_tmp)
                #                     # print(w_tmp)
                #                     res_tmp = F.conv2d(a_tmp.float(), w_tmp.float(), bias=None, stride=self.wrapped_module.stride, padding=0)
                #                     # print('after', res_tmp)

                #                     if bw == bit_weight-1:
                #                         # print('-res',res_tmp)
                #                         res_tmp = -res_tmp

                #                     res_sliced_tmp = res_sliced_tmp + (res_tmp<<(ba*2+bw)).float()#!!!!!!!
                #     if(middle_number != 0):
                #         mid_channel = int(middle_number/kernel_size)
                #         # print('mid_channel', mid_channel)
                #         # print('begin_middle_channel')
                #         # print(type(mid_channel))
                #         in_cis = input_tmp[:, tmp_channel:tmp_channel+mid_channel, :, :]
                #         w_cis = weight_tmp[:, tmp_channel:tmp_channel+mid_channel, :, :]
                #         tmp_channel = tmp_channel + mid_channel

                #         for krow in range(0, self.wrapped_module.kernel_size[0]):
                #             for kcol in range(0, self.wrapped_module.kernel_size[1]):
                #                 # print('padding',self.wrapped_module.padding)
                #                 m = nn.ZeroPad2d((self.wrapped_module.padding[0] - kcol, self.wrapped_module.padding[0]-(self.wrapped_module.kernel_size[1]-kcol-1), self.wrapped_module.padding[1] - krow, self.wrapped_module.padding[1]-(self.wrapped_module.kernel_size[0]-krow-1)))
                #                 # print('pad_m',m)
                #                 a_pad = m(in_cis)
                #                 # a_pad = in_cis
                #                 # print('m',m)
                #                 # print('in_cis', in_cis.size())
                #                 # print('a_pad.size',a_pad.size())
                #                 # print('w_cis',w_cis[:,:,krow:krow+1,kcol:kcol+1])
                #                 # print('input:',a_pad)
                #                 # print('w:', w_cis)
                #                 for ba in range(0, bit_activ//2):#��֧��2������������
                #                     for bw in range(0, bit_weight):
                #                         # print('ba & bw', ba, bw)
                #                         # print('w_tmp_size',)
                #                         # print('bw:',bw)
                #                         w_tmp = ((w_cis[:,:,krow:krow+1,kcol:kcol+1]+(1<<(bit_weight))).int() & (1<<bw)) >> bw
                #                         # print('w_cis', w_cis[:,:,krow:krow+1,kcol:kcol+1])
                #                         # print('w_cis+1', w_cis[0,0,krow:krow+1,kcol:kcol+1]+(1<<(bit_weight)))
                #                         # print('w_tmp', w_tmp[0,0,krow:krow+1,kcol:kcol+1])
                #                         # w_tmp = ((w_cis[:,:,krow,kcol]+(1<<(bit_weight))).int() & (1<<bw)) >> bw
                #                         # print('w_tmp_size', w_tmp.size(), w_tmp)
                #                         #ȡ��1����λ��weight,���ֱ�ȡ��1bit
                #                         # print('w_tmp',w_tmp)
                #                         w_tmp = w_tmp.float()  
                #                         # print('bw%w', bw, w_tmp[0:10,:,:,:])                                  

                #                         a_tmp = (a_pad.int() & (3<<(ba*2))) >> (ba*2)#����λȡ�����ֵ
                #                         # print('a_tmp:', a_tmp)

                #                         # res_tmp = F.conv2d(a_tmp.float(), w_tmp.float(), bias=None, stride=self.wrapped_module.stride, padding=0)
                #                         # print('before', res_tmp)
                #                         if(R_on):
                #                             w_tmp[w_tmp==0] = low_condu
                #                             w_tmp[w_tmp==1] = high_condu
                                        
                #                         # print(a_tmp)
                #                         # print(w_tmp)

                #                         res_tmp = F.conv2d(a_tmp.float(), w_tmp.float(), bias=None, stride=self.wrapped_module.stride, padding=0)
                #                         # print('res_tmp:',res_tmp)
                #                         # print('after', bw,res_tmp)
                #                         # res_tmp = torch.where(res_tmp > ad_limit, 15*torch.ones_like(res_tmp), res_tmp)
                #                         # print(res_com)
                #                         # print(res_com==(res_tmp_0+res_tmp_1))
                #                         # print(res_com.size())
                #                         # print((res_tmp_0+res_tmp_1).size())

                                        

                #                         #res_tmp_2 = F.conv2d(a_tmp.float(), w_tmp.float(), bias=None, stride=self.wrapped_module.stride, padding=0)
                #                         #print(a_nz.shape)
                #                         #print(res_tmp.shape)
                #                         # truc_flag = (a_nz > thre) & (res_tmp > 15)

                #                         # ber_sampled = ber[res_tmp.long()]
                #                         # res_tmp += ber_sampled

                #                         # err_prob = torch.cuda.FloatTensor(res_tmp.shape).fill_(0)
                #                         # err_prob = torch.rand(size=res_tmp.shape, out=err_prob)
                #                         # err = torch.where(err_prob<ber_sampled, torch.ones_like(res_tmp), torch.zeros_like(res_tmp))
                #                         # err = torch.where(err_prob<ber_sampled/2, err, -err)
                #                         # print('res_tmp.long', res_tmp.long())
                #                         # print('ber_tmp',ber_tmp)
                #                         #res_tmp = res_tmp_0 + res_tmp_1
                #                         #res_tmp = res_tmp_2
                #                         if bw == bit_weight-1:
                #                             res_tmp = -res_tmp
                #                             # print('bw=3:',res_tmp)
                #                         # print('res_sliced:',res_sliced.size())
                #                         # print('res_tmp:',res_tmp.int().size())
                #                         # print(ba*2+bw)
                #                         # print((res_tmp.int()<<(ba*2+bw)).float().size())
                #                         # print('result:',(res_tmp<<(ba*2+bw)))
                #                         res_sliced_tmp = res_sliced_tmp + (res_tmp<<(ba*2+bw)).float()#!!!!!!!
                #             # print(res_sliced)
                #     # exit(0)
                #     if(remain_number != 0):
                #         # print('begin_remain_channel')
                #         in_cis = input_tmp[:, tmp_channel:tmp_channel+1, :, :]
                #         w_cis = weight_tmp[:, tmp_channel:tmp_channel+1, :, :]
                #         tmp_channel += 1

                #         for num_tmp in range(0, remain_number):
                #             krow = num_tmp // self.wrapped_module.kernel_size[1]
                #             kcol = num_tmp % self.wrapped_module.kernel_size[0]
                #             # print('padding',self.wrapped_module.padding)
                #             m = nn.ZeroPad2d((self.wrapped_module.padding[0] - kcol, self.wrapped_module.padding[0]-(self.wrapped_module.kernel_size[1]-kcol-1), self.wrapped_module.padding[1] - krow, self.wrapped_module.padding[1]-(self.wrapped_module.kernel_size[0]-krow-1)))
                #             a_pad = m(in_cis)

                #             for ba in range(0, bit_activ//2):#��֧��2������������
                #                 for bw in range(0, bit_weight):
                #                     # print('ba & bw', ba, bw)
                #                     # print('w_tmp_size',)
                #                     w_tmp = ((w_cis[:,:,krow:krow+1,kcol:kcol+1]+(1<<(bit_weight))).int() & (1<<bw)) >> bw
                #                     #ȡ��1����λ��weight,���ֱ�ȡ��1bit
                #                     w_tmp = w_tmp.float()

                #                     a_tmp = (a_pad.int() & (3<<(ba*2))) >> (ba*2)#����λȡ�����ֵ
                #                     # a_tmp = a_tmp.float()
                #                     # res_tmp = F.conv2d(a_tmp.float(), w_tmp.float(), bias=None, stride=self.wrapped_module.stride, padding=0)
                #                     # print('before', res_tmp)
                #                     if(R_on):
                #                         w_tmp[w_tmp==0] = low_condu
                #                         w_tmp[w_tmp==1] = high_condu
                                    
                #                     # a_tmp = a_tmp.to(torch.float32)
                #                     # w_tmp = w_tmp.to(torch.float32)
                #                     # print(a_tmp.size(), type(a_tmp))
                #                     # print(w_tmp.size(), type(w_tmp))
                #                     # res_tmp = F.conv2d(a_tmp.float(), w_tmp.float(), bias=None, stride=self.wrapped_module.stride, padding=0)
                                    
                #                     res_tmp = F.conv2d(a_tmp.float(), w_tmp.float(), bias=None, stride=self.wrapped_module.stride, padding=0)
                #                     # print('after', res_tmp)

                #                     if bw == bit_weight-1:
                #                         res_tmp = -res_tmp

                #                     res_sliced_tmp = res_sliced_tmp + (res_tmp<<(ba*2+bw)).float()#!!!!!!!

                #     # print('res_sliced_afterRon_beforeADC', type(res_sliced_tmp),res_sliced_tmp)
                #     bit_answer = (2**bit_activ-1)*(2**(bit_weight-1)-1)
                #     # if(R_off):
                #     if(R_on):
                #         # for i in range(-(bit_answer*paranumber), 0):
                #         #     res_sliced_tmp[((i*high_condu)<=res_sliced_tmp) & (res_sliced_tmp < (i+1)*high_condu)] = i
                #         # for i in range(0, bit_answer*paranumber+1):
                #         #     res_sliced_tmp[((i*high_condu)<=res_sliced_tmp) & (res_sliced_tmp < (i+1)*high_condu)] = i
                #         # # print('res_sliced_R_ON', res_sliced)
                #         res_sliced_tmp /= high_condu
                #         res_sliced_tmp = torch.round(res_sliced_tmp)
                        
                #     res_sliced = res_sliced + res_sliced_tmp
                    # print('res_sliced_R_ON', res_sliced)
                
                # exit(0)



                # if(last_channel != 0):
                #     # print('c & r:', paralevel*para, paralevel*para+last_channel)
                #     in_cis = input_tmp[:, paralevel*para:paralevel*para+last_channel, :, :]
                #     w_cis = weight_tmp[:, paralevel*para:paralevel*para+last_channel, :, :]
                #     for krow in range(0, self.wrapped_module.kernel_size[0]):
                #         for kcol in range(0, self.wrapped_module.kernel_size[1]):
                #             # print('padding',self.wrapped_module.padding)
                #             m = nn.ZeroPad2d((self.wrapped_module.padding[0] - kcol, self.wrapped_module.padding[0]-(self.wrapped_module.kernel_size[1]-kcol-1), self.wrapped_module.padding[1] - krow, self.wrapped_module.padding[1]-(self.wrapped_module.kernel_size[0]-krow-1)))
                #             # print('pad_m',m)
                #             a_pad = m(in_cis)
                #             for ba in range(0, bit_activ//2):
                #                 for bw in range(0, bit_weight):
                #                     w_tmp = ((w_cis[:,:,krow:krow+1,kcol:kcol+1]+(1<<(bit_weight))).int() & (1<<bw)) >> bw
                #                     a_tmp = (a_pad.int() & (3<<(ba*2))) >> (ba*2)
                #                     res_tmp = F.conv2d(a_tmp.float(), w_tmp.float(), bias=None, stride=self.wrapped_module.stride, padding=0)
                #                     # print('tmp:',res_tmp.size())
                                    
                #                     adc = False
                #                     if(adc):
                #                         ber_sampled = ber[res_tmp.long()]
                #                         err_prob = torch.cuda.FloatTensor(res_tmp.shape).fill_(0)
                #                         err_prob = torch.rand(size=res_tmp.shape, out=err_prob)
                                        
                #                         err = torch.where(err_prob<ber_sampled, torch.ones_like(res_tmp), torch.zeros_like(res_tmp))
                #                         err = torch.where(err_prob<ber_sampled/2, err, -err)
                #                         res_tmp = res_tmp+err


                #                     #print("pred:",err.std()**2)
                #                     #print("misc:",err.mean())
                #                     #print((err!=0).float().mean())
                #                     #print("theo:",ber_sampled.mean())
                #                     #exit(0)
                #                     if bw == bit_weight-1:
                #                         res_tmp = -res_tmp
                #                     res_sliced = res_sliced + (res_tmp.int()<<(ba*2+bw)).float()

                # print(res_sliced)

                if self.conv_q.zero_point != 0:
                    #print("self.conv_q.zero_point:", self.conv_q.zero_point)
                    in_tmp_b = torch.ones_like(in_a)*self.conv_q.zero_point
                    res_tmp_b = F.conv2d(in_tmp_b, weight_tmp, stride = self.wrapped_module.stride, padding = self.wrapped_module.padding)
                    res_sliced = res_sliced + res_tmp_b
                #res2 = F.conv2d(input_tmp+self.conv_q.zero_point, weight_tmp, padding=1)
                #print("wb:", (res2-res_sliced).std())
                
                res_sliced = res_sliced / scale_wa
                #print((res-res_sliced).std())
                #exit(0)
                #if (self.wrapped_module.bias is not None):
                #    print(self.wrapped_module.bias)
                #    exit(0)
                if self.wrapped_module.bias is not None:
                    res_sliced = res_sliced + self.wrapped_module.bias.reshape(1, -1, 1, 1)
                # print('END2')
                # print('res_sliced', res_sliced.size())
                # print(res_sliced)
                return res_sliced
            else:#����kernel
                #return res
                # exit(0)
                para = paranumber
                res2 = F.linear(input_tmp, weight_tmp, bias=None)
                paralevel = self.wrapped_module.in_features // para
                last_element = self.wrapped_module.in_features % para
                # print(last_element)
                for cib in range(0, paralevel):
                    res_sliced_tmp = torch.cuda.FloatTensor(res.shape).fill_(0)
                    #print(input_tmp.shape)
                    #print(weight_tmp.shape)
                    #exit(0)
                    # print('c & r:', cib*para, cib*para+para)
                    in_cis = input_tmp[:, cib*para:cib*para+para]
                    w_cis = weight_tmp[:, cib*para:cib*para+para] 
                    #maybe w_cis = weight_tmp[cib*para:cib*para+para, :] 
                    for ba in range(0, bit_activ // 2):
                        for bw in range(0, bit_weight):
                            w_tmp = ((w_cis+(1<<(bit_weight))).int() & (1<<bw)) >> bw
                            w_tmp = w_tmp.float()

                            # print('w_tmp',w_tmp)

                            if(R_on):
                            # if(R_off):
                                # w_tmp[w_tmp==0] = 0.000001
                                # w_tmp[w_tmp==1] = 0.001
                                for i in range(0, w_tmp.size(0)):
                                    for j in range(0, w_tmp.size(1)):
                                        if(w_tmp[i,j] == 0): w_tmp[i,j] = get_ln_r()
                                        if(w_tmp[i,j] == 1): w_tmp[i,j] = get_hn_r()
                                # #         # if(w_cis[i,j] == -1): w_cis[i,j] = -(get_hn_r())
                                # # print('w_tmp_Ron',w_tmp)

                            # w_tmp_0, w_tmp_1 = w_tmp.split(para//2, dim = 1)
                            a_tmp = (in_cis.int() & (3<<(ba*2))) >> (ba*2)
                            a_nz = (a_tmp!=0).sum(dim=1, keepdim=True) 
                            
                            res_tmp = F.linear(a_tmp.float(), w_tmp.float(), bias=None)
                            # a_tmp_0, a_tmp_1 = a_tmp.split(para//2, dim = 1)
                            # res_tmp_0 = F.linear(a_tmp_0.float(), w_tmp_0.float(), bias=None)
                            # res_tmp_1 = F.linear(a_tmp_1.float(), w_tmp_1.float(), bias=None)
                            # res_tmp_0 = torch.where(res_tmp_0>15, 15*torch.ones_like(res_tmp_0), res_tmp_0)
                            # res_tmp_1 = torch.where(res_tmp_1>15, 15*torch.ones_like(res_tmp_1), res_tmp_1)
                            # res_tmp = res_tmp_0 + res_tmp_1


                            if bw == bit_weight-1:
                                res_tmp = -res_tmp
                            # print('``````````````````````')
                            # print(res_sliced)
                            # print(res_tmp)
                            # print((res_tmp.int()<<(2*ba+bw)).float())
                            # print('``````````````````````')
                            # res_sliced = res_sliced+(res_tmp<<(2*ba+bw)).float()
                            res_sliced_tmp = res_sliced_tmp + (res_tmp<<(2*ba+bw)).float()
                    # bit_answer = (2**bit_activ-1)*(2**(bit_weight-1)-1)
                    # if(R_off):
                    if(R_on):
                        res_sliced_tmp /= 2050
                        # res_sliced_tmp /= 0.001
                        res_sliced_tmp = torch.round(res_sliced_tmp)
                    res_sliced = res_sliced + res_sliced_tmp
                
                if(last_element != 0):
                    res_sliced_tmp = torch.cuda.FloatTensor(res.shape).fill_(0)
                    # print('c & r:', paralevel*para, paralevel*para+last_element)
                    in_cis = input_tmp[:, paralevel*para:paralevel*para+last_element]
                    w_cis = weight_tmp[:, paralevel*para:paralevel*para+last_element]
                    for ba in range(0, bit_activ // 2):
                        for bw in range(0, bit_weight):
                            w_tmp = ((w_cis+(1<<(bit_weight))).int() & (1<<bw)) >> bw
                            w_tmp = w_tmp.float()

                            if(R_on):
                            # if(R_off):
                                # w_tmp[w_tmp==0] = 0.000001
                                # w_tmp[w_tmp==1] = 0.001
                                for i in range(0, w_tmp.size(0)):
                                    for j in range(0, w_tmp.size(1)):
                                        if(w_tmp[i,j] == 0): w_tmp[i,j] = get_ln_r()
                                        if(w_tmp[i,j] == 1): w_tmp[i,j] = get_hn_r()
                                #         # if(w_cis[i,j] == -1): w_cis[i,j] = -(get_hn_r())

                            # w_tmp_0, w_tmp_1 = w_tmp.split(para//2, dim = 1)
                            a_tmp = (in_cis.int() & (3<<(ba*2))) >> (ba*2)
                            a_nz = (a_tmp!=0).sum(dim=1, keepdim=True) 
                            
                            res_tmp = F.linear(a_tmp.float(), w_tmp.float(), bias=None)
                            if bw == bit_weight-1:
                                res_tmp = -res_tmp

                            res_sliced_tmp = res_sliced_tmp+(res_tmp<<(2*ba+bw)).float()
                    # bit_answer = (2**bit_activ-1)*(2**(bit_weight-1)-1)
                    # if(R_off):
                    if(R_on):
                        res_sliced_tmp /= 2050
                        # res_sliced_tmp /= 0.001
                        res_sliced_tmp = torch.round(res_sliced_tmp)
                    res_sliced = res_sliced + res_sliced_tmp
                
                # print('res_sliced_befor_adc', res_sliced)
                # bit_answer = (2**bit_activ-1)*(2**bit_weight-1)
                # if(R_on):
                #     for i in range(0, bit_answer*para):
                #         res_sliced[((i*high_condu)<=res_sliced) & (res_sliced < (i+1)*high_condu)] = i
                #     print('res_sliced_Ron', res_sliced)
                # bit_answer = (2**bit_activ-1)*(2**(bit_weight-1)-1)
                # # if(R_off):
                # if(R_on):
                # # if(R_off):
                #     # for i in range(-(bit_answer*paranumber), 0):
                #     #     res_sliced[((i*high_condu)<=res_sliced) & (res_sliced < (i+1)*high_condu)] = i
                #     # res_sliced[((-(paranumber*low_condu))<=res_sliced) & (res_sliced < paranumber*low_condu)] = 0
                #     # for i in range(1, bit_answer*paranumber+1):
                #     #     res_sliced[((i*high_condu)<=res_sliced) & (res_sliced < (i+1)*high_condu)] = i
                #     res_sliced /= 2050
                #     res_sliced = torch.round(res_sliced)
                    # print('res_sliced_R_ON', res_sliced)
                    
                     

                #print((res_sliced-res2).std())
                #exit(0)
                res_sliced = res_sliced / scale_wa
                res_sliced = res_sliced + self.wrapped_module.bias
                # print('END3')
                # print('res_sliced', res_sliced.size())
                # print(res_sliced)
                return res_sliced
        
            '''
            return res

            torch.set_printoptions(precision=8)
            print(self.wrapped_module.weight.shape)
            r_t = torch.tensor(0)
            for i in range(0, 64):
                test_res = (res-res_sliced)[0,i,:,:]
                if len(test_res.unique()) > 1:
                    for c in range(0, 3):
                        for x in range(0,3):
                            for y in range(0,2):
                                print("in_a(",c,",",x,",",y,"):", in_a[0,c,x,y])
                                print("w(",c,",",x,",",y,"):", self.wrapped_module.weight[i,c,x,y+1])
                                r_t = r_t + in_a[0,c,x,y]*self.wrapped_module.weight[i,c,x,y+1]

                    print(r_t)
                    print(res_sliced[0,i,1,0])
                    print(res[0,i,1,0])
                    print("channel:", i)
                    print(res.shape)
                    print(len(test_res.unique())) 
                    print(test_res.unique())
                    print(test_res)
                    exit(0)
            print("succ")
            '''

            exit(0)
        elif mode == "bit_kernel":
            torch.backends.cudnn.deterministic = True
            thre = 7 
            ad_limit = 15
            bit_weight = 4
            bit_activ = 4
            # para = 18
            paranumber = 1024
            loc=34.16666667
            scale=1.929166667
            # print(paranumber, loc, scale)

            scale_w = self.wrapped_module.weight_scale
            scale_a = self.conv_q.scale
            scale_wa = scale_w * scale_a
            # print('****************************************************************************************')
            # print('w,a,wa',scale_w, scale_a, scale_wa)
            # print(self.wrapped_module)
            ber = [0,8.61527E-13,4.18694E-07,6.1348E-05,0.000751256,0.004207439,0.012610766,0.024916498,0.045734124,0.072046802,0.104019711,0.140023727,0.174666477,0.201189703,0.22912297,0.112170035,1]
            ber = torch.tensor(ber).cuda()

            input_ori = torch.round(input[0] * scale_a - self.conv_q.zero_point)
            input_ori = torch.clamp(input_ori, 0, (2**bit_activ)-1)
            # print('input_ori.size', input_ori.size())
            weight_ori = torch.round(self.wrapped_module.weight*scale_w)
            
            # print('weight', self.wrapped_module.weight)
            ce = self.wrapped_module.weight*scale_w
            # print('max',torch.max(ce))
            # print('weight*scale',self.wrapped_module.weight*scale_w)
            # print('weight_ori',weight_ori.size())
            wnz = (weight_ori.view(-1) == 0).sum()*1.0/weight_ori.numel()
            #print("wnz:",wnz)
            in_a = self.conv_q(*input)
            # print('in_a',in_a)
            res = self.wrapped_module(in_a)
            res_sliced = torch.cuda.FloatTensor(res.shape).fill_(0) 
            # print('res_sliced_ideal',res_sliced.size())

            # for ba in range(0, bit_activ//2):#��֧��2������������
            #     for bw in range(0, bit_weight):
            #         input_tmp = (input_ori.int() & (3<<(ba*2))) >> (ba*2)#����λȡ�����ֵ
            #         weight_tmp = ((weight_ori+(1<<(bit_weight))).int() & (1<<bw)) >> bw
            if hasattr(self.wrapped_module, 'kernel_size'):
                # if (self.wrapped_module.in_channels == 31):
                #return res
                for ba in range(0, bit_activ//2):#��֧��2������������
                    for bw in range(0, bit_weight):
                        input_tmp = (input_ori.int() & (3<<(ba*2))) >> (ba*2)#����λȡ�����ֵ
                        weight_tmp = ((weight_ori+(1<<(bit_weight))).int() & (1<<bw)) >> bw

                        kernel_size = self.wrapped_module.kernel_size[0] * self.wrapped_module.kernel_size[1]
                        para = paranumber // kernel_size
                        para_remain = paranumber % kernel_size
                        # print('channel', self.wrapped_module.in_channels)
                        paralevel = (self.wrapped_module.in_channels * kernel_size) // paranumber
                        last_number = (self.wrapped_module.in_channels * kernel_size) % paranumber
                        if(last_number != 0): paralevel += 1
                        # print('last_number', (last_number & 1))
                        # print('paralevel', paralevel)
                        # print('lastchannel', last_channel)
                        # if(paralevel==0): paralevel += 1
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

                                w_tmp = w_cis[:,:,krow:krow+1,kcol:kcol+1]
                                #ȡ��1����λ��weight,���ֱ�ȡ��1bit
                                w_tmp = w_tmp.float()
                                a_tmp = a_pad.int()
                                if(R_on):
                                    # w_tt = w_tmp
                                    # w_tmp[w_tmp==0] = low_condu
                                    # w_tmp[w_tmp==1] = high_condu
                                    # w_tt[w_tt==0] = 0.000001
                                    # w_tt[w_tt==1] = 0.001
                                    # print(w_tmp[0,:,:,:])
                                    rram_error_1 = torch.cuda.FloatTensor(w_tmp.shape).normal_(2050,59.646)

                                    rram_error_0 = torch.cuda.FloatTensor(w_tmp.shape).normal_(loc, scale)


                                    w_tmp = torch.where(w_tmp==1, rram_error_1, w_tmp)
                                    w_tmp = torch.where(w_tmp==0, rram_error_0, w_tmp)
                                    # for i in range(0, w_tmp.size(0)):
                                    # #     # for j in range(0, w_tmp.size(1)):
                                    # #         # print(w_tmp[i,j,:,:])
                                    # #     # print(w_tmp)
                                    # #     # print('1',w_tmp[i,:,:,:].size(),w_tmp[i,:,:,:])
                                    # #     # print('2',w_tmp[i,0,0,0].size(),w_tmp[i,0,0,0])

                                    #     if(w_tmp[i,:,:,:] == 0): w_tmp[i,:,:,:] = get_ln_r(loc = loc, scale = scale)
                                    #     if(w_tmp[i,:,:,:] == 1): w_tmp[i,:,:,:] = get_hn_r()

                                res_tmp = F.conv2d(a_tmp.float(), w_tmp.float(), bias=None, stride=self.wrapped_module.stride, padding=0)

                                if bw == bit_weight-1:
                                    res_tmp = -res_tmp

                                res_sliced_tmp = res_sliced_tmp + (res_tmp<<(ba*2+bw)).float()#!!!!!!!
                            # bit_answer = (2**bit_activ-1)*(2**(bit_weight-1)-1)
                            # if(R_off):
                            if(R_on):
                                # print(res_sliced_tmp)
                                res_sliced_tmp /= (2050)
                                # res_sliced_tmp /= 0.001
                                res_sliced_tmp = torch.round(res_sliced_tmp)
                                # print(res_sliced_tmp)
                            # exit(0)
                            res_sliced = res_sliced + res_sliced_tmp
                        

                if self.conv_q.zero_point != 0:
                    #print("self.conv_q.zero_point:", self.conv_q.zero_point)
                    in_tmp_b = torch.ones_like(in_a)*self.conv_q.zero_point
                    res_tmp_b = F.conv2d(in_tmp_b, weight_ori, stride = self.wrapped_module.stride, padding = self.wrapped_module.padding)
                    res_sliced = res_sliced + res_tmp_b

                
                res_sliced = res_sliced / scale_wa

                if self.wrapped_module.bias is not None:
                    res_sliced = res_sliced + self.wrapped_module.bias.reshape(1, -1, 1, 1)
                # print('END2')
                # print('res_sliced', res_sliced.size())
                # print(res_sliced)
                return res_sliced
            else:#����kernel
                #return res
                para = paranumber

                for ba in range(0, bit_activ//2):#��֧��2������������
                    for bw in range(0, bit_weight):
                        input_tmp = (input_ori.int() & (3<<(ba*2))) >> (ba*2)#����λȡ�����ֵ
                        weight_tmp = ((weight_ori+(1<<(bit_weight))).int() & (1<<bw)) >> bw

                        # para = 100
                        # res2 = F.linear(input_ori, weight_ori, bias=None)
                        paralevel = self.wrapped_module.in_features // para
                        last_element = self.wrapped_module.in_features % para
                        # print(last_element)
                        for cib in range(0, paralevel):
                            res_sliced_tmp = torch.cuda.FloatTensor(res.shape).fill_(0)
                            in_cis = input_tmp[:, cib*para:cib*para+para]
                            w_cis = weight_tmp[:, cib*para:cib*para+para] 

                            w_tmp = w_cis
                            w_tmp = w_tmp.float()

                            if(R_on):
                            # if(R_off):
                                # w_tmp[w_tmp==0] = 0.000001
                                # w_tmp[w_tmp==1] = 0.001
                                rram_error_1 = torch.cuda.FloatTensor(w_tmp.shape).normal_(2050,59.646)
                                rram_error_0 = torch.cuda.FloatTensor(w_tmp.shape).normal_(loc, scale)

                                w_tmp = torch.where(w_tmp==1, rram_error_1, w_tmp)
                                w_tmp = torch.where(w_tmp==0, rram_error_0, w_tmp)
                                # for i in range(0, w_tmp.size(0)):
                                #     for j in range(0, w_tmp.size(1)):
                                #         if(w_tmp[i,j] == 0): w_tmp[i,j] = get_ln_r(loc=loc, scale=scale)
                                #         if(w_tmp[i,j] == 1): w_tmp[i,j] = get_hn_r()
                                #         # if(w_cis[i,j] == -1): w_cis[i,j] = -(get_hn_r())

                            # w_tmp_0, w_tmp_1 = w_tmp.split(para//2, dim = 1)
                            a_tmp = in_cis.int()
                            
                            res_tmp = F.linear(a_tmp.float(), w_tmp.float(), bias=None)

                            if bw == bit_weight-1:
                                res_tmp = -res_tmp

                            res_sliced_tmp = res_sliced_tmp+(res_tmp.int()<<(2*ba+bw)).float()
                            if(R_on):
                                # print(res_sliced_tmp)
                                res_sliced_tmp /= (2050)
                                # res_sliced_tmp /= 0.001
                                res_sliced_tmp = torch.round(res_sliced_tmp)
                                # print(res_sliced_tmp)
                            # exit(0)
                            res_sliced = res_sliced + res_sliced_tmp
                        
                        if(last_element != 0):
                            res_sliced_tmp = torch.cuda.FloatTensor(res.shape).fill_(0)
                            # print('c & r:', paralevel*para, paralevel*para+last_element)
                            in_cis = input_tmp[:, paralevel*para:paralevel*para+last_element]
                            w_cis = weight_tmp[:, paralevel*para:paralevel*para+last_element]
                            # in_cis = input_tmp[:, cib*para:cib*para+para]
                            # w_cis = weight_tmp[:, cib*para:cib*para+para] 

                            w_tmp = w_cis

                            if(R_on):
                            # if(R_off):
                                # w_tmp[w_tmp==0] = 0.000001
                                # w_tmp[w_tmp==1] = 0.001
                                for i in range(0, w_tmp.size(0)):
                                    for j in range(0, w_tmp.size(1)):
                                        if(w_tmp[i,j] == 0): w_tmp[i,j] = get_ln_r(loc=loc, scale=scale)
                                        if(w_tmp[i,j] == 1): w_tmp[i,j] = get_hn_r()

                            # w_tmp_0, w_tmp_1 = w_tmp.split(para//2, dim = 1)
                            a_tmp = in_cis.int()
                            
                            res_tmp = F.linear(a_tmp.float(), w_tmp.float(), bias=None)

                            if bw == bit_weight-1:
                                res_tmp = -res_tmp

                            res_sliced_tmp = res_sliced_tmp+(res_tmp.int()<<(2*ba+bw)).float()
                            if(R_on):
                                # print(res_sliced_tmp)
                                res_sliced_tmp /= 2050
                                # res_sliced_tmp /= 0.001
                                res_sliced_tmp = torch.round(res_sliced_tmp)
                                # print(res_sliced_tmp)
                            # exit(0)
                            res_sliced = res_sliced + res_sliced_tmp
                            
                # bit_answer = (2**bit_activ-1)*(2**bit_weight-1)
                # if(R_on):
                #     for i in range(0, bit_answer*para):
                #         res_sliced[((i*high_condu)<=res_sliced) & (res_sliced < (i+1)*high_condu)] = i
                    
                res_sliced = res_sliced / scale_wa
                res_sliced = res_sliced + self.wrapped_module.bias
                # print('END3')
                # print('res_sliced', res_sliced.size())
                # print(res_sliced)
                return res_sliced
            exit(0)
        elif mode == "sparate_pn_kernel":
            torch.backends.cudnn.deterministic = True
            thre = 7 
            ad_limit = 15
            bit_weight = 4
            bit_activ = 4
            # para = 18
            paranumber = 16
            loc = 10.25
            scale = 0.57875
            delta = 340


            scale_w = self.wrapped_module.weight_scale
            scale_a = self.conv_q.scale
            scale_wa = scale_w * scale_a
            input_ori = torch.round(input[0] * scale_a - self.conv_q.zero_point)
            input_ori = torch.clamp(input_ori, 0, (2**bit_activ)-1)
            # print('input_ori.size', input_ori.size())
            weight_ori = torch.round(self.wrapped_module.weight*scale_w)
            w_p_ori = torch.where(weight_ori > 0,weight_ori, torch.zeros_like(weight_ori) )#����
            w_n_ori = torch.where(weight_ori < 0,-(weight_ori), torch.zeros_like(weight_ori) )#����
            # print(weight_ori.device, w_p_ori.device, w_n_ori.device)
            # print(w_p_ori)
            # print(w_n_ori)
            # exit(0)
            #print("wnz:",wnz)
            in_a = self.conv_q(*input)
            # print('in_a',in_a)
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
                        # print('channel', self.wrapped_module.in_channels)
                        paralevel = (self.wrapped_module.in_channels * kernel_size) // paranumber
                        last_number = (self.wrapped_module.in_channels * kernel_size) % paranumber
                        if(last_number != 0): paralevel += 1
                        # print('last_number', (last_number & 1))
                        # print('paralevel', paralevel)
                        # print('lastchannel', last_channel)
                        # if(paralevel==0): paralevel += 1
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
                                    # w_tt = w_tmp
                                    # w_tmp[w_tmp==0] = low_condu
                                    # w_tmp[w_tmp==1] = high_condu
                                    # w_tt[w_tt==0] = 0.000001
                                    # w_tt[w_tt==1] = 0.001
                                    # print(w_tmp[0,:,:,:])
                                    rram_error_1 = torch.cuda.FloatTensor(w_tmp_p.shape).normal_(2050,59.646)
                                    # print(rram_error_1)
                                    # exit(0)
                                    # print('rram1 & w',rram_error_1.device, w_tmp_p.device)
                                    rram_error_0_p = torch.cuda.FloatTensor(w_tmp_p.shape).normal_(loc, scale)
                                    rram_error_0_n = torch.cuda.FloatTensor(w_tmp_n.shape).normal_(loc, scale)

                                    # rram_error_1 = torch.randn(size=w_tmp_p.shape, out=rram_error_1) #*59.646 + 2050
                                    # rram_error_1 = rram_error_1 * 59.646 + 2050
                                    # rram_error_0_p = torch.randn(size=w_tmp_p.shape, out=rram_error_0_p) *scale + loc
                                    # rram_error_0_n = torch.randn(size=w_tmp_n.shape) *scale + loc

                                    # print('rram & w',rram_error_1.device, w_tmp_p.device)

                                    w_tmp_p = torch.where(w_tmp_p==1, rram_error_1, w_tmp_p)
                                    w_tmp_p = torch.where(w_tmp_p==0, rram_error_0_p, w_tmp_p)
                                    w_tmp_n = torch.where(w_tmp_n==1, rram_error_1, w_tmp_n)
                                    w_tmp_n = torch.where(w_tmp_n==0, rram_error_0_n, w_tmp_n)

                                    # for i in range(0, w_tmp_p.size(0)):
                                    # #     # for j in range(0, w_tmp.size(1)):
                                    # #         # print(w_tmp[i,j,:,:])
                                    # #     # print(w_tmp)
                                    # #     # print('1',w_tmp[i,:,:,:].size(),w_tmp[i,:,:,:])
                                    # #     # print('2',w_tmp[i,0,0,0].size(),w_tmp[i,0,0,0])
                                    #     if(w_tmp_p[i,:,:,:] == 0): w_tmp_p[i,:,:,:] = get_ln_r(loc=loc, scale=scale)
                                    #     if(w_tmp_p[i,:,:,:] == 1): w_tmp_p[i,:,:,:] = get_hn_r()
                                    #     if(w_tmp_n[i,:,:,:] == 0): w_tmp_n[i,:,:,:] = get_ln_r(loc=loc, scale=scale)
                                    #     if(w_tmp_n[i,:,:,:] == 1): w_tmp_n[i,:,:,:] = get_hn_r()

                                res_tmp_p = F.conv2d(a_tmp.float(), w_tmp_p.float(), bias=None, stride=self.wrapped_module.stride, padding=0)
                                res_tmp_n = F.conv2d(a_tmp.float(), w_tmp_n.float(), bias=None, stride=self.wrapped_module.stride, padding=0)
                                res_tmp = res_tmp_p - res_tmp_n

                                res_sliced_tmp = res_sliced_tmp + (res_tmp<<(ba*2+bw)).float()#!!!!!!!
                            # bit_answer = (2**bit_activ-1)*(2**(bit_weight-1)-1)
                            # if(R_off):
                            if(R_on):
                                # print(res_sliced_tmp)
                                # adc_error = torch.randn(size=res_sliced_tmp.shape, out=adc_error) * delta
                                # res_sliced_tmp += adc_error
                                # print(res_sliced_tmp)
                                adc_error = torch.normal(mean=res_sliced_tmp, std = delta)
                                res_sliced_tmp = adc_error / (2050-loc)
                                # res_sliced_tmp /= 0.001
                                res_sliced_tmp = torch.round(res_sliced_tmp)
                                # print(res_sliced_tmp)
                            # exit(0)
                            res_sliced = res_sliced + res_sliced_tmp
                        

                if self.conv_q.zero_point != 0:
                    #print("self.conv_q.zero_point:", self.conv_q.zero_point)
                    in_tmp_b = torch.ones_like(in_a)*self.conv_q.zero_point
                    res_tmp_b = F.conv2d(in_tmp_b, weight_ori, stride = self.wrapped_module.stride, padding = self.wrapped_module.padding)
                    res_sliced = res_sliced + res_tmp_b

                
                res_sliced = res_sliced / scale_wa

                if self.wrapped_module.bias is not None:
                    res_sliced = res_sliced + self.wrapped_module.bias.reshape(1, -1, 1, 1)
                # print('END2')
                # print('res_sliced', res_sliced.size())
                # print(res_sliced)
                return res_sliced
            else:#����kernel
                #return res
                para = paranumber

                for ba in range(0, bit_activ//2):#��֧��2������������
                    for bw in range(0, bit_weight):
                        input_tmp = (input_ori.int() & (3<<(ba*2))) >> (ba*2)#����λȡ�����ֵ
                        # weight_tmp = ((weight_ori+(1<<(bit_weight))).int() & (1<<bw)) >> bw
                        weight_tmp_p = ((w_p_ori+(1<<(bit_weight))).int() & (1<<bw)) >> bw
                        weight_tmp_n = ((w_n_ori+(1<<(bit_weight))).int() & (1<<bw)) >> bw

                        # para = 100
                        # res2 = F.linear(input_ori, weight_ori, bias=None)
                        paralevel = self.wrapped_module.in_features // para
                        last_element = self.wrapped_module.in_features % para
                        # print(last_element)
                        for cib in range(0, paralevel):
                            res_sliced_tmp = torch.cuda.FloatTensor(res.shape).fill_(0)
                            in_cis = input_tmp[:, cib*para:cib*para+para]
                            w_cis_p = weight_tmp_p[:, cib*para:cib*para+para] 
                            w_cis_n = weight_tmp_n[:, cib*para:cib*para+para] 

                            w_tmp_p = w_cis_p.float()
                            w_tmp_n = w_cis_n.float()

                            if(R_on):
                            # if(R_off):
                                # w_tmp[w_tmp==0] = 0.000001
                                # w_tmp[w_tmp==1] = 0.001
                                # rram_error_1 = torch.cuda.FloatTensor(w_tmp_p.shape).fill_(0)
                                # rram_error_0_p = torch.cuda.FloatTensor(w_tmp_p.shape).fill_(0)
                                # rram_error_0_n = torch.cuda.FloatTensor(w_tmp_n.shape).fill_(0)

                                # rram_error_1 = torch.randn(size=w_tmp_p.shape) *59.646 + 2050
                                # rram_error_0_p = torch.randn(size=w_tmp_p.shape) *scale + loc
                                # rram_error_0_n = torch.randn(size=w_tmp_n.shape) *scale + loc
                                
                                rram_error_1 = torch.cuda.FloatTensor(w_tmp_p.shape).normal_(2050,59.646)
                                rram_error_0_p = torch.cuda.FloatTensor(w_tmp_p.shape).normal_(loc, scale)
                                rram_error_0_n = torch.cuda.FloatTensor(w_tmp_n.shape).normal_(loc, scale)

                                # print(type(w_tmp_p), type(rram_error_1))

                                w_tmp_p = torch.where(w_tmp_p==1, rram_error_1, w_tmp_p)
                                w_tmp_p = torch.where(w_tmp_p==0, rram_error_0_p, w_tmp_p)
                                w_tmp_n = torch.where(w_tmp_n==1, rram_error_1, w_tmp_n)
                                w_tmp_n = torch.where(w_tmp_n==0, rram_error_0_n, w_tmp_n)
                                # for i in range(0, w_tmp_p.size(0)):
                                #     for j in range(0, w_tmp_p.size(1)):
                                #         if(w_tmp_p[i,j] == 0): w_tmp_p[i,j] = get_ln_r(loc=loc, scale=scale)
                                #         if(w_tmp_p[i,j] == 1): w_tmp_p[i,j] = get_hn_r()
                                #         if(w_tmp_n[i,j] == 0): w_tmp_n[i,j] = get_ln_r(loc=loc, scale=scale)
                                #         if(w_tmp_n[i,j] == 1): w_tmp_n[i,j] = get_hn_r()
                                #         # if(w_cis[i,j] == -1): w_cis[i,j] = -(get_hn_r())

                            # w_tmp_0, w_tmp_1 = w_tmp.split(para//2, dim = 1)
                            a_tmp = in_cis.int()
                            
                            res_tmp_p = F.linear(a_tmp.float(), w_tmp_p.float(), bias=None)
                            res_tmp_n = F.linear(a_tmp.float(), w_tmp_n.float(), bias=None)
                            res_tmp = res_tmp_p - res_tmp_n


                            res_sliced_tmp = res_sliced_tmp+(res_tmp.int()<<(2*ba+bw)).float()
                            if(R_on):
                                # print(res_sliced_tmp)
                                # adc_error = torch.rand(size=res_sliced_tmp.shape, out=adc_error) * delta
                                # res_sliced_tmp += adc_error
                                # res_sliced_tmp = get_ln_r(loc=res_sliced_tmp, scale=1)
                                adc_error = torch.normal(mean=res_sliced_tmp, std = delta)
                                res_sliced_tmp = adc_error / (2050-loc)
                                # res_sliced_tmp /= 2050
                                # res_sliced_tmp /= 0.001
                                res_sliced_tmp = torch.round(res_sliced_tmp)
                                # print(res_sliced_tmp)
                            # exit(0)
                            res_sliced = res_sliced + res_sliced_tmp
                        
                        if(last_element != 0):
                            res_sliced_tmp = torch.cuda.FloatTensor(res.shape).fill_(0)
                            # print('c & r:', paralevel*para, paralevel*para+last_element)
                            in_cis = input_tmp[:, paralevel*para:paralevel*para+last_element]
                            w_cis_p = weight_tmp_p[:, paralevel*para:paralevel*para+last_element]
                            w_cis_n = weight_tmp_n[:, paralevel*para:paralevel*para+last_element]
                            # in_cis = input_tmp[:, cib*para:cib*para+para]
                            # w_cis = weight_tmp[:, cib*para:cib*para+para] 

                            w_tmp_p = w_cis_p.float()
                            w_tmp_n = w_cis_n.float()

                            if(R_on):
                            # if(R_off):
                                # w_tmp[w_tmp==0] = 0.000001
                                # w_tmp[w_tmp==1] = 0.001
                                # rram_error_1 = torch.cuda.FloatTensor(w_tmp_p.shape).fill_(0)
                                # rram_error_0_p = torch.cuda.FloatTensor(w_tmp_p.shape).fill_(0)
                                # rram_error_0_n = torch.cuda.FloatTensor(w_tmp_n.shape).fill_(0)

                                # rram_error_1 = torch.randn(size=w_tmp_p.shape) *59.646 + 2050
                                # rram_error_0_p = torch.randn(size=w_tmp_p.shape) *scale + loc
                                # rram_error_0_n = torch.randn(size=w_tmp_n.shape) *scale + loc

                                rram_error_1 = torch.cuda.FloatTensor(w_tmp_p.shape).normal_(2050,59.646)
                                rram_error_0_p = torch.cuda.FloatTensor(w_tmp_p.shape).normal_(loc, scale)
                                rram_error_0_n = torch.cuda.FloatTensor(w_tmp_n.shape).normal_(loc, scale)

                                w_tmp_p = torch.where(w_tmp_p==1, rram_error_1, w_tmp_p)
                                w_tmp_p = torch.where(w_tmp_p==0, rram_error_0_p, w_tmp_p)
                                w_tmp_n = torch.where(w_tmp_n==1, rram_error_1, w_tmp_n)
                                w_tmp_n = torch.where(w_tmp_n==0, rram_error_0_n, w_tmp_n)
                                # for i in range(0, w_tmp_p.size(0)):
                                #     for j in range(0, w_tmp_p.size(1)):
                                #         if(w_tmp_p[i,j] == 0): w_tmp_p[i,j] = get_ln_r(loc=loc, scale=scale)
                                #         if(w_tmp_p[i,j] == 1): w_tmp_p[i,j] = get_hn_r()
                                #         if(w_tmp_n[i,j] == 0): w_tmp_n[i,j] = get_ln_r(loc=loc, scale=scale)
                                #         if(w_tmp_n[i,j] == 1): w_tmp_n[i,j] = get_hn_r()
                                ##         # if(w_cis[i,j] == -1): w_cis[i,j] = -(get_hn_r())

                            # w_tmp_0, w_tmp_1 = w_tmp.split(para//2, dim = 1)
                            a_tmp = in_cis.int()
                            
                            res_tmp_p = F.linear(a_tmp.float(), w_tmp_p.float(), bias=None)
                            res_tmp_n = F.linear(a_tmp.float(), w_tmp_n.float(), bias=None)
                            res_tmp = res_tmp_p - res_tmp_n

                            res_sliced_tmp = res_sliced_tmp+(res_tmp.int()<<(2*ba+bw)).float()
                            if(R_on):
                                # print(res_sliced_tmp)
                                # res_sliced_tmp = get_ln_r(loc=res_sliced_tmp.shape, scale=1)
                                # adc_error = torch.rand(size=res_sliced_tmp, out=adc_error) * delta
                                # res_sliced_tmp += adc_error
                                adc_error = torch.normal(mean=res_sliced_tmp, std = delta)
                                res_sliced_tmp = adc_error / (2050-loc)
                                # res_sliced_tmp /= 2050
                                # res_sliced_tmp /= 0.001
                                res_sliced_tmp = torch.round(res_sliced_tmp)
                                # print(res_sliced_tmp)
                            # exit(0)
                            res_sliced = res_sliced + res_sliced_tmp
                            
                # bit_answer = (2**bit_activ-1)*(2**bit_weight-1)
                # if(R_on):
                #     for i in range(0, bit_answer*para):
                #         res_sliced[((i*high_condu)<=res_sliced) & (res_sliced < (i+1)*high_condu)] = i
                    
                res_sliced = res_sliced / scale_wa
                res_sliced = res_sliced + self.wrapped_module.bias
                # print('END3')
                # print('res_sliced', res_sliced.size())
                # print(res_sliced)
                return res_sliced
        elif mode == "varia_pn_kernel":
            torch.backends.cudnn.deterministic = True
            thre = 7 
            ad_limit = 15
            bit_weight = 4
            bit_activ = 4
            # para = 18
            paranumber = 16
            loc = 10.25
            scale = 0.57875
            delta = 400

            scale_w = self.wrapped_module.weight_scale
            scale_a = self.conv_q.scale
            scale_wa = scale_w * scale_a
            input_ori = torch.round(input[0] * scale_a - self.conv_q.zero_point)
            input_ori = torch.clamp(input_ori, 0, (2**bit_activ)-1)
            # print('input_ori.size', input_ori.size())
            weight_ori = torch.round(self.wrapped_module.weight*scale_w)
            w_p_ori = torch.where(weight_ori > 0,weight_ori, torch.zeros_like(weight_ori) )#����
            w_n_ori = torch.where(weight_ori < 0,-(weight_ori), torch.zeros_like(weight_ori) )#����
            # print("in,w_p,w_n", input_ori.size(), w_p_ori.size(), w_n_ori.size())
            # print(weight_ori.device, w_p_ori.device, w_n_ori.device)
            # print(w_p_ori)
            # print(w_n_ori)
            # exit(0)
            #print("wnz:",wnz)
            in_a = self.conv_q(*input)
            # print('in_a',in_a)
            res = self.wrapped_module(in_a)
            # print('res', res.size())
            res_sliced = torch.cuda.FloatTensor(res.shape).fill_(0) 

            if hasattr(self.wrapped_module, 'kernel_size'):
                for ba in range(0, bit_activ//2):#��֧��2������������
                    for bw in range(0, bit_weight):
                        res_sliced_tmp = torch.cuda.FloatTensor(res.shape).fill_(0)
                        input_tmp = (input_ori.int() & (3<<(ba*2))) >> (ba*2)#����λȡ�����ֵ
                        weight_tmp_p = ((w_p_ori+(1<<(bit_weight))).int() & (1<<bw)) >> bw
                        weight_tmp_n = ((w_n_ori+(1<<(bit_weight))).int() & (1<<bw)) >> bw

                        kernel_size = self.wrapped_module.kernel_size[0] * self.wrapped_module.kernel_size[1]
                        # para = paranumber // kernel_size
                        # para_remain = paranumber % kernel_size
                        # print('channel', self.wrapped_module.in_channels)
                        paralevel = (self.wrapped_module.in_channels * kernel_size) // paranumber
                        last_number = (self.wrapped_module.in_channels * kernel_size) % paranumber
                        if(last_number != 0): paralevel += 1
                        # print("in, wn, wp",input_tmp.size(), weight_tmp_n.size(), weight_tmp_p.size())
                        
                        

                        input_tmp = input_tmp.float()
                        weight_tmp_p = weight_tmp_p.float()
                        weight_tmp_n = weight_tmp_n.float()

                        if(R_SAF):
                            saf_p = torch.cuda.FloatTensor(weight_tmp_p.shape).fill_(0)
                            saf_p = torch.rand(size=weight_tmp_p.shape, out=saf_p)
                            # print(saf_p, type(saf_p))
                            zero = torch.cuda.FloatTensor(weight_tmp_p.shape).fill_(0)
                            one = torch.cuda.FloatTensor(weight_tmp_p.shape).fill_(1)
                            saf_n = torch.cuda.FloatTensor(weight_tmp_n.shape).fill_(0)
                            saf_n = torch.rand(size=weight_tmp_n.shape, out=saf_n)
                            weight_tmp_p = torch.where(saf_p<0.09, zero, weight_tmp_p)
                            weight_tmp_p = torch.where(saf_p>0.99, one, weight_tmp_p)
                            weight_tmp_n = torch.where(saf_n<0.09, zero, weight_tmp_n)
                            weight_tmp_n = torch.where(saf_n>0.99, one, weight_tmp_n)

                        if(R_on):

                            rram_error_1 = torch.cuda.FloatTensor(weight_tmp_p.shape).normal_(2050,59.646)#59.646

                            rram_error_0_p = torch.cuda.FloatTensor(weight_tmp_p.shape).normal_(loc, scale)
                            rram_error_0_n = torch.cuda.FloatTensor(weight_tmp_n.shape).normal_(loc, scale)


                            weight_tmp_p = torch.where(weight_tmp_p==1, rram_error_1, weight_tmp_p)
                            weight_tmp_p = torch.where(weight_tmp_p==0, rram_error_0_p, weight_tmp_p)
                            weight_tmp_n = torch.where(weight_tmp_n==1, rram_error_1, weight_tmp_n)
                            weight_tmp_n = torch.where(weight_tmp_n==0, rram_error_0_n, weight_tmp_n)

                        res_tmp_p = F.conv2d(input_tmp.float(), weight_tmp_p.float(), bias=None, stride=self.wrapped_module.stride, padding=self.wrapped_module.padding)
                        res_tmp_n = F.conv2d(input_tmp.float(), weight_tmp_n.float(), bias=None, stride=self.wrapped_module.stride, padding=self.wrapped_module.padding)
                        res_tmp = res_tmp_p - res_tmp_n
                        # print('res_tmp', res_tmp.size())

                        res_sliced_tmp = res_sliced_tmp + (res_tmp<<(ba*2+bw)).float()#!!!!!!!


                            # if(R_off):
                        if(R_on):
                            # print(res_sliced_tmp)
                            # adc_error = torch.randn(size=res_sliced_tmp.shape, out=adc_error) * delta
                            # res_sliced_tmp += adc_error
                            # print(res_sliced_tmp)
                            adc_error = torch.cuda.FloatTensor(res_sliced_tmp.shape).fill_(0)
                            for i in range(0, paralevel):
                                adc_error += torch.normal(mean=torch.cuda.FloatTensor(res_sliced_tmp.shape).fill_(0), std = delta)
                            res_sliced_tmp += adc_error
                            res_sliced_tmp = res_sliced_tmp / (2050-loc)
                            # res_sliced_tmp /= 0.001
                            res_sliced_tmp = torch.round(res_sliced_tmp)
                            # print(res_sliced_tmp)
                        # exit(0)
                        res_sliced = res_sliced + res_sliced_tmp
                        

                if self.conv_q.zero_point != 0:
                    #print("self.conv_q.zero_point:", self.conv_q.zero_point)
                    in_tmp_b = torch.ones_like(in_a)*self.conv_q.zero_point
                    res_tmp_b = F.conv2d(in_tmp_b, weight_ori, stride = self.wrapped_module.stride, padding = self.wrapped_module.padding)
                    res_sliced = res_sliced + res_tmp_b

                
                res_sliced = res_sliced / scale_wa

                if self.wrapped_module.bias is not None:
                    res_sliced = res_sliced + self.wrapped_module.bias.reshape(1, -1, 1, 1)
                # print('END2')
                # print('res_sliced', res_sliced.size())
                # print(res_sliced)
                return res_sliced
            else:#����kernel
                #return res
                para = paranumber

                for ba in range(0, bit_activ//2):#��֧��2������������
                    for bw in range(0, bit_weight):
                        res_sliced_tmp = torch.cuda.FloatTensor(res.shape).fill_(0)
                        input_tmp = (input_ori.int() & (3<<(ba*2))) >> (ba*2)#����λȡ�����ֵ
                        # weight_tmp = ((weight_ori+(1<<(bit_weight))).int() & (1<<bw)) >> bw
                        weight_tmp_p = ((w_p_ori+(1<<(bit_weight))).int() & (1<<bw)) >> bw
                        weight_tmp_n = ((w_n_ori+(1<<(bit_weight))).int() & (1<<bw)) >> bw

                        paralevel = self.wrapped_module.in_features // para
                        last_element = self.wrapped_module.in_features % para
                        if(last_element != 0): paralevel += 1

                        # para = 100
                        # res2 = F.linear(input_ori, weight_ori, bias=None)
                        input_tmp = input_tmp.float()
                        weight_tmp_p = weight_tmp_p.float()
                        weight_tmp_n = weight_tmp_n.float()
                        # print(last_element)
                        if(R_on):
                            # if(R_off):
                                
                            rram_error_1 = torch.cuda.FloatTensor(weight_tmp_p.shape).normal_(2050,59.646)
                            rram_error_0_p = torch.cuda.FloatTensor(weight_tmp_p.shape).normal_(loc, scale)
                            rram_error_0_n = torch.cuda.FloatTensor(weight_tmp_p.shape).normal_(loc, scale)

                            # print(type(w_tmp_p), type(rram_error_1))

                            weight_tmp_p = torch.where(weight_tmp_p==1, rram_error_1, weight_tmp_p)
                            weight_tmp_p = torch.where(weight_tmp_p==0, rram_error_0_p, weight_tmp_p)
                            weight_tmp_n = torch.where(weight_tmp_n==1, rram_error_1, weight_tmp_n)
                            weight_tmp_n = torch.where(weight_tmp_n==0, rram_error_0_n, weight_tmp_n)
                            
                        res_tmp_p = F.linear(input_tmp.float(), weight_tmp_p.float(), bias=None)
                        res_tmp_n = F.linear(input_tmp.float(), weight_tmp_n.float(), bias=None)
                        res_tmp = res_tmp_p - res_tmp_n


                        res_sliced_tmp = res_sliced_tmp+(res_tmp.int()<<(2*ba+bw)).float()

                        
                        if(R_on):
                            # print(res_sliced_tmp)
                            # res_sliced_tmp = get_ln_r(loc=res_sliced_tmp.shape, scale=1)
                            # adc_error = torch.rand(size=res_sliced_tmp, out=adc_error) * delta
                            # res_sliced_tmp += adc_error
                            # adc_error = torch.normal(mean=res_sliced_tmp, std = delta)
                            adc_error = torch.cuda.FloatTensor(res_sliced_tmp.shape).fill_(0)
                            for i in range(0, paralevel):
                                adc_error += torch.normal(mean=torch.cuda.FloatTensor(res_sliced_tmp.shape).fill_(0), std = delta)
                            res_sliced_tmp += adc_error
                            res_sliced_tmp = res_sliced_tmp / (2050-loc)
                            # res_sliced_tmp /= 2050
                            # res_sliced_tmp /= 0.001
                            res_sliced_tmp = torch.round(res_sliced_tmp)
                            # print(res_sliced_tmp)
                        # exit(0)
                        res_sliced = res_sliced + res_sliced_tmp
                            
                # bit_answer = (2**bit_activ-1)*(2**bit_weight-1)
                # if(R_on):
                #     for i in range(0, bit_answer*para):
                #         res_sliced[((i*high_condu)<=res_sliced) & (res_sliced < (i+1)*high_condu)] = i
                    
                res_sliced = res_sliced / scale_wa
                res_sliced = res_sliced + self.wrapped_module.bias
                # print('END3')
                # print('res_sliced', res_sliced.size())
                # print(res_sliced)
                return res_sliced
        elif mode == "fast_2mod":
            torch.backends.cudnn.deterministic = True
            thre = 7 
            ad_limit = 15
            bit_weight = 4
            bit_activ = 4
            # para = 18
            paranumber = 16
            loc = 10.25
            scale = 0.57875
            delta = 2400

            scale_w = self.wrapped_module.weight_scale
            scale_a = self.conv_q.scale
            scale_wa = scale_w * scale_a
            input_ori = torch.round(input[0] * scale_a - self.conv_q.zero_point)
            input_ori = torch.clamp(input_ori, 0, (2**bit_activ)-1)
            # print('input_ori.size', input_ori.size())
            weight_ori = torch.round(self.wrapped_module.weight*scale_w)
            # w_p_ori = torch.where(weight_ori > 0,weight_ori, torch.zeros_like(weight_ori) )#����
            # w_n_ori = torch.where(weight_ori < 0,-(weight_ori), torch.zeros_like(weight_ori) )#����
            # print("in,w_p,w_n", input_ori.size(), w_p_ori.size(), w_n_ori.size())
            # print(weight_ori.device, w_p_ori.device, w_n_ori.device)
            # print(w_p_ori)
            # print(w_n_ori)
            # exit(0)
            #print("wnz:",wnz)
            in_a = self.conv_q(*input)
            # print('in_a',in_a)
            res = self.wrapped_module(in_a)
            # print('res', res.size())
            res_sliced = torch.cuda.FloatTensor(res.shape).fill_(0) 

            if hasattr(self.wrapped_module, 'kernel_size'):
                for ba in range(0, bit_activ//2):#��֧��2������������
                    for bw in range(0, bit_weight):
                        res_sliced_tmp = torch.cuda.FloatTensor(res.shape).fill_(0)
                        input_tmp = (input_ori.int() & (3<<(ba*2))) >> (ba*2)#����λȡ�����ֵ
                        weight_tmp = ((weight_ori+(1<<(bit_weight))).int() & (1<<bw)) >> bw
                        # weight_tmp_n = ((w_n_ori+(1<<(bit_weight))).int() & (1<<bw)) >> bw
                        # print("in, wn, wp",input_tmp.size(), weight_tmp_n.size(), weight_tmp_p.size())
                        
                        

                        input_tmp = input_tmp.float()
                        weight_tmp = weight_tmp.float()

                        if(R_SAF):
                            saf = torch.cuda.FloatTensor(weight_tmp.shape).fill_(0)
                            saf = torch.rand(size=weight_tmp.shape, out=saf)
                            # print(saf_p, type(saf_p))
                            zero = torch.cuda.FloatTensor(weight_tmp.shape).fill_(0)
                            one = torch.cuda.FloatTensor(weight_tmp.shape).fill_(1)
                            weight_tmp = torch.where(saf_p<0.09, zero, weight_tmp)
                            weight_tmp = torch.where(saf_p>0.99, one, weight_tmp)

                        if(R_on):

                            rram_error_1 = torch.cuda.FloatTensor(weight_tmp_p.shape).normal_(2050,59.646)

                            rram_error_0 = torch.cuda.FloatTensor(weight_tmp.shape).normal_(loc, scale)


                            weight_tmp = torch.where(weight_tmp_p==1, rram_error_1, weight_tmp)
                            weight_tmp = torch.where(weight_tmp_p==0, rram_error_0_p, weight_tmp)

                        res_tmp = F.conv2d(input_tmp.float(), weight_tmp.float(), bias=None, stride=self.wrapped_module.stride, padding=self.wrapped_module.padding)
                        # print('res_tmp', res_tmp.size())

                        res_sliced_tmp = res_sliced_tmp + (res_tmp<<(ba*2+bw)).float()#!!!!!!!


                            # if(R_off):
                        if(R_on):
                            # print(res_sliced_tmp)
                            # adc_error = torch.randn(size=res_sliced_tmp.shape, out=adc_error) * delta
                            # res_sliced_tmp += adc_error
                            # print(res_sliced_tmp)
                            # adc_error = torch.normal(mean=res_sliced_tmp, std = delta)
                            res_sliced_tmp = res_sliced_tmp / 2050
                            # res_sliced_tmp /= 0.001
                            res_sliced_tmp = torch.round(res_sliced_tmp)
                            # print(res_sliced_tmp)
                        # exit(0)
                        res_sliced = res_sliced + res_sliced_tmp
                        

                if self.conv_q.zero_point != 0:
                    #print("self.conv_q.zero_point:", self.conv_q.zero_point)
                    in_tmp_b = torch.ones_like(in_a)*self.conv_q.zero_point
                    res_tmp_b = F.conv2d(in_tmp_b, weight_ori, stride = self.wrapped_module.stride, padding = self.wrapped_module.padding)
                    res_sliced = res_sliced + res_tmp_b

                
                res_sliced = res_sliced / scale_wa

                if self.wrapped_module.bias is not None:
                    res_sliced = res_sliced + self.wrapped_module.bias.reshape(1, -1, 1, 1)
                # print('END2')
                # print('res_sliced', res_sliced.size())
                # print(res_sliced)
                return res_sliced
            else:#����kernel
                #return res
                para = paranumber

                for ba in range(0, bit_activ//2):#��֧��2������������
                    for bw in range(0, bit_weight):
                        res_sliced_tmp = torch.cuda.FloatTensor(res.shape).fill_(0)
                        input_tmp = (input_ori.int() & (3<<(ba*2))) >> (ba*2)#����λȡ�����ֵ
                        # weight_tmp = ((weight_ori+(1<<(bit_weight))).int() & (1<<bw)) >> bw
                        weight_tmp = ((weight_ori+(1<<(bit_weight))).int() & (1<<bw)) >> bw

                        # para = 100
                        # res2 = F.linear(input_ori, weight_ori, bias=None)
                        input_tmp = input_tmp.float()
                        weight_tmp = weight_tmp.float()
                        # print(last_element)
                        if(R_on):
                            # if(R_off):
                                
                            rram_error_1 = torch.cuda.FloatTensor(weight_tmp.shape).normal_(2050,59.646)
                            rram_error_0 = torch.cuda.FloatTensor(weight_tmp.shape).normal_(loc, scale)

                            # print(type(w_tmp_p), type(rram_error_1))

                            weight_tmp = torch.where(weight_tmp_p==1, rram_error_1, weight_tmp)
                            weight_tmp = torch.where(weight_tmp_p==0, rram_error_0, weight_tmp)
                            
                        res_tmp = F.linear(input_tmp.float(), weight_tmp.float(), bias=None)

                        res_sliced_tmp = res_sliced_tmp+(res_tmp.int()<<(2*ba+bw)).float()

                        
                        if(R_on):
                            # print(res_sliced_tmp)
                            # res_sliced_tmp = get_ln_r(loc=res_sliced_tmp.shape, scale=1)
                            # adc_error = torch.rand(size=res_sliced_tmp, out=adc_error) * delta
                            # res_sliced_tmp += adc_error
                            # adc_error = torch.normal(mean=res_sliced_tmp, std = delta)
                            res_sliced_tmp = res_sliced_tmp / 2050
                            # res_sliced_tmp /= 2050
                            # res_sliced_tmp /= 0.001
                            res_sliced_tmp = torch.round(res_sliced_tmp)
                            # print(res_sliced_tmp)
                        # exit(0)
                        res_sliced = res_sliced + res_sliced_tmp
                            
                # bit_answer = (2**bit_activ-1)*(2**bit_weight-1)
                # if(R_on):
                #     for i in range(0, bit_answer*para):
                #         res_sliced[((i*high_condu)<=res_sliced) & (res_sliced < (i+1)*high_condu)] = i
                    
                res_sliced = res_sliced / scale_wa
                res_sliced = res_sliced + self.wrapped_module.bias
                # print('END3')
                # print('res_sliced', res_sliced.size())
                # print(res_sliced)
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
                # print('last_number', last_number)
                # print('paralevel', paralevel)
                # print('lastchannel', last_channel)
                # if(paralevel==0): paralevel += 1
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
                                    # for i in range(0, w_tmp.size(0)):
                                    #     if(w_tmp[i,:,:,:] >= 0):
                                    #         if(w_tmp[i,:,:,:] == 0): w_tmp[i,:,:,:] = get_ln_r(loc=loc_0, scale=scale_0)
                                    #         if(w_tmp[i,:,:,:] == 1): w_tmp[i,:,:,:] = get_ln_r(loc=loc_1, scale=scale_1)
                                    #         if(w_tmp[i,:,:,:] == 2): w_tmp[i,:,:,:] = get_ln_r(loc=loc_2, scale=scale_2)
                                    #         if(w_tmp[i,:,:,:] == 3): w_tmp[i,:,:,:] = get_ln_r(loc=loc_3, scale=scale_3)
                                    #         if(w_tmp[i,:,:,:] == 4): w_tmp[i,:,:,:] = get_ln_r(loc=loc_4, scale=scale_4)
                                    #         if(w_tmp[i,:,:,:] == 5): w_tmp[i,:,:,:] = get_ln_r(loc=loc_5, scale=scale_5)
                                    #         if(w_tmp[i,:,:,:] == 6): w_tmp[i,:,:,:] = get_ln_r(loc=loc_6, scale=scale_6)
                                    #         if(w_tmp[i,:,:,:] == 7): w_tmp[i,:,:,:] = get_ln_r(loc=loc_7, scale=scale_7)
                                    #         if(w_tmp[i,:,:,:] == 8): w_tmp[i,:,:,:] = get_ln_r(loc=loc_8, scale=scale_8)
                                    #         if(w_tmp[i,:,:,:] == 9): w_tmp[i,:,:,:] = get_ln_r(loc=loc_9, scale=scale_9)
                                    #         if(w_tmp[i,:,:,:] == 10): w_tmp[i,:,:,:] = get_ln_r(loc=loc_10, scale=scale_10)
                                    #         if(w_tmp[i,:,:,:] == 11): w_tmp[i,:,:,:] = get_ln_r(loc=loc_11, scale=scale_11)
                                    #         if(w_tmp[i,:,:,:] == 12): w_tmp[i,:,:,:] = get_ln_r(loc=loc_12, scale=scale_12)
                                    #         if(w_tmp[i,:,:,:] == 13): w_tmp[i,:,:,:] = get_ln_r(loc=loc_13, scale=scale_13)
                                    #         if(w_tmp[i,:,:,:] == 14): w_tmp[i,:,:,:] = get_ln_r(loc=loc_14, scale=scale_14)
                                    #         if(w_tmp[i,:,:,:] == 15): w_tmp[i,:,:,:] = get_ln_r(loc=loc_15, scale=scale_15)
                                    #     else:
                                    #         if(w_tmp[i,:,:,:] == -1): w_tmp[i,:,:,:] = -get_ln_r(loc=loc_1, scale=scale_1)
                                    #         if(w_tmp[i,:,:,:] == -2): w_tmp[i,:,:,:] = -get_ln_r(loc=loc_2, scale=scale_2)
                                    #         if(w_tmp[i,:,:,:] == -3): w_tmp[i,:,:,:] = -get_ln_r(loc=loc_3, scale=scale_3)
                                    #         if(w_tmp[i,:,:,:] == -4): w_tmp[i,:,:,:] = -get_ln_r(loc=loc_4, scale=scale_4)
                                    #         if(w_tmp[i,:,:,:] == -5): w_tmp[i,:,:,:] = -get_ln_r(loc=loc_5, scale=scale_5)
                                    #         if(w_tmp[i,:,:,:] == -6): w_tmp[i,:,:,:] = -get_ln_r(loc=loc_6, scale=scale_6)
                                    #         if(w_tmp[i,:,:,:] == -7): w_tmp[i,:,:,:] = -get_ln_r(loc=loc_7, scale=scale_7)
                                    #         if(w_tmp[i,:,:,:] == -8): w_tmp[i,:,:,:] = -get_ln_r(loc=loc_8, scale=scale_8)
                                    #         if(w_tmp[i,:,:,:] == -9): w_tmp[i,:,:,:] = -get_ln_r(loc=loc_9, scale=scale_9)
                                    #         if(w_tmp[i,:,:,:] == -10): w_tmp[i,:,:,:] = -get_ln_r(loc=loc_10, scale=scale_10)
                                    #         if(w_tmp[i,:,:,:] == -11): w_tmp[i,:,:,:] = -get_ln_r(loc=loc_11, scale=scale_11)
                                    #         if(w_tmp[i,:,:,:] == -12): w_tmp[i,:,:,:] = -get_ln_r(loc=loc_12, scale=scale_12)
                                    #         if(w_tmp[i,:,:,:] == -13): w_tmp[i,:,:,:] = -get_ln_r(loc=loc_13, scale=scale_13)
                                    #         if(w_tmp[i,:,:,:] == -14): w_tmp[i,:,:,:] = -get_ln_r(loc=loc_14, scale=scale_14)
                                    #         if(w_tmp[i,:,:,:] == -15): w_tmp[i,:,:,:] = -get_ln_r(loc=loc_15, scale=scale_15)

                                res_tmp = F.conv2d(a_tmp.float(), w_tmp.float(), bias=None, stride=self.wrapped_module.stride, padding=0)
                                # print('after', res_tmp)


                                res_sliced_tmp = res_sliced_tmp + (res_tmp<<(ba*2)).float()#!!!!!!!
                    # bit_answer = (2**bit_activ-1)*(2**(bit_weight-1)-1)
                    # if(R_off):
                    if(R_on):
                        # print(res_sliced_tmp)
                        res_sliced_tmp /= loc_1
                        # res_sliced_tmp /= 0.001
                        res_sliced_tmp = torch.round(res_sliced_tmp)
                        # print(res_sliced_tmp)
                    # exit(0)
                    res_sliced = res_sliced + res_sliced_tmp


                if self.conv_q.zero_point != 0:
                    #print("self.conv_q.zero_point:", self.conv_q.zero_point)
                    in_tmp_b = torch.ones_like(in_a)*self.conv_q.zero_point
                    res_tmp_b = F.conv2d(in_tmp_b, weight_tmp, stride = self.wrapped_module.stride, padding = self.wrapped_module.padding)
                    res_sliced = res_sliced + res_tmp_b
                #res2 = F.conv2d(input_tmp+self.conv_q.zero_point, weight_tmp, padding=1)
                #print("wb:", (res2-res_sliced).std())
                
                res_sliced = res_sliced / scale_wa
                if self.wrapped_module.bias is not None:
                    res_sliced = res_sliced + self.wrapped_module.bias.reshape(1, -1, 1, 1)
                # print('END2')
                # print('res_sliced', res_sliced.size())
                # print(res_sliced)
                return res_sliced
            else:#����kernel
                #return res
                # exit(0)
                para = paranumber
                res2 = F.linear(input_tmp, weight_tmp, bias=None)
                paralevel = self.wrapped_module.in_features // para
                last_element = self.wrapped_module.in_features % para
                # print(last_element)
                for cib in range(0, paralevel):
                    res_sliced_tmp = torch.cuda.FloatTensor(res.shape).fill_(0)

                    in_cis = input_tmp[:, cib*para:cib*para+para]
                    w_cis = weight_tmp[:, cib*para:cib*para+para] 
                    #maybe w_cis = weight_tmp[cib*para:cib*para+para, :] 
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
                            # for i in range(0, w_tmp.size(0)):
                            #     for j in range(0, w_tmp.size(1)):
                            #         if(w_tmp[i,j] == 0): w_tmp[i,j] = get_ln_r()
                            #         if(w_tmp[i,j] == 1): w_tmp[i,j] = get_hn_r()
                            # #         # if(w_cis[i,j] == -1): w_cis[i,j] = -(get_hn_r())
                            # # print('w_tmp_Ron',w_tmp)

                        # w_tmp_0, w_tmp_1 = w_tmp.split(para//2, dim = 1)
                        a_tmp = (in_cis.int() & (3<<(ba*2))) >> (ba*2)
                        a_nz = (a_tmp!=0).sum(dim=1, keepdim=True) 
                        
                        res_tmp = F.linear(a_tmp.float(), w_tmp.float(), bias=None)


                        res_sliced_tmp = res_sliced_tmp + (res_tmp<<(2*ba)).float()
                    # bit_answer = (2**bit_activ-1)*(2**(bit_weight-1)-1)
                    # if(R_off):
                    if(R_on):
                        res_sliced_tmp /= loc_1
                        # res_sliced_tmp /= 0.001
                        res_sliced_tmp = torch.round(res_sliced_tmp)
                    res_sliced = res_sliced + res_sliced_tmp
                
                if(last_element != 0):
                    res_sliced_tmp = torch.cuda.FloatTensor(res.shape).fill_(0)
                    # print('c & r:', paralevel*para, paralevel*para+last_element)
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
                            # for i in range(0, w_tmp.size(0)):
                            #     for j in range(0, w_tmp.size(1)):
                            #         if(w_tmp[i,j] == 0): w_tmp[i,j] = get_ln_r()
                            #         if(w_tmp[i,j] == 1): w_tmp[i,j] = get_hn_r()
                            # #         # if(w_cis[i,j] == -1): w_cis[i,j] = -(get_hn_r())
                            # # print('w_tmp_Ron',w_tmp)

                        # w_tmp_0, w_tmp_1 = w_tmp.split(para//2, dim = 1)
                        a_tmp = (in_cis.int() & (3<<(ba*2))) >> (ba*2)
                        a_nz = (a_tmp!=0).sum(dim=1, keepdim=True) 
                        
                        res_tmp = F.linear(a_tmp.float(), w_tmp.float(), bias=None)

                        res_sliced_tmp = res_sliced_tmp+(res_tmp<<(2*ba)).float()
                    # bit_answer = (2**bit_activ-1)*(2**(bit_weight-1)-1)
                    # if(R_off):
                    if(R_on):
                        res_sliced_tmp /= loc_1
                        # res_sliced_tmp /= 0.001
                        res_sliced_tmp = torch.round(res_sliced_tmp)
                    res_sliced = res_sliced + res_sliced_tmp
                res_sliced = res_sliced / scale_wa
                res_sliced = res_sliced + self.wrapped_module.bias
                # print('END3')
                # print('res_sliced', res_sliced.size())
                # print(res_sliced)
                return res_sliced

        elif mode == "bit_channel":
            torch.backends.cudnn.deterministic = True
            thre = 7 
            ad_limit = 15
            bit_weight = 4
            bit_activ = 4
            # para = 18
            paranumber = 100

            scale_w = self.wrapped_module.weight_scale
            scale_a = self.conv_q.scale
            scale_wa = scale_w * scale_a
            print('****************************************************************************************')
            print('w,a,wa',scale_w, scale_a, scale_wa)
            print(self.wrapped_module)
            ber = [0,8.61527E-13,4.18694E-07,6.1348E-05,0.000751256,0.004207439,0.012610766,0.024916498,0.045734124,0.072046802,0.104019711,0.140023727,0.174666477,0.201189703,0.22912297,0.112170035,1]
            ber = torch.tensor(ber).cuda()

            input_tmp = torch.round(input[0] * scale_a - self.conv_q.zero_point)
            input_tmp = torch.clamp(input_tmp, 0, (2**bit_activ)-1)
            print('input_tmp.size', input_tmp.size())
            weight_tmp = torch.round(self.wrapped_module.weight*scale_w)
            
            # print('weight', self.wrapped_module.weight)
            ce = self.wrapped_module.weight*scale_w
            # print('max',torch.max(ce))
            # print('weight*scale',self.wrapped_module.weight*scale_w)
            print('weight_tmp',weight_tmp.size())
            wnz = (weight_tmp.view(-1) == 0).sum()*1.0/weight_tmp.numel()
            #print("wnz:",wnz)
            in_a = self.conv_q(*input)
            # print('in_a',in_a)
            res = self.wrapped_module(in_a)
            res_sliced = torch.cuda.FloatTensor(res.shape).fill_(0) 
            print('res_sliced_ideal',res_sliced.size())


            if hasattr(self.wrapped_module, 'kernel_size'):
                # if (self.wrapped_module.in_channels == 31):
                #return res
                kernel_size = self.wrapped_module.kernel_size[0] * self.wrapped_module.kernel_size[1]
                para = paranumber // kernel_size
                para_remain = paranumber % kernel_size
                print('channel', self.wrapped_module.in_channels)
                paralevel = (self.wrapped_module.in_channels * kernel_size) // paranumber
                last_number = (self.wrapped_module.in_channels * kernel_size) % paranumber
                paralevel += last_number & 1
                print('last_number', (last_number & 1))
                print('paralevel', paralevel)
                # print('lastchannel', last_channel)
                # if(paralevel==0): paralevel += 1
                first_number = 0
                middle_number = 0
                remain_number = 0
                begin_channel = 0
                end_channel = 0
                tmp_channel = 0
                for cib in range(0, paralevel):
                    # print('��{}��'.format(cib))
                    # print('in_size & w_size',input_tmp.size(), weight_tmp.size())
                    # print('c & r:', cib*para, cib*para+para)
                    arrayrow = min(paranumber, kernel_size * self.wrapped_module.in_channels - cib*paranumber)
                    first_number = (kernel_size - remain_number) % kernel_size
                    middle_number = ((arrayrow - first_number) // kernel_size) * kernel_size
                    remain_number = arrayrow - first_number - middle_number

                    # in_cis = input_tmp[:, cib*para:cib*para+para, :, :]
                    # w_cis = weight_tmp[:, cib*para:cib*para+para, :, :]
                    # print('in_cis_size & w_cis_size',in_cis.size(), w_cis.size())
                    # print('in_cis', in_cis.size())
                    # print('kenel_size',self.wrapped_module.kernel_size)
                    if(first_number != 0):
                        in_cis = input_tmp[:, tmp_channel:tmp_channel+1, :, :]
                        w_cis = weight_tmp[:, tmp_channel:tmp_channel+1, :, :]
                        tmp_channel += 1

                        for num_tmp in range(kernel_size-first_number, kernel_size):
                            krow = num_tmp // self.wrapped_module.kernel_size[1]
                            kcol = num_tmp % self.wrapped_module.kernel_size[0]
                            # print('padding',self.wrapped_module.padding)
                            m = nn.ZeroPad2d((self.wrapped_module.padding[0] - kcol, self.wrapped_module.padding[0]-(self.wrapped_module.kernel_size[1]-kcol-1), self.wrapped_module.padding[1] - krow, self.wrapped_module.padding[1]-(self.wrapped_module.kernel_size[0]-krow-1)))
                            a_pad = m(in_cis)

                            for ba in range(0, bit_activ//2):#��֧��2������������
                                for bw in range(0, bit_weight):
                                    # print('ba & bw', ba, bw)
                                    # print('w_tmp_size',)
                                    w_tmp = ((w_cis[:,:,krow:krow+1,kcol:kcol+1]+(1<<(bit_weight))).int() & (1<<bw)) >> bw
                                    #ȡ��1����λ��weight,���ֱ�ȡ��1bit

                                    if(R_on):
                                        w_tmp[w_tmp==0] = low_condu
                                        w_tmp[w_tmp==1] = high_condu

                                    a_tmp = (a_pad.int() & (3<<(ba*2))) >> (ba*2)#����λȡ�����ֵ

                                    res_tmp = F.conv2d(a_tmp.float(), w_tmp.float(), bias=None, stride=self.wrapped_module.stride, padding=0)

                                    if bw == bit_weight-1:
                                        res_tmp = -res_tmp

                                    res_sliced = res_sliced + (res_tmp.int()<<(ba*2+bw)).float()#!!!!!!!
                    mid_channel = int(middle_number/kernel_size)
                    print('mid_channel', mid_channel)
                    print(type(mid_channel))
                    in_cis = input_tmp[:, tmp_channel:tmp_channel+mid_channel, :, :]
                    w_cis = weight_tmp[:, tmp_channel:tmp_channel+mid_channel, :, :]
                    tmp_channel = tmp_channel + mid_channel

                    for krow in range(0, self.wrapped_module.kernel_size[0]):
                        for kcol in range(0, self.wrapped_module.kernel_size[1]):
                            # print('padding',self.wrapped_module.padding)
                            m = nn.ZeroPad2d((self.wrapped_module.padding[0] - kcol, self.wrapped_module.padding[0]-(self.wrapped_module.kernel_size[1]-kcol-1), self.wrapped_module.padding[1] - krow, self.wrapped_module.padding[1]-(self.wrapped_module.kernel_size[0]-krow-1)))
                            # print('pad_m',m)
                            a_pad = m(in_cis)
                            # a_pad = in_cis
                            # print('m',m)
                            # print('in_cis', in_cis.size())
                            # print('a_pad.size',a_pad.size())
                            for ba in range(0, bit_activ//2):#��֧��2������������
                                for bw in range(0, bit_weight):
                                    # print('ba & bw', ba, bw)
                                    # print('w_tmp_size',)
                                    w_tmp = ((w_cis[:,:,krow:krow+1,kcol:kcol+1]+(1<<(bit_weight))).int() & (1<<bw)) >> bw
                                    # print('w_cis', w_cis[0,0,krow:krow+1,kcol:kcol+1])
                                    # print('w_cis+1', w_cis[0,0,krow:krow+1,kcol:kcol+1]+(1<<(bit_weight)))
                                    # print('w_tmp', w_tmp[0,0,krow:krow+1,kcol:kcol+1])
                                    # w_tmp = ((w_cis[:,:,krow,kcol]+(1<<(bit_weight))).int() & (1<<bw)) >> bw
                                    # print('w_tmp_size', w_tmp.size(), w_tmp)
                                    #ȡ��1����λ��weight,���ֱ�ȡ��1bit
                                    if(R_on):
                                        w_tmp[w_tmp==0] = low_condu
                                        w_tmp[w_tmp==1] = high_condu

                                    a_tmp = (a_pad.int() & (3<<(ba*2))) >> (ba*2)#����λȡ�����ֵ
                                    a_nz_t = (a_tmp!=0).sum(dim=1, keepdim=True)  
                                    # print(a_nz_t)
                                    # print(a_nz_t.size())
                                    # print(a_nz_t.shape[2],a_nz_t.shape[3],self.wrapped_module.stride[0],self.wrapped_module.stride[1])
                                    a_nz = a_nz_t[:,:,0:a_nz_t.shape[2]:self.wrapped_module.stride[0],0:a_nz_t.shape[3]:self.wrapped_module.stride[1]]#������ȡ��������Ҫֵ��1�ĸ�����
                                    # print('a_tmp',a_tmp.size())
                                    # print('w_tmp',w_tmp.size())
                                    # a_tmp_0, a_tmp_1 = a_tmp.split(para//2, dim = 1)#�ڶ���ά�ȶ԰����������
                                    # print(a_tmp.size(),a_tmp_0.size(), a_tmp_1.size())
                                    # print('----')
                                    # w_tmp_0, w_tmp_1 = w_tmp.split(para//2, dim = 1)
                                    # print(w_tmp.size(),w_tmp_0.size(),w_tmp_1.size())
                                    # print('a_tmp_0',a_tmp_0.size())
                                    # print('a_tmp_1',a_tmp_1.size())
                                    # print('w_tmp_0',w_tmp_0.size())
                                    # print('w_tmp_1',w_tmp_1.size())

                                    # res_tmp_0 = F.conv2d(a_tmp_0.float(), w_tmp_0.float(), bias=None, stride=self.wrapped_module.stride, padding=0)
                                    # res_tmp_1 = F.conv2d(a_tmp_1.float(), w_tmp_1.float(), bias=None, stride=self.wrapped_module.stride, padding=0)

                                    # print('res_tmp_0',res_tmp_0.size())
                                    # print('res_tmp_1',res_tmp_1.size())

                                    #16bit 4bitADC�������Ϊ16
                                    # res_tmp_0 = torch.where(res_tmp_0 > ad_limit, 15*torch.ones_like(res_tmp_0), res_tmp_0)
                                    # res_tmp_1 = torch.where(res_tmp_1 > ad_limit, 15*torch.ones_like(res_tmp_0), res_tmp_1)
                                    
                                    # print('res_tmp_0',res_tmp_0.size())
                                    # print('res_tmp_1',res_tmp_1.size())
                                    # res_tmp = res_tmp_0 + res_tmp_1
                                    # print('res_before',res_tmp)


                                    res_tmp = F.conv2d(a_tmp.float(), w_tmp.float(), bias=None, stride=self.wrapped_module.stride, padding=0)
                                    # res_tmp = torch.where(res_tmp > ad_limit, 15*torch.ones_like(res_tmp), res_tmp)
                                    # print(res_com)
                                    # print(res_com==(res_tmp_0+res_tmp_1))
                                    # print(res_com.size())
                                    # print((res_tmp_0+res_tmp_1).size())

                                    

                                    #res_tmp_2 = F.conv2d(a_tmp.float(), w_tmp.float(), bias=None, stride=self.wrapped_module.stride, padding=0)
                                    #print(a_nz.shape)
                                    #print(res_tmp.shape)
                                    # truc_flag = (a_nz > thre) & (res_tmp > 15)

                                    # ber_sampled = ber[res_tmp.long()]
                                    # res_tmp += ber_sampled

                                    # err_prob = torch.cuda.FloatTensor(res_tmp.shape).fill_(0)
                                    # err_prob = torch.rand(size=res_tmp.shape, out=err_prob)
                                    # err = torch.where(err_prob<ber_sampled, torch.ones_like(res_tmp), torch.zeros_like(res_tmp))
                                    # err = torch.where(err_prob<ber_sampled/2, err, -err)
                                    # print('res_tmp.long', res_tmp.long())
                                    # print('ber_tmp',ber_tmp)
                                    #res_tmp = res_tmp_0 + res_tmp_1
                                    #res_tmp = res_tmp_2
                                    if bw == bit_weight-1:
                                        res_tmp = -res_tmp
                                    # print('res_sliced:',res_sliced.size())
                                    # print('res_tmp:',res_tmp.int().size())
                                    # print(ba*2+bw)
                                    # print((res_tmp.int()<<(ba*2+bw)).float().size())
                                    res_sliced = res_sliced + (res_tmp.int()<<(ba*2+bw)).float()#!!!!!!!
                    if(remain_number != 0):
                        in_cis = input_tmp[:, tmp_channel:tmp_channel+1, :, :]
                        w_cis = weight_tmp[:, tmp_channel:tmp_channel+1, :, :]
                        tmp_channel += 1

                        for num_tmp in range(0, remain_number):
                            krow = num_tmp // self.wrapped_module.kernel_size[1]
                            kcol = num_tmp % self.wrapped_module.kernel_size[0]
                            # print('padding',self.wrapped_module.padding)
                            m = nn.ZeroPad2d((self.wrapped_module.padding[0] - kcol, self.wrapped_module.padding[0]-(self.wrapped_module.kernel_size[1]-kcol-1), self.wrapped_module.padding[1] - krow, self.wrapped_module.padding[1]-(self.wrapped_module.kernel_size[0]-krow-1)))
                            a_pad = m(in_cis)

                            for ba in range(0, bit_activ//2):#��֧��2������������
                                for bw in range(0, bit_weight):
                                    # print('ba & bw', ba, bw)
                                    # print('w_tmp_size',)
                                    w_tmp = ((w_cis[:,:,krow:krow+1,kcol:kcol+1]+(1<<(bit_weight))).int() & (1<<bw)) >> bw
                                    #ȡ��1����λ��weight,���ֱ�ȡ��1bit

                                    if(R_on):
                                        w_tmp[w_tmp==0] = low_condu
                                        w_tmp[w_tmp==1] = high_condu

                                    a_tmp = (a_pad.int() & (3<<(ba*2))) >> (ba*2)#����λȡ�����ֵ

                                    res_tmp = F.conv2d(a_tmp.float(), w_tmp.float(), bias=None, stride=self.wrapped_module.stride, padding=0)

                                    if bw == bit_weight-1:
                                        res_tmp = -res_tmp

                                    res_sliced = res_sliced + (res_tmp.int()<<(ba*2+bw)).float()#!!!!!!!

                bit_answer = (2**bit_activ-1)*(2**bit_weight-1)
                if(R_on):
                    for i in range(0, bit_answer*paranumber):
                        res_sliced[((i*high_condu)<=res_sliced) & (res_sliced < (i+1)*high_condu)] = i



                # if(last_channel != 0):
                #     # print('c & r:', paralevel*para, paralevel*para+last_channel)
                #     in_cis = input_tmp[:, paralevel*para:paralevel*para+last_channel, :, :]
                #     w_cis = weight_tmp[:, paralevel*para:paralevel*para+last_channel, :, :]
                #     for krow in range(0, self.wrapped_module.kernel_size[0]):
                #         for kcol in range(0, self.wrapped_module.kernel_size[1]):
                #             # print('padding',self.wrapped_module.padding)
                #             m = nn.ZeroPad2d((self.wrapped_module.padding[0] - kcol, self.wrapped_module.padding[0]-(self.wrapped_module.kernel_size[1]-kcol-1), self.wrapped_module.padding[1] - krow, self.wrapped_module.padding[1]-(self.wrapped_module.kernel_size[0]-krow-1)))
                #             # print('pad_m',m)
                #             a_pad = m(in_cis)
                #             for ba in range(0, bit_activ//2):
                #                 for bw in range(0, bit_weight):
                #                     w_tmp = ((w_cis[:,:,krow:krow+1,kcol:kcol+1]+(1<<(bit_weight))).int() & (1<<bw)) >> bw
                #                     a_tmp = (a_pad.int() & (3<<(ba*2))) >> (ba*2)
                #                     res_tmp = F.conv2d(a_tmp.float(), w_tmp.float(), bias=None, stride=self.wrapped_module.stride, padding=0)
                #                     # print('tmp:',res_tmp.size())
                                    
                #                     adc = False
                #                     if(adc):
                #                         ber_sampled = ber[res_tmp.long()]
                #                         err_prob = torch.cuda.FloatTensor(res_tmp.shape).fill_(0)
                #                         err_prob = torch.rand(size=res_tmp.shape, out=err_prob)
                                        
                #                         err = torch.where(err_prob<ber_sampled, torch.ones_like(res_tmp), torch.zeros_like(res_tmp))
                #                         err = torch.where(err_prob<ber_sampled/2, err, -err)
                #                         res_tmp = res_tmp+err


                #                     #print("pred:",err.std()**2)
                #                     #print("misc:",err.mean())
                #                     #print((err!=0).float().mean())
                #                     #print("theo:",ber_sampled.mean())
                #                     #exit(0)
                #                     if bw == bit_weight-1:
                #                         res_tmp = -res_tmp
                #                     res_sliced = res_sliced + (res_tmp.int()<<(ba*2+bw)).float()

                if self.conv_q.zero_point != 0:
                    #print("self.conv_q.zero_point:", self.conv_q.zero_point)
                    in_tmp_b = torch.ones_like(in_a)*self.conv_q.zero_point
                    res_tmp_b = F.conv2d(in_tmp_b, weight_tmp, stride = self.wrapped_module.stride, padding = self.wrapped_module.padding)
                    res_sliced = res_sliced + res_tmp_b
                #res2 = F.conv2d(input_tmp+self.conv_q.zero_point, weight_tmp, padding=1)
                #print("wb:", (res2-res_sliced).std())
                
                res_sliced = res_sliced / scale_wa
                #print((res-res_sliced).std())
                #exit(0)
                #if (self.wrapped_module.bias is not None):
                #    print(self.wrapped_module.bias)
                #    exit(0)
                if self.wrapped_module.bias is not None:
                    res_sliced = res_sliced + self.wrapped_module.bias.reshape(1, -1, 1, 1)
                print('END2')
                print('res_sliced', res_sliced.size())
                # print(res_sliced)
                return res_sliced
            else:#����kernel
                #return res
                para = 100
                res2 = F.linear(input_tmp, weight_tmp, bias=None)
                paralevel = self.wrapped_module.in_features // para
                last_element = self.wrapped_module.in_features % para
                print(last_element)
                for cib in range(0, paralevel):
                    #print(input_tmp.shape)
                    #print(weight_tmp.shape)
                    #exit(0)
                    # print('c & r:', cib*para, cib*para+para)
                    in_cis = input_tmp[:, cib*para:cib*para+para]
                    w_cis = weight_tmp[:, cib*para:cib*para+para] 
                    for ba in range(0, bit_activ // 2):
                        for bw in range(0, bit_weight):
                            w_tmp = ((w_cis+(1<<(bit_weight))).int() & (1<<bw)) >> bw

                            if(R_on):
                                w_tmp[w_tmp==0] = low_condu
                                w_tmp[w_tmp==1] = high_condu

                            # w_tmp_0, w_tmp_1 = w_tmp.split(para//2, dim = 1)
                            a_tmp = (in_cis.int() & (3<<(ba*2))) >> (ba*2)
                            a_nz = (a_tmp!=0).sum(dim=1, keepdim=True) 
                            
                            res_tmp = F.linear(a_tmp.float(), w_tmp.float(), bias=None)
                            # a_tmp_0, a_tmp_1 = a_tmp.split(para//2, dim = 1)
                            # res_tmp_0 = F.linear(a_tmp_0.float(), w_tmp_0.float(), bias=None)
                            # res_tmp_1 = F.linear(a_tmp_1.float(), w_tmp_1.float(), bias=None)
                            # res_tmp_0 = torch.where(res_tmp_0>15, 15*torch.ones_like(res_tmp_0), res_tmp_0)
                            # res_tmp_1 = torch.where(res_tmp_1>15, 15*torch.ones_like(res_tmp_1), res_tmp_1)
                            # res_tmp = res_tmp_0 + res_tmp_1


                            if bw == bit_weight-1:
                                res_tmp = -res_tmp
                            # print('``````````````````````')
                            # print(res_sliced)
                            # print(res_tmp)
                            # print((res_tmp.int()<<(2*ba+bw)).float())
                            # print('``````````````````````')
                            res_sliced = res_sliced+(res_tmp.int()<<(2*ba+bw)).float()
                
                if(last_element != 0):
                    # print('c & r:', paralevel*para, paralevel*para+last_element)
                    in_cis = input_tmp[:, paralevel*para:paralevel*para+last_element]
                    w_cis = weight_tmp[:, paralevel*para:paralevel*para+last_element]
                    for ba in range(0, bit_activ // 2):
                        for bw in range(0, bit_weight):
                            w_tmp = ((w_cis+(1<<(bit_weight))).int() & (1<<bw)) >> bw

                            if(R_on):
                                w_tmp[w_tmp==0] = low_condu
                                w_tmp[w_tmp==1] = high_condu

                            # w_tmp_0, w_tmp_1 = w_tmp.split(para//2, dim = 1)
                            a_tmp = (in_cis.int() & (3<<(ba*2))) >> (ba*2)
                            a_nz = (a_tmp!=0).sum(dim=1, keepdim=True) 
                            
                            res_tmp = F.linear(a_tmp.float(), w_tmp.float(), bias=None)
                            if bw == bit_weight-1:
                                res_tmp = -res_tmp

                            res_sliced = res_sliced+(res_tmp.int()<<(2*ba+bw)).float()
                bit_answer = (2**bit_activ-1)*(2**bit_weight-1)
                if(R_on):
                    for i in range(0, bit_answer*para):
                        res_sliced[((i*high_condu)<=res_sliced) & (res_sliced < (i+1)*high_condu)] = i
                    
                     

                #print((res_sliced-res2).std())
                #exit(0)
                res_sliced = res_sliced / scale_wa
                res_sliced = res_sliced + self.wrapped_module.bias
                print('END3')
                print('res_sliced', res_sliced.size())
                # print(res_sliced)
                return res_sliced
        
            '''
            return res

            torch.set_printoptions(precision=8)
            print(self.wrapped_module.weight.shape)
            r_t = torch.tensor(0)
            for i in range(0, 64):
                test_res = (res-res_sliced)[0,i,:,:]
                if len(test_res.unique()) > 1:
                    for c in range(0, 3):
                        for x in range(0,3):
                            for y in range(0,2):
                                print("in_a(",c,",",x,",",y,"):", in_a[0,c,x,y])
                                print("w(",c,",",x,",",y,"):", self.wrapped_module.weight[i,c,x,y+1])
                                r_t = r_t + in_a[0,c,x,y]*self.wrapped_module.weight[i,c,x,y+1]

                    print(r_t)
                    print(res_sliced[0,i,1,0])
                    print(res[0,i,1,0])
                    print("channel:", i)
                    print(res.shape)
                    print(len(test_res.unique())) 
                    print(test_res.unique())
                    print(test_res)
                    exit(0)
            print("succ")
            '''

            exit(0)      

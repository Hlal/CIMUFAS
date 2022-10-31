import yaml
import torch.nn as nn

class Config:
    def __init__(self, config_file):
        """
        read config file
        """
        config_dir = config_file
        try:
            with open(config_dir, "r") as stream:
                raw_dict = yaml.load(stream)

                # netmodel:
                    # arch:
                        # resnet18
                    # dataset:
                        # cifar10
                    # data:
                        # ./data/
                    # load_path:
                        # './checkpoint_resnet18_imagenet/exp6/quanted_b44_cosine_warmup.t7'
                    # bit_wt:
                        # 4
                    # bit_act:
                        # 4
                # simulation:
                    # mode:
                        # normal
                    # R_on:
                        # False
                    # R_SAF:
                        # False
                    # SA1:
                        # 0.09
                    # SA0:
                        # 0.99
                    # ADC_error:
                        # False
                    # para:
                        # 32
                # storage:
                    # rram_bit:
                        #   1
                #     rram_conductance:
                    #     - 706.25
                    #     - 2050
                #     rram_variation:
                    #     - 39.878
                    #     - 59.646
                self.arch = raw_dict['netmodel']['arch']
                self.dataset = raw_dict['netmodel']['dataset']
                self.data = raw_dict['netmodel']['data']
                self.load_path = raw_dict['netmodel']['load_path']
                self.bit_wt = raw_dict['netmodel']['bit_wt']
                self.bit_act = raw_dict['netmodel']['bit_act']
                self.mode = raw_dict['simulation']['mode']
                self.R_on = raw_dict['simulation']['R_on']
                self.R_SAF = raw_dict['simulation']['R_SAF']
                self.SA1 = raw_dict['simulation']['SA1']
                self.SA0 = raw_dict['simulation']['SA0']
                self.ADC_error = raw_dict['simulation']['ADC_error']
                self.ADC_delta = raw_dict['simulation']['ADC_delta']
                self.para = raw_dict['simulation']['para']
                self.rram_bit = raw_dict['storage']['rram_bit']
                self.rram_conductance = raw_dict['storage']['rram_conductance']
                self.rram_variation = raw_dict['storage']['rram_variation']
                # print((2**self.rram_bit))
                # print(len(self.rram_conductance))
                # print(self.rram_conductance[0])

                if 2**self.rram_bit != len(self.rram_conductance):
                    raise Exception("Config error : rram conductance's number does not match the number of bits provided")
                if len(self.rram_variation) != len(self.rram_conductance):
                    raise Exception("Config error : rram conductance's number does not match the number of rram variation")


                

        except yaml.YAMLError as exc:
            print(exc)

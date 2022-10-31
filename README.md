Welcome to CIMUFAS.

The 'config' file contains most of the parameter configuration.

The 'models' folder stores the network structure files.

The 'quantization' folder stores the quantization methods and simulation functions we use. Since the quantization function is not the content of this project, nor is it open-source content, we have encrypted it. The simulation function modifies the implementation of convolution and full connections, turning direct matrix multiplication into matrix computations that conform to on-chip logic. The difference in the quantization function does not affect the effectiveness of the simulation logic. Inside the 'simulation' file is our simulation framework code.

test.py is our executable file. After setting the parameters in 'config.yaml.txt', run test.sh directly to execute. We present a quantified model to help you use it.

##### config.yaml.txt

````yaml
netmodel:
#model para
 arch:
  resnet18
 dataset:
  cifar10
 data:
 #data is the storage location of the dataset
  ./data/
 load_path:
 #load_path is where the model is stored
  ./checkpoint_resnet18_cifar10/exp0/quanted_b44_cosine_warmup.t7
 bit_wt:
 #Weight quantization bits
  4
 bit_act:
 #activation quantization bits
  4
simulation:
 mode:
 #The execution mode of the emulator includes 'normal' 'bit_slice' 'bit_kernel' 'sparate_pn_kernel' 'varia_pn_kernel' 'fast_2mod' '4bit_rram'.
  varia_pn_kernel
 R_on:
 #Whether to enable simulation of rram storage (including variation)
  True
 R_SAF:
 #Whether to enable simulation of SAF problems
  False
 SA1:
 #Probability of SA1 problem
  0.09
 SA0:
 #Probability of SA0 problem
  0.99
 ADC_error:
 ##Whether to enable simulation of ADC error problems
  False
 ADC_delta:
 #The model given here is the Gaussian model used in the article
  10
 para:
 #The row parallelism of the array; if you want to set different row parallelisms for different layers, you need to change the input to an array, and determine the id of each layer in simulation.
  32
storage:
 rram_bit:
 #The number of rram bits; 1bit rram is used by default in most modes
  1
 rram_conductance:
 #high and low conductance of rram
  - 706.25
  - 2050
 rram_variation:
 #Variation of rram under Gaussian model
  - 39.878
  - 59.646
````

##### mode

'normal' :Normal mode, the standard accuracy of the input model (under our quantified results).

'bit_slice' : A mode for matrix multiplication by bit division; this mode simulates different bits of the same data stored in different columns in the same array; The mode of data storage is two's complement.

'bit_kernel' : In this mode, the order of weights in the same column is in accordance with the direction of the kernel; The mode of data storage is two's complement.

'sparate_pn_kernel': Simulation with separate positive and negative arrays; stored data is the true form.

 'varia_pn_kernel': Fast mode that simulates variation and saf problems only. Fast mode that simulates variation and saf problems only. However, possible ADC model insertion methods are also given.

 'fast_2mod' : Fast mode that simulates variation and saf problems only.

'4bit_rram': 4bit rram emulation. The method of multi-bit simulation is given.

##### Thanks for using CIMUFAS. Due to the version update, some codes may be incomplete; due to the long experimental process, there may be omissions in the code. If you find any errors in the code, please let me know.
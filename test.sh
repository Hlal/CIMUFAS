#CUDA_VISIBLE_DEVICES=3,4,5 python -u test.py | tee -a testlog/lenets2_mnist/hcj_10000.log
#CUDA_VISIBLE_DEVICES=6,7,8,9 python -u testcj.py | tee -a testlog/hcj.log
CUDA_VISIBLE_DEVICES=6,7,8,9 python test.py 
#CUDA_VISIBLE_DEVICES=1,2,3,4,5 python -u test.py | tee -a testlog/adc.log
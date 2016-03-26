import os
import sys
import subprocess

train_path = '/home/common/imagenet/train'
output_path = '/data/courses/iw16/jjliu/dcgan.torch/train_imagenet'

dirs = subprocess.check_output(['ls', '-1', train_path])
dirs_array = dirs.splitlines()
for d in dirs_array: 
    print "Processing " + d
    subprocess.call('wnid=' + d + ' DATA_ROOT=' + output_path + ' th crop_synset.lua', shell=True)


import os
import sys
import subprocess

orig_path = '/home/jjliu/celebA'
attr_path = os.path.join(orig_path, 'list_attr_celeba.txt')
img_path = os.path.join(orig_path, 'img_align_celeba')
dest_path = '/home/jjliu/dcgan.torch/train_celebA'

attrf = open(attr_path, 'r')
line = attrf.readline()
tokens = line.split()
male_index = -1
sm_index =  -1
is_male = {}
is_sm = {}

i = 0
for line in attrf:
    if i == 0:
        tokens = line.split()
        for j in range(len(tokens)):
            if tokens[j] == "Male":
                male_index = j
            elif tokens[j] == "Smiling":
                sm_index = j
        i += 1
        continue
    tokens = line.split()
    fname = tokens[0]
    if tokens[male_index+1] == "-1":
        is_male[fname] = False
    else: 
        is_male[fname] = True

    if tokens[sm_index+1] == "-1":
        is_sm[fname] = False
    else:
        is_sm[fname] = True

    i += 1

counts = {}
for key in is_male:
    male = is_male[key]
    smile = is_sm[key]
    if male and smile:
        outfolder = "ms"
    elif male and not smile:
        outfolder = "mn"
    elif not male and smile:
        outfolder = "fs"
    elif not male and not smile:
        outfolder = "fn"

    if outfolder in counts:
        counts[outfolder] += 1
    else:
        counts[outfolder] = 1

    if counts[outfolder] < 40000:
        src_img_path = os.path.join(img_path, key)
        dest_img_path = os.path.join(dest_path, outfolder)
        subprocess.call(["cp", src_img_path, dest_img_path])

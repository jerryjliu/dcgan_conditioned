import os
import sys

f = open('/home/jjliu/dcgan.torch/main_conditioned_celebs.log', 'r')
count = 1
for line in f:
  #print line
  if "Err_G" in line: 
    tokens = line.split()
    errg = -1
    errd = -1
    for i in range(len(tokens)):
      if tokens[i] == "Err_G:": 
        errg = float(tokens[i+1])
      elif tokens[i] == "Err_D:":
        errd = float(tokens[i+1])
    print(str(count) + "," + str(errg) + "," + str(errd))
    count += 1

sys.stdout.flush()

f.close()

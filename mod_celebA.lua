require 'torch'
require 'nn'
require 'image'

local noise = torch.DoubleTensor(32, 100, 1, 1)
print(type(noise))
noise_vis = noise:clone()
noise_vis:uniform(-1, 1)

util = paths.dofile('util.lua')
print("Hello world")
netG = util.load('checkpoints/experiment1_25_net_G.t7', 1)
netG:double()
print("Hello world2")
print(netG)
print(torch.type(noise_vis))
local imgOutput = netG:forward(noise_vis)
local imgOutput2 = imgOutput:select(1,1)
print("Hello world3")
print(torch.type(imgOutput2))
print(imgOutput2:nDimension())
image.save('test_image.png', image.toDisplayTensor(imgOutput))




require 'image'
require 'nn'
util = paths.dofile('util.lua')
torch.setdefaulttensortype('torch.FloatTensor')

word2vec_map_path = '/data/courses/iw16/jjliu/word2vec/word2vec/word2vec_output.txt'

opt = {
    batchSize = 100,        -- number of samples to produce
    noisetype = 'normal',  -- type of noise distribution (uniform / normal).
    net = '',              -- path to the generator network
    imsize = 1,            -- used to produce larger images. 1 = 64px. 2 = 80px, 3 = 96px, ...
    noisemode = 'random',  -- random / line / linefull1d / linefull
    name = 'generation1',  -- name of the file saved
    gpu = 1,               -- gpu mode. 0 = CPU, 1 = GPU
    display = 0,           -- Display image: 0 = false, 1 = true
    nz = 100,              
    nw = 80,
    nr = 20,
    word = "barrel",       -- the word on which to generate
    class_id = 1,          -- the class id on which to generate
    num_classes=3,         -- the total number of classes
    word2vec = 0,          -- toggles whether to generate based on word2vec (using word field)
                           -- or class_id (using class_id) field. Depends on training.
    old = 0,               -- determines dimensionality of input vector
}
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end

assert(net ~= '', 'provide a generator model')

word = opt.word
nz = opt.nz
nw = opt.nw
nr = opt.nr
local word2vec_vec = torch.Tensor(opt.batchSize, nw, 1, 1)
local word2vec_noise = torch.Tensor(opt.batchSize, nr, 1, 1)
if opt.old == 0 then
  word2vec_total = torch.Tensor(opt.batchSize, nz + nw, 1, 1)
else 
  word2vec_total = torch.Tensor(opt.batchSize, nz, 1, 1)
end

-- one-hot specific
local onehot_vec = torch.Tensor(opt.batchSize, opt.num_classes, 1, 1)
local onehot_total
if opt.old == 0 then 
  onehot_total = torch.Tensor(opt.batchSize, nz + opt.num_classes, 1, 1)
else 
  onehot_total = torch.Tensor(opt.batchSize, nz, 1, 1)
end

--noise = torch.Tensor(opt.batchSize, opt.nz, opt.imsize, opt.imsize)
net = util.load(opt.net, opt.gpu)

-- for older models, there was nn.View on the top
-- which is unnecessary, and hinders convolutional generations.
if torch.type(net:get(1)) == 'nn.View' then
    net:remove(1)
end

print(net)

Cond_Util = paths.dofile('data/conditional_util.lua')

-- build word2vec map
word2vec_map = Cond_Util.load_word2vec_map(word2vec_map_path, nw)

if opt.word2vec == 1 then
  -- word2vec_noise
  if opt.noisetype == 'uniform' then
      word2vec_noise:uniform(-1, 1)
  elseif opt.noisetype == 'normal' then
      word2vec_noise:normal(0, 1)
  end
  -- word2vec_vec
  if word2vec_map[word] ~= nil then
    print("word is valid")
    for i=1,opt.batchSize do
      word2vec_vec[{i,{}}] = word2vec_map[word]
    end
  else
    error("word is invalid: " .. word)
  end
  word2vec_total[{{}, {1,nw}, {}, {}}] = word2vec_vec
  if opt.old == 0 then
    word2vec_total[{{}, {nw+1, nw + nz}, {}, {}}] = word2vec_noise
  else 
    word2vec_total[{{}, {nw+1, nz}, {}, {}}] = word2vec_noise
  end
elseif opt.word2vec == 0 then
   onehot_vec:zero()
   for i=1, opt.batchSize do
     onehot_vec[{i, opt.class_id, 1, 1}] = 1.0
   end

   if opt.noisetype == 'uniform' then -- regenerate random noise
       onehot_total:uniform(-1, 1)
   elseif opt.noisetype == 'normal' then
       onehot_total:normal(0, 1)
   end
   -- insert one-hot encoding into first few entries
   if opt.old == 0 then
     onehot_total[{{}, {nz + 1,nz + opt.num_classes}, {}, {}}] = onehot_vec
   else
     onehot_total[{{}, {1, opt.num_classes}, {}, {}}] = onehot_vec
   end
   print(onehot_total[{{1,5}}])
   print("hello world")
end

--noiseL = torch.FloatTensor(opt.nz):uniform(-1, 1)
--noiseR = torch.FloatTensor(opt.nz):uniform(-1, 1)
--if opt.noisemode == 'line' then
   ---- do a linear interpolation in Z space between point A and point B
   ---- each sample in the mini-batch is a point on the line
    --line  = torch.linspace(0, 1, opt.batchSize)
    --for i = 1, opt.batchSize do
        --noise:select(1, i):copy(noiseL * line[i] + noiseR * (1 - line[i]))
    --end
--elseif opt.noisemode == 'linefull1d' then
   ---- do a linear interpolation in Z space between point A and point B
   ---- however, generate the samples convolutionally, so a giant image is produced
    --assert(opt.batchSize == 1, 'for linefull1d mode, give batchSize(1) and imsize > 1')
    --noise = noise:narrow(3, 1, 1):clone()
    --line  = torch.linspace(0, 1, opt.imsize)
    --for i = 1, opt.imsize do
        --noise:narrow(4, i, 1):copy(noiseL * line[i] + noiseR * (1 - line[i]))
    --end
--elseif opt.noisemode == 'linefull' then
   ---- just like linefull1d above, but try to do it in 2D
    --assert(opt.batchSize == 1, 'for linefull mode, give batchSize(1) and imsize > 1')
    --line  = torch.linspace(0, 1, opt.imsize)
    --for i = 1, opt.imsize do
        --noise:narrow(3, i, 1):narrow(4, i, 1):copy(noiseL * line[i] + noiseR * (1 - line[i]))
    --end
--end

if opt.gpu > 0 then
    require 'cunn'
    require 'cudnn'
    net:cuda()
    util.cudnn(net)
    --noise = noise:cuda()
    word2vec_noise = word2vec_noise:cuda()
    word2vec_vec = word2vec_vec:cuda()
    word2vec_total = word2vec_total:cuda()
    onehot_vec = onehot_vec:cuda()
    onehot_total = onehot_total:cuda()
else
   net:float()
end

-- a function to setup double-buffering across the network.
-- this drastically reduces the memory needed to generate samples
util.optimizeInferenceMemory(net)

local images = nil
if opt.word2vec == 1 then
  print(word2vec_total)
  images = net:forward(word2vec_total)
elseif opt.word2vec == 0 then
  images = net:forward(onehot_total)
end

print('Images size: ', images:size(1)..' x '..images:size(2) ..' x '..images:size(3)..' x '..images:size(4))
images:add(1):mul(0.5)
--print(images)
print('Min, Max, Mean, Stdv', images:min(), images:max(), images:mean(), images:std())
image.save(opt.name .. '.png', image.toDisplayTensor(images))
print('Saved image to: ', opt.name .. '.png')

if opt.display then
    disp = require 'display'
    disp.image(images)
    print('Displayed image')
end


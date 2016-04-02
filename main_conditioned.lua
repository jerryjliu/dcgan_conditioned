require 'torch'
require 'nn'
require 'optim'

word2vec_map_path = '/data/courses/iw16/jjliu/word2vec/word2vec/word2vec_output.txt'
synset_map_path = '/data/courses/iw16/jjliu/imagenet/wnid_synset_map.txt'

util = paths.dofile('util.lua')

opt = {
   dataset = 'lsun',       -- imagenet / lsun / folder
   batchSize = 64,
   loadSize = 96,
   fineSize = 64,
   nz = 100,               -- #  of dim for Z (nz = nw + nr)
   nw = 80,                -- # of dim for word2vec
   nr = 20,                -- # of dim for noise  
   ngf = 64,               -- #  of gen filters in first conv layer
   ndf = 64,               -- #  of discrim filters in first conv layer
   nThreads = 4,           -- #  of data loading threads to use
   niter = 50,             -- #  of iter at starting learning rate
   --lr = 0.0002,            -- initial learning rate for adam
   lr=0.0002,
   beta1 = 0.5,            -- momentum term of adam
   ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
   display = 0,            -- display samples while training. 0 = false
   display_id = 10,        -- display window id.
   gpu = 2,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
   name = 'experiment1',
   noise = 'normal',       -- uniform / normal
   wnid = 0,               -- jerry: wnid=1 means training folders are labeled by their
                           -- synset id's. wnid = 0 means they're labeled by words.
   word2vec = 0,           -- jerry: word2vec=1 means conditional vectors are word2vec embeddings 
                           -- word2vec = 0 means they are one-hot encodings
}

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end

opt.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- create data loader
local DataLoader = paths.dofile('data/data.lua')
local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
print("Dataset: " .. opt.dataset, " Size: ", data:size())
----------------------------------------------------------------------------
local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m.bias:fill(0)
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end

local nc = 3
local nz = opt.nz

local nw = opt.nw
local nr = opt.nr

local ndf = opt.ndf
local ngf = opt.ngf
local real_label = 1
local fake_label = 0

local SpatialBatchNormalization = nn.SpatialBatchNormalization
local SpatialConvolution = nn.SpatialConvolution
local SpatialFullConvolution = nn.SpatialFullConvolution

local netG = nn.Sequential()
-- input is Z, going into a convolution
netG:add(SpatialFullConvolution(nz, ngf * 8, 4, 4))
netG:add(SpatialBatchNormalization(ngf * 8)):add(nn.ReLU(true))
-- state size: (ngf*8) x 4 x 4
netG:add(SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))
-- state size: (ngf*4) x 8 x 8
netG:add(SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))
-- state size: (ngf*2) x 16 x 16
netG:add(SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
-- state size: (ngf) x 32 x 32
netG:add(SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1))
netG:add(nn.Tanh())
-- state size: (nc) x 64 x 64

netG:apply(weights_init)

-- modified discriminator that applies conditioning
local netD_C = nn.Sequential()
local netD_par = nn.ParallelTable()

local netD = nn.Sequential()
-- input is (nc) x 64 x 64
netD:add(SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))
netD:add(nn.LeakyReLU(0.2, true))
-- state size: (ndf) x 32 x 32
netD:add(SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*2) x 16 x 16
netD:add(SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*4) x 8 x 8
netD:add(SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
-- state size: (ndf*8) x 4 x 4
--netD:add(SpatialConvolution(ndf * 8, numClasses, 4, 4))
netD:add(SpatialConvolution(ndf * 8, data:numClasses() * 2, 4, 4))
-- state size: numClasses x 1 x 1
netD:add(nn.View(data:numClasses() * 2):setNumInputDims(3))
---- state size: numClasses
netD:apply(weights_init)

-- input is {(nc)x64x64, numClassesx1x1}
netD_par:add(netD)
netD_par:add(nn.View(data:numClasses()):setNumInputDims(3))

-- input is {(nc)x64x64, numClassesx1x1}
netD_C:add(netD_par)
-- table of dim {numClasses, numClasses}
netD_C:add(nn.JoinTable(2)) -- dim 2 b/c dim 1 is the batch number
-- tensor of size (numClasses * 2)
netD_C:add(nn.Linear(data:numClasses() * 3, 1))
 --state size: 1
--netD_C:add(nn.JoinTable(2)) -- dim 2 b/c dim 1 is the batch number
---- tensor of size (numClasses * 2)x1x1
--netD_C:add(nn.View(data:numClasses() + 1))
---- tensor of size (numClasses + 1)
--netD_C:add(nn.Linear(data:numClasses() + 1, 1))
---- linear transformation to output value, dim: 1
netD_C:add(nn.Sigmoid())
-- apply sigmoid to output value, dim: 1

local criterion = nn.BCECriterion()
---------------------------------------------------------------------------
optimStateG = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
optimStateD = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
----------------------------------------------------------------------------
local input = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
local noise = torch.Tensor(opt.batchSize, nz, 1, 1)

-- word2vec specific
local word2vec_vec = torch.Tensor(opt.batchSize, nw, 1, 1)
local word2vec_noise = torch.Tensor(opt.batchSize, nr, 1, 1)
local word2vec_total = torch.Tensor(opt.batchSize, nz, 1, 1)

-- one-hot specific
local onehot_vec = torch.Tensor(opt.batchSize, data:numClasses(), 1, 1)
local onehot_total = torch.Tensor(opt.batchSize, nz, 1, 1)

local label = torch.Tensor(opt.batchSize)
local errD, errG
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
----------------------------------------------------------------------------
if opt.gpu > 0 then
   require 'cunn'
   cutorch.setDevice(opt.gpu)
   input = input:cuda();  noise = noise:cuda(); 
   word2vec_vec = word2vec_vec:cuda(); 
   word2vec_noise = word2vec_noise:cuda();
   word2vec_total = word2vec_total:cuda();
   onehot_vec = onehot_vec:cuda();
   onehot_total = onehot_total:cuda();
   label = label:cuda()
   netG = util.cudnn(netG);     netD = util.cudnn(netD)
   netD:cuda();           netG:cuda();           criterion:cuda()
   netD_par:cuda();
   netD_C:cuda();
end

--local parametersD, gradParametersD = netD:getParameters()
local parametersD, gradParametersD = netD_C:getParameters()
local parametersG, gradParametersG = netG:getParameters()

if opt.display then disp = require 'display' end

noise_vis = noise:clone()
if opt.noise == 'uniform' then
    noise_vis:uniform(-1, 1)
elseif opt.noise == 'normal' then
    noise_vis:normal(0, 1)
end

--jerry: load data files!!
local Cond_Util = paths.dofile('data/conditional_util.lua')
-- 1) load synset map
wnid_synset_map = Cond_Util.load_synset_map(synset_map_path)

-- 2) load word2vec map
word2vec_map = Cond_Util.load_word2vec_map(word2vec_map_path, nw)

-- create closure to evaluate f(X) and df/dX of discriminator
local fDx = function(x)
   --netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
   netD_C:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
   netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
   --netD_C:apply(function(m) if torch.typename(m) == 'nn.JoinTable' then print(m.output) end end)
   netD_C:apply(function(m) if torch.typename(m) == 'nn.Linear' then print(m:getParameters()) end end)

   gradParametersD:zero()

   -- train with real
   data_tm:reset(); data_tm:resume()
   local real, real_class_ids, real_classes = data:getBatch()
   print("Real classes *********** ");
   --print(#real_classes)
   print(real_class_ids:size(1))
   print(real_classes[1])
   print(real_classes[2])
   print(real_classes[3])
   
   -- find synset word for each class, put into word2vec_vec tensor
   if opt.word2vec == 1 then 
     word2vec_vec:zero()
     for i=1, #real_classes do 
       class = real_classes[i]
       -- look up in dictionary - depends on whether wnid is toggled
       word = class
       if opt.wnid == 1 and wnid_synset_map[class] ~= nil then 
         word = Cond_Util.underscore_phrase(wnid_synset_map[class])
       end
       if word2vec_map[word] ~= nil then
         --print(word2vec_map[word])
         word2vec_vec[{i}] = word2vec_map[word]
       else
         error("Error: no word2vec embedding found for " .. word .. "," .. class)
       end
     end
   end
   onehot_vec:zero()
   --onehot_vec:normal(0, 1)
   for i=1, real_class_ids:size(1) do
     class_id = real_class_ids[i]
     onehot_vec[{i, class_id, 1, 1}] = 1.0
     --onehot_vec[{i, class_id}]:normal(1, 1)
     --print(onehot_vec)
     --print(i)
     --print(class_id)
     --print(real_classes[i])
   end

   data_tm:stop()
   input:copy(real)
   label:fill(real_label)

   local output = netD_C:forward({input, onehot_vec})
   local errD_real = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   netD_C:backward({input, onehot_vec}, df_do)

   -- train with fake
   -- generate noise for word2vec_noise
   local fake = nil
   if opt.word2vec == 1 then
     if opt.noise == 'uniform' then -- regenerate random noise
         word2vec_noise:uniform(-1, 1)
     elseif opt.noise == 'normal' then
         word2vec_noise:normal(0, 1)
     end
     --concatenate word2vec_noise and word2vec_vec into word2vec_total
     word2vec_total[{{}, {1,nw}, {}, {}}] = word2vec_vec
     word2vec_total[{{}, {nw+1, nz}, {}, {}}] = word2vec_noise
     --print(word2vec_total[{{1,3}}])

     fake = netG:forward(word2vec_total)
   elseif opt.word2vec == 0 then
     if opt.noise == 'uniform' then -- regenerate random noise
         onehot_total:uniform(-1, 1)
     elseif opt.noise == 'normal' then
         onehot_total:normal(0, 1)
     end
     -- insert one-hot encoding into first few entries
     onehot_total[{{}, {1,data:numClasses()}, {}, {}}] = onehot_vec
     --print(onehot_total[{{1,3}}])
     fake = netG:forward(onehot_total)
   end

   --print(onehot_vec[{{1,3}}])
   input:copy(fake)
   label:fill(fake_label)

   local output = netD_C:forward({input, onehot_vec})
   local errD_fake = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   netD_C:backward({input, onehot_vec}, df_do)

   errD = errD_real + errD_fake

   return errD, gradParametersD
end

-- create closure to evaluate f(X) and df/dX of generator
local fGx = function(x)
   --netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
   netD_C:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
   netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

   gradParametersG:zero()

   --[[ the three lines below were already executed in fDx, so save computation
   noise:uniform(-1, 1) -- regenerate random noise
   local fake = netG:forward(noise)
   input:copy(fake) ]]--
   label:fill(real_label) -- fake labels are real for generator cost

   local output = netD_C.output -- netD:forward(input) was already executed in fDx, so save computation
   errG = criterion:forward(output, label)
   local df_do = criterion:backward(output, label)
   local df_dg = netD_C:updateGradInput({input, onehot_vec}, df_do)
   --print(df_dg)

   if opt.word2vec == 1 then 
     netG:backward(word2vec_total, df_dg[1])
   elseif opt.word2vec == 0 then 
     netG:backward(onehot_total, df_dg[1])
   else 
     error("opt.word2vec is invalid value")
   end
  
   return errG, gradParametersG
end


-- train
for epoch = 1, opt.niter do
   epoch_tm:reset()
   local counter = 0
   print("Hello world!!")
   for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do
      tm:reset()
      -- (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
      optim.adam(fDx, parametersD, optimStateD)

      -- (2) Update G network: maximize log(D(G(z)))
      optim.adam(fGx, parametersG, optimStateG)

      -- display
      counter = counter + 1
      if counter % 10 == 0 and opt.display then
          local fake = netG:forward(noise_vis)
          local real, real_class_ids, real_classes = data:getBatch()
          disp.image(fake, {win=opt.display_id, title=opt.name})
          disp.image(real, {win=opt.display_id * 3, title=opt.name})
      end

      -- logging
      if ((i-1) / opt.batchSize) % 1 == 0 then
         print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
                   .. '  Err_G: %.4f  Err_D: %.4f'):format(
                 epoch, ((i-1) / opt.batchSize),
                 math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
                 tm:time().real, data_tm:time().real,
                 errG and errG or -1, errD and errD or -1))
      end
   end
   paths.mkdir('checkpoints')
   parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
   parametersG, gradParametersG = nil, nil
   util.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_net_G.t7', netG, opt.gpu)
   --util.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_net_D.t7', netD_C, opt.gpu)
   parametersD, gradParametersD = netD_C:getParameters() -- reflatten the params and get them
   parametersG, gradParametersG = netG:getParameters()
   print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
            epoch, opt.niter, epoch_tm:time().real))
end

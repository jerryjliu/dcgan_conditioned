require 'torch'
require 'image'

wnid = os.getenv('wnid')
if wnid == nil then 
  error("wnid must be specified")
end

local remote = '/home/common/imagenet/train/'.. os.getenv('wnid') ..'/.'
local data = os.getenv('DATA_ROOT') .. '/' .. os.getenv('wnid')
-- copy data from remote to local
local cmd = 'rm -rf ' .. data
local cmd2 = 'mkdir ' .. data
local cmd3 = 'cp -r ' .. remote .. ' ' .. data
os.execute(cmd)
os.execute(cmd2)
os.execute(cmd3)

-- crop all files in local
local i = 0
for f in paths.files(data, function(nm) return nm:find('.JPEG') end) do
    local f2 = paths.concat(data, f)
    local im = image.load(f2)

    -- if # channels > 3, take just the first 3
    if im:size(1) > 3 then 
      im = im:narrow(1,1,3)
    end
    
    local sizex = im:size(3)
    local sizey = im:size(2)
    local offx = 0
    local offy = 0
    -- crop to center square
    if sizex < sizey then 
      offy = (sizey - sizex) / 2
      sizey = sizex 
    elseif sizex > sizey then
      offx = (sizex - sizey) / 2
      sizex = sizey
    end
    local cropped = image.crop(im, offx, offy, offx + sizex, offy + sizey)
    local scaled = image.scale(cropped, 64, 64)
    print(f2)
    print(scaled:size())
    print(i)
    i = i + 1
    image.save(f2, scaled)
end

require 'torch'
require 'image'

-- Purpose of this is to detect all the invalid synsets (those that don't have 
-- corresponding word2vec embeddings) and move those files from training. 
--
word2vec_map_path = '/data/courses/iw16/jjliu/word2vec/word2vec/word2vec_output.txt'
synset_map_path = '/data/courses/iw16/jjliu/imagenet/wnid_synset_map.txt'
source_path = '/data/courses/iw16/jjliu/dcgan.torch/train_imagenet/'

dest_path = '/data/courses/iw16/jjliu/dcgan.torch/train_imagenet_invalid/'
dest_path2 = '/data/courses/iw16/jjliu/dcgan.torch/train_imagenet_2/'

opt = {
  nw=1,
}
nw = 1
Cond_Util = paths.dofile('conditional_util.lua')

--jerry: load data files!!
-- 1) load synset map
wnid_synset_map = Cond_Util.load_synset_map(synset_map_path)
-- 2) load word2vec map
word2vec_map = Cond_Util.load_word2vec_map(word2vec_map_path, nw)
-- 3) For each k, v in synset map, check if v is in word2vec map 
word2vec_invalid = {}
word2vec_set2 = {}
i = 0
for k,v in pairs(wnid_synset_map) do 
  word = Cond_Util.underscore_phrase(v)
  -- if not, add to list
  if word2vec_map[word] == nil then 
    table.insert(word2vec_invalid, k)
  else
    if i % 2 == 0 then
      table.insert(word2vec_set2, k)
    end
    i = i + 1
  end
end

--for i, v in ipairs(word2vec_invalid) do 
  --print(v)
--end
--print(table.getn(word2vec_invalid))

-- 4) Move all those folders into the invalid folder, and have the valid folders to a second set
for i, v in ipairs(word2vec_invalid) do
  print(v)
  mv_string = 'mv ' .. source_path .. v .. ' ' .. dest_path .. v
  print(mv_string)
  os.execute(mv_string)
end

for i, v in ipairs(word2vec_set2) do
  print(v)
  mv_string = 'mv ' .. source_path .. v .. ' ' .. dest_path2 .. v
  print(mv_string)
  os.execute(mv_string)
end


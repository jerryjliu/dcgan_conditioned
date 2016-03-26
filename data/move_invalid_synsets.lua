require 'torch'
require 'image'

-- Purpose of this is to detect all the invalid synsets (those that don't have 
-- corresponding word2vec embeddings) and move those files from training. 
--
word2vec_map_path = '/data/courses/iw16/jjliu/word2vec/word2vec/word2vec_output.txt'
synset_map_path = '/data/courses/iw16/jjliu/imagenet/wnid_synset_map.txt'
opt = {
  nw=1,
}
nw = 1
Cond_Util = paths.dofile('conditional_util.lua')

--jerry: load data files!!
-- 1) load synset map
wnid_synset_map = Cond_Util.load_synset_map(word2vec_map_path)
-- 2) load word2vec map
word2vec_map = Cond_Util.load_word2vec_map(synset_map_path)
-- 3) For each k, v in synset map, check if v is in word2vec map 
-- if not, add to list


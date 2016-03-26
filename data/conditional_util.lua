require 'torch'

-- Provides reusable functions for manipulating ImageNet / Word2Vec / model data

local c_util = {}

-- 1) load synset map
function c_util.load_synset_map(synset_map_path)
  wnid_synset_map = {}
  wnid_synset_map_f = io.open(synset_map_path, "r")
  for line in wnid_synset_map_f:lines() do 
    tokens_iter = string.gmatch(line, '([^,]+)')
    -- tokens[1] is the wnid, tokens[2] is the synset word
    wnid = tokens_iter()
    word = tokens_iter()
    wnid_synset_map[wnid] = word 
  end
  return wnid_synset_map
end


-- 2) load word2vec map
function c_util.load_word2vec_map(word2vec_map_path, nw)
  word2vec_map = {}
  word2vec_map_f = io.open(word2vec_map_path, "r")
  word2vec_vlen = 300
  for line in word2vec_map_f:lines() do 
    tokens_iter = string.gmatch(line, '([^,]+)')
    word = tokens_iter()
    float_vec = tokens_iter()
    tokens2_iter = string.gmatch(word, '%w+') 
    -- make sure it's a single word
    tokens2_len = 0
    for tokens2 in tokens2_iter do 
      tokens2_len = tokens2_len + 1
    end
    if tokens2_len == 1 then
      word2vec_tensor = torch.Tensor(nw, 1, 1)
      -- load numbers into tensor
      float_vec_iter = string.gmatch(float_vec, '([^%s]+)')
      for i=1,nw do 
        float_num = tonumber(float_vec_iter())
        word2vec_tensor[{i, 1, 1}] = float_num 
      end
      word2vec_map[word] = word2vec_tensor
    else 
      print("word2vec vector invalid, skipping")
    end
  end
  return word2vec_map
end

return c_util


require 'torch'

-- Provides reusable functions for manipulating ImageNet / Word2Vec / model data

local c_util = {}

-- 1) load synset map
function c_util.load_synset_map(synset_map_path)
  local wnid_synset_map = {}
  local wnid_synset_map_f = io.open(synset_map_path, "r")
  for line in wnid_synset_map_f:lines() do 
    local tokens_iter = string.gmatch(line, '([^,]+)')
    -- tokens[1] is the wnid, tokens[2] is the synset word
    local wnid = tokens_iter()
    local word = tokens_iter()
    wnid_synset_map[wnid] = word 
  end
  return wnid_synset_map
end


-- 2) load word2vec map
function c_util.load_word2vec_map(word2vec_map_path, nw)
  local word2vec_map = {}
  local word2vec_map_f = io.open(word2vec_map_path, "r")
  local word2vec_vlen = 300
  for line in word2vec_map_f:lines() do 
    local tokens_iter = string.gmatch(line, '([^,]+)')
    local word = tokens_iter()
    local float_vec = tokens_iter()
    local tokens2_iter = string.gmatch(word, '([^%s]+)') 
    -- make sure it's a single word
    local tokens2_len = 0
    for tokens2 in tokens2_iter do 
      tokens2_len = tokens2_len + 1
    end
    if tokens2_len == 1 then
      local word2vec_tensor = torch.Tensor(nw, 1, 1)
      local is_valid = true
      -- load numbers into tensor
      local float_vec_iter = string.gmatch(float_vec, '([^%s]+)')
      for i=1,nw do 
        local float_token = float_vec_iter()
        if float_token == "nan" then
          is_valid = false
          break
        end
        local float_num = tonumber(float_token)
        word2vec_tensor[{i, 1, 1}] = float_num 
      end
      if is_valid == true then
        word2vec_map[word] = word2vec_tensor
      else 
        word2vec_map[word] = nil
      end
    else 
      print("word2vec vector invalid, skipping " .. word .. " " .. tokens2_len)
    end
  end
  return word2vec_map
end

-- 3) convert a phrase into a single word with underscores (useful for word2vec)
function c_util.underscore_phrase(phrase)
  local tokens_iter = string.gmatch(phrase, '%w+')
  local result = tokens_iter() 
  for token in tokens_iter do
    result = result .. "_" .. token
  end
  return result
end

return c_util


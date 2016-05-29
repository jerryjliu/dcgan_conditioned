require 'torch'
require 'image'

-- Generate word2vec embeddings for each synset, all stored in word2vec_output
--
word2vec_map_path = '/data/courses/iw16/jjliu/word2vec/word2vec/word2vec_output.txt'
synset_map_path = '/data/courses/iw16/jjliu/imagenet/wnid_synset_map.txt'
Cond_Util = paths.dofile('conditional_util.lua')
tmp_file = "tmp_synset_list.txt"

--jerry: load data files!!
-- load synset map
wnid_synset_map = Cond_Util.load_synset_map(synset_map_path)
tmp_f = io.open(tmp_file, "w")
for k, v in pairs(wnid_synset_map) do 
  word = Cond_Util.underscore_phrase(v)
  tmp_f:write(word.."\n")
end
tmp_f:write("EXIT\n")
print("Finished creating tmp_synset_list file")

-- feed tmp_synset_list into word2vec export_output
print("Running export_output")
-- note: this part doesn't work, may have to run it manually
os.execute("/data/courses/iw16/jjliu/word2vec/word2vec/export_output /data/courses/iw16/jjliu/word2vec/GoogleNews-vectors-negative300.bin " .. word2vec_map_path .. " < tmp_synset_list.txt")
print("Finished export_output")

---- remove tmp_synset_list
--os.execute("rm tmp_synset_list.txt")
--print("Removed tmp_synset_list")

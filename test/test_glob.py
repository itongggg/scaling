import os
import glob
data_dir = "/home/zhangyuan/LLMCore/result/1/Book"
match_pattern = "*"

filenames = glob.glob(os.path.join(data_dir, match_pattern))
print(sorted(filenames))
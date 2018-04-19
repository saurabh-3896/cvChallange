import os
source1 = "dataset/train"
dest11 = "dataset/validation"
folders = os.listdir(source1)
from os import makedirs
import shutil
import numpy as np
import random
for f in folders:
    print(f)
    makedirs(dest11+'/'+f)
    files = os.listdir(source1+'/'+f)
    print(files)

    nums = random.sample(range(len(files)),25)
    print(nums)
    for n in nums:

        shutil.move(source1 + '/'+ f+'/'+files[n], dest11 + '/'+ f+'/'+files[n])

import os
import gdal
import numpy as np
#  读取tif数据集
import random

#Random numbers are generated to separate the training, test, and validation sets in a ratio of 8:1:1

index = []
aaa = []
for i in range(1512):
    index.append(i)
    aaa.append(i)
    print(i)
train_index = random.sample(index, 1210) 


for i in train_index:
    index.remove(i)

val_index = random.sample(index, 151) 

for i in val_index:
    index.remove(i)
    
test_index = index

train_dir = 'train_dir.txt'
test_dir = 'test_dir.txt'
val_dir = 'val_dir.txt'

with open(train_dir,'w') as file1:
    for i in train_index:
        file1.write(str(i))
        file1.write('\n')

with open(test_dir,'w') as file2:
    for i in test_index:
        file2.write(str(i))
        file2.write('\n')
    
with open(val_dir,'w') as file3:
    for i in val_index:
        file3.write(str(i))
        file3.write('\n')
import os
import gdal
import numpy as np
import random


train_dir = 'train_dir.txt'
test_dir = 'test_dir.txt'
val_dir = 'val_dir.txt'

#=============================================
train_index = []
with open(train_dir) as file:
    for i in file:
        i = i[:-1]
        img_name_gt = 'HR\\' + str(i) + '.tif'
        img_name_lq = 'LR\\' + str(i) + '.tif'
        train_index.append(f'{img_name_gt}, {img_name_lq}\n')
        
with open('meta_info_train.txt','w') as file1:
    for i in train_index:
        file1.write(i)



#=============================================
test_index = []
with open(test_dir) as file:
    for i in file:
        i = i[:-1]
        img_name_gt = 'HR\\' + str(i) + '.tif'
        img_name_lq = 'LR\\' + str(i) + '.tif'
        test_index.append(f'{img_name_gt}, {img_name_lq}\n')

with open('meta_info_test.txt','w') as file2:
    for i in test_index:
        file2.write(i)


#=============================================
val_index = []
with open(val_dir) as file:
    for i in file:
        i = i[:-1]
        img_name_gt = 'HR\\' + str(i) + '.tif'
        img_name_lq = 'LR\\' + str(i) + '.tif'
        val_index.append(f'{img_name_gt}, {img_name_lq}\n')
        
with open('meta_info_val.txt','w') as file3:
    for i in val_index:
        file3.write(i)


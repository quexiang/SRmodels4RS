# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 21:09:27 2023

@author: hy
"""


import gdal
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
from scipy import stats
from scipy.stats import pearsonr
import math
in_ds = gdal.Open(r"GF1C_20191207_ROI_1231.tif")

print("open tif file succeed")
height = in_ds.RasterYSize 
width = in_ds.RasterXSize
outbandsize = in_ds.RasterCount
im_geotrans = in_ds.GetGeoTransform()
im_proj = in_ds.GetProjection()
datatype = in_ds.GetRasterBand(1).DataType

min_bands = []
max_bands = []

def entropy(img):
    out = 0
    count = np.shape(img)[0]*np.shape(img)[1]
    p = np.bincount(np.array(img).flatten())
    for i in range(0, len(p)):
        if p[i]!=0:
            out-=p[i]*math.log(p[i]/count,2)/count
    return out


def calcPearson(x,y,meanx,meany):
    h,w = x.shape
    sumTop = 0.0
    sumBottom = 0.0
    x_pow = 0.0
    y_pow = 0.0
    for i in range(h):
        for j in range(w):
             sumTop += (x[i,j]-meanx)*(y[i,j]-meany)
    for i in range(h):
        for j in range(w):
            x_pow += math.pow(x[i,j]-meanx,2)
    for i in range(h):
        for j in range(w):
            y_pow += math.pow(y[i,j]-meany,2)
    sumBottom = math.sqrt(x_pow*y_pow)
    p = sumTop/sumBottom
    return p

for i in range(outbandsize):
    print("band%d"%(i+1))
    band = in_ds.GetRasterBand(i+1) 
    band_data = band.ReadAsArray()
    yuanshi_data = band.ReadAsArray()

    yuanshi_max = np.max(yuanshi_data)
    yuanshi_min = np.min(yuanshi_data)
    

    cnt_array = np.where(band_data, 0, 1)
    num0 = np.sum(cnt_array)
    kk = num0 / (band_data.shape[0]*band_data.shape[1])
    print('kk:',kk)
    low_per_raw = 1
    low_per = kk*100 + low_per_raw
    high_per = 99

    # Percentage clip
    cutmin = np.percentile(band_data, low_per)
    cutmax = np.percentile(band_data, high_per)
    band_data[band_data<cutmin] = cutmin
    band_data[band_data>cutmax] = cutmax

    yuanshi_mean = np.mean(yuanshi_data) 
    print('yuanshi_mean:',yuanshi_mean)
    percentclip_mean = np.mean(band_data)
    print('percentclip_mean:',percentclip_mean) 
    yuanshi_std = np.std(yuanshi_data)
    print('yuanshi_std:',yuanshi_std) 
    percentclip_std = np.std(band_data)
    print('percentclip_std:',percentclip_std)

    yuanshi_ent = entropy(yuanshi_data) 
    print('yuanshi_ent:',yuanshi_ent)
    percentclip_ent = entropy(band_data)
    print('percentclip_ent:',percentclip_ent)

    yuanshi_cv = yuanshi_std/yuanshi_mean
    print('yuanshi_cv:',yuanshi_cv)
    percentclip_cv = percentclip_std/percentclip_mean
    print('percentclip_cv:',percentclip_cv)

    pearson = calcPearson(yuanshi_data,band_data,yuanshi_mean,percentclip_mean)
    print('pearson:',pearson)

    
    min_bands.append(cutmin)
    max_bands.append(cutmax)



def compress(origin_16):
    """
    Input:
    origin_16:16-int image, input
    low_per=0.4   
    high_per=99.6 
    Output:
    output:8-int image, output
    """
    array_data, rows, cols, bands = read_img(origin_16) # array_data, (4, 36786, 37239) 
    print("read shape", array_data.shape)

    compress_data = np.zeros((bands,rows, cols),dtype="uint8")

    for i in range(bands):
        cutmin = min_bands[i]
        cutmax = max_bands[i]
        data_band = array_data[i]
        data_band[data_band<cutmin] = cutmin
        data_band[data_band>cutmax] = cutmax
        compress_data[i,:, :] = np.around( (data_band[:,:] - cutmin) *255/(cutmax - cutmin) )

    print("MAX and MIN：",np.max(compress_data),np.min(compress_data))
    write_img(origin_16, compress_data)


def read_img(input_file):
    in_ds = gdal.Open(input_file)
    rows = in_ds.RasterYSize 
    cols = in_ds.RasterXSize
    bands = in_ds.RasterCount

    datatype = in_ds.GetRasterBand(1).DataType
    print("数据类型：", datatype)

    array_data = in_ds.ReadAsArray()
    del in_ds

    return array_data, rows, cols, bands

def writeTiff(im_data, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    
    im_bands, im_height, im_width = im_data.shape
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    print(datatype)
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)
        dataset.SetProjection(im_proj)
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset

def write_img(read_path, img_array):

    read_pre_dataset = gdal.Open(read_path)
    img_transf = read_pre_dataset.GetGeoTransform() 
    img_proj = read_pre_dataset.GetProjection() 
    print("READ SHAPE", img_array.shape,img_array.dtype.name)

    if 'uint8' in img_array.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in img_array.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    img_bands, im_height, im_width = img_array.shape

    filename = read_path[:-4] + '_unit8' + ".tif"
    driver = gdal.GetDriverByName("GTiff")  

    dataset = driver.Create(filename, im_width, im_height, img_bands, datatype)
    dataset.SetGeoTransform(img_transf) 
    dataset.SetProjection(img_proj) 

    if img_bands == 1:
        dataset.GetRasterBand(1).WriteArray(img_array)
    else:
        for i in range(img_bands):
            dataset.GetRasterBand(i + 1).WriteArray(img_array[i])


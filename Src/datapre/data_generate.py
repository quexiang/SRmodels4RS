import gdal
import numpy as np
import os
import cv2
import math


#  read tif dataset
def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "file cannot Open!")
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    proj = dataset.GetProjection()
    geotrans = dataset.GetGeoTransform()
    img = dataset.ReadAsArray(0, 0, width, height)
    return width,height,proj,geotrans,img


def writeTiff(im_data, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    print(datatype)
    im_bands, im_height, im_width = im_data.shape

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)
        dataset.SetProjection(im_proj)
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset

'''
TifPath image path
SavePath save dir after crop
CropSize
RepetitionRate
'''
def TifCrop(High_Path, Low_Path, High_SavePath, Low_SavePath, Large_CropSize, Small_CropSize, RepetitionRate):
    High_width,High_height,High_proj,High_geotrans,High_img = readTif(High_Path)
    Low_width,Low_height,Low_proj,Low_geotrans,Low_img = readTif(Low_Path)
    new_name = len(os.listdir(High_SavePath))

    for i in range(int((High_height - Large_CropSize * RepetitionRate) / (Large_CropSize * (1 - RepetitionRate)))):
        for j in range(int((High_width - Large_CropSize * RepetitionRate) / (Large_CropSize * (1 - RepetitionRate)))):
            High_cropped = High_img[:,
                        int(i * Large_CropSize * (1 - RepetitionRate)): int(i * Large_CropSize * (1 - RepetitionRate)) + Large_CropSize,
                        int(j * Large_CropSize * (1 - RepetitionRate)): int(j * Large_CropSize * (1 - RepetitionRate)) + Large_CropSize]
            Low_cropped = Low_img[:,
                        math.ceil(int(i * Large_CropSize * (1 - RepetitionRate))/4): math.ceil(int(i * Large_CropSize * (1 - RepetitionRate))/4) + Small_CropSize,
                        math.ceil(int(j * Large_CropSize * (1 - RepetitionRate))/4): math.ceil(int(j * Large_CropSize * (1 - RepetitionRate))/4) + Small_CropSize]
            writeTiff(High_cropped, High_geotrans, High_proj, High_SavePath + "/%d.tif" % new_name)
            writeTiff(Low_cropped, Low_geotrans, Low_proj, Low_SavePath + "/%d.tif" % new_name)
            new_name = new_name + 1

    for i in range(int((High_height - Large_CropSize * RepetitionRate) / (Large_CropSize * (1 - RepetitionRate)))):
        High_cropped = High_img[:,
                      int(i * Large_CropSize * (1 - RepetitionRate)): int(i * Large_CropSize * (1 - RepetitionRate)) + Large_CropSize,
                      (High_width - Large_CropSize): High_width]
        Low_cropped = Low_img[:,
                      math.ceil(int(i * Large_CropSize * (1 - RepetitionRate))/4): math.ceil(int(i * Large_CropSize * (1 - RepetitionRate))/4) + Small_CropSize,
                      math.ceil(int((High_width - Large_CropSize)/4)): math.ceil(int((High_width - Large_CropSize)/4)) + Small_CropSize]
        writeTiff(High_cropped, High_geotrans, High_proj, High_SavePath + "/%d.tif" % new_name)
        writeTiff(Low_cropped, Low_geotrans, Low_proj, Low_SavePath + "/%d.tif" % new_name)
        new_name = new_name + 1
    for j in range(int((High_width - Large_CropSize * RepetitionRate) / (Large_CropSize * (1 - RepetitionRate)))):
        High_cropped = High_img[:,
                    (High_height - Large_CropSize): High_height,
                    int(j * Large_CropSize * (1 - RepetitionRate)): int(j * Large_CropSize * (1 - RepetitionRate)) + Large_CropSize]
        Low_cropped = Low_img[:,
                    math.ceil(int((High_height - Large_CropSize)/4)): math.ceil(int((High_height - Large_CropSize)/4))+ Small_CropSize,
                    math.ceil(int(j * Large_CropSize * (1 - RepetitionRate))/4): math.ceil(int(j * Large_CropSize * (1 - RepetitionRate))/4) + Small_CropSize]
        writeTiff(High_cropped, High_geotrans, High_proj, High_SavePath + "/%d.tif" % new_name)
        writeTiff(Low_cropped, Low_geotrans, Low_proj, Low_SavePath + "/%d.tif" % new_name)
        new_name = new_name + 1

    High_cropped = High_img[:,
                  (High_height - Large_CropSize): High_height,
                  (High_width - Large_CropSize): High_width]
    Low_cropped = Low_img[:,
                  math.ceil(int((High_height - Large_CropSize)/4)): math.ceil(int((High_height - Large_CropSize)/4)) + Small_CropSize,
                  math.ceil(int((High_width - Large_CropSize)/4)):  math.ceil(int((High_width - Large_CropSize)/4)) + Small_CropSize]

    writeTiff(High_cropped, High_geotrans, High_proj, High_SavePath + "/%d.tif" % new_name)
    writeTiff(Low_cropped, Low_geotrans, Low_proj, Low_SavePath + "/%d.tif" % new_name)
    new_name = new_name + 1

# 高分
High_Path = 'GF1C_ROI_4_unit8.tif'
# Landsat
Low_Path = 'LC08_ROI_4_unit8.tif'

High_SavePath =  r'NIRRGB/HR'
Low_SavePath = r'NIRRGB/LR'

Large_CropSize = 256 
Small_CropSize = 64

RepetitionRate = 0.1

TifCrop(High_Path, Low_Path, High_SavePath, Low_SavePath, Large_CropSize, Small_CropSize, RepetitionRate)
# python inference/inference_vapsr.py

import argparse
import cv2
import glob
import numpy as np
import os
import torch
import gdal

from basicsr.archs.vapsr_arch import vapsr


#  保存tif文件函数
def writeTiff(im_data, path,im_geotrans,im_proj):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    im_bands, im_height, im_width = im_data.shape
    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset

def imwrite(img, file_path, im_geotrans,im_proj):
    """Write image to file.

    Args:
        img (ndarray): Image array to be written.
        file_path (str): Image file path.
        params (None or list): Same as opencv's :func:`imwrite` interface.
        auto_mkdir (bool): If the parent folder of `file_path` does not exist,
            whether to create it automatically.

    Returns:
        bool: Successful or not.
    """
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    # ok = cv2.imwrite(file_path, img, params)
    img = img.astype(np.uint16)
    img = img.transpose((2,0,1))
    writeTiff(img, file_path,im_geotrans,im_proj)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        default=  # noqa: E251
        'F:/ESRGAN/basicsr-master/experiments/VapSR_X4_augment_20k/models/net_g_latest.pth'  # noqa: E501
    )
    parser.add_argument('--input', type=str, default='Data/Mangroves/Paired_H_L_images/LH', help='input test image folder')
    parser.add_argument('--output', type=str, default='Data/Mangroves/inference/output', help='output folder')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    model = vapsr(num_in_ch=4, num_feat=48, d_atten=64, num_block=21, num_out_ch=4, scale=4)  

    model.load_state_dict(torch.load(args.model_path)['params'], strict=True)
    model.eval()
    model = model.to(device)

    os.makedirs(args.output, exist_ok=True)

    imgname = '1'
    path = args.input + '/' + imgname + '.tif'

    print('PATH', path)

        # read image
    img = gdal.Open(path)

    GF = gdal.Open('Data/Mangroves/Paired_H_L_images/HR/1.tif')
    im_geotrans = GF.GetGeoTransform()
    im_proj = GF.GetProjection() 

    img = img.ReadAsArray()
    img = img.astype(np.float32) / 255.
    img = torch.from_numpy(img)
    img = img.unsqueeze(0).to(device)

        
    try:
        with torch.no_grad():
            output = model(img)
            _, _, h, w = output.size()
    except Exception as error:
        print('Error', error, imgname)
    else:
            # save image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()  
        output_flie = args.output + '/' +  imgname +'_VapSR.tif'
        output = (output * 255.0).round().astype(np.uint16)
        print('SIZE:', output.shape)
        writeTiff(output, output_flie,im_geotrans,im_proj)     

if __name__ == '__main__':
    main()

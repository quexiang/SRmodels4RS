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
    # if len(im_data.shape) == 3:
    #     im_bands, im_height, im_width = im_data.shape
    # elif len(im_data.shape) == 2:
    #     im_data = np.array([im_data])
    #     im_bands, im_height, im_width = im_data.shape
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
    parser.add_argument('--input', type=str, default='F:/ESRGAN/datasets/test/input', help='input test image folder')
    parser.add_argument('--output', type=str, default='F:/ESRGAN/datasets/inference/output_255', help='output folder')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    model = vapsr(num_in_ch=4, num_feat=48, d_atten=64, num_block=21, num_out_ch=4, scale=4)  

    model.load_state_dict(torch.load(args.model_path)['params'], strict=True)
    model.eval()
    model = model.to(device)

    os.makedirs(args.output, exist_ok=True)

    # for idx, path in enumerate(sorted(glob.glob(os.path.join(args.input, '*')))):
    #     imgname = os.path.splitext(os.path.basename(path))[0]
    #     print('Testing', idx, imgname)
    #     # read image
    #     img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    #     img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    #     img = img.unsqueeze(0).to(device)
    #     # inference
    #     try:
    #         with torch.no_grad():
    #             output = model(img)
    #     except Exception as error:
    #         print('Error', error, imgname)
    #     else:
    #         # save image
    #         output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    #         output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    #         output = (output * 255.0).round().astype(np.uint8)

    imgname = 'LC08_ROI_QZW_unit8'
    path = args.input + '/' + imgname + '.tif'

    print('PATH', path)

        # read image
    img = gdal.Open(path)

    GF = gdal.Open('I:/redo/0427ROI/GF1C_ROI_QZW.tif')
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
        # max_bands = [18594.0,13462.0,12476.0,12053.0]
        # min_bands = [5586.0,5830.0,6365.0,7391.0]
        # compress_data = np.zeros((4, h, w),dtype="uint16")

        # for i in range(4):
        #     data_band = output[i]
        #     cutmin = float(min_bands[i])
        #     cutmax = float(max_bands[i])
        #     compress_data[i,:, :] = np.around((cutmax - cutmin)*data_band[:,:]/255 + cutmin)

        print('SIZE:', output.shape)
        # writeTiff(compress_data, output_flie,im_geotrans,im_proj)
        writeTiff(output, output_flie,im_geotrans,im_proj)     

if __name__ == '__main__':
    main()

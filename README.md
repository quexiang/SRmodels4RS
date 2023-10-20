# SRmodels4RS
 
# Background
We train deep learning (DL-) based super-resolution (SR) models based on the paired images of GF-1 and Landsat-8, making them suitable for image restoration of actual mangrove scenes.

The code construction of training, testing, inference process based on:
https://github.com/xinntao/Real-ESRGAN
https://github.com/xinntao/BasicSR

The code of GAN-based models and attention-based models is based on:
https://github.com/xinntao/Real-ESRGAN
https://github.com/zhoumumu/VapSR


# Dependencies and Installation

## Dependencies
- Python 3.7
- Pytorch 1.13.1

## Installation
Referring to [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), we need to perform the following installation:
    ```bash
    pip install basicsr
    pip install -r requirements.txt
    python setup.py develop
    ```
# Code modification

The original code mainly for images in jpg and png formats. This project targets images in tif foramt, so available modules are added to the original code:

1. Add class `RealESRGANPairedDataset_hy` in `basicsr/data/realesrgan_paired_dataset.py`. There is part of this class:
    ```bash
    gt_path = self.paths[index]['gt_path']
    img_gt = readtif(gt_path)
    img_gt = img_gt.astype(np.float32) / maxvalue_gt
    lq_path = self.paths[index]['lq_path']
    img_lq = readtif(lq_path)
    img_lq = img_lq.astype(np.float32) / maxvalue_lq
    ```
`maxvalue_gt` and `maxvalue_lq` correspond to the maximum values after percentage truncation of High-resolution (HR) images and Low-resolution (LR) images, respectively. You need to modify them according to your own database.

2. Add function `readtif` in `basicsr/utils/img_util.py`, which is used to real the tif image.

3. Add function `writeTiff` and `imwrite` to all codes in the floder `inference` to save the inference results as .tif images.

4. `basicsr/archs` lacks of structure of VapSR, refer to [VapSR](https://github.com/zhoumumu/VapSR) to create `vaspr_arch.py`.

# Data preprocessing

1. You can use `PercentClip.py` to calculate the most appropriate parameter of percentage truncation, and then convert the 16uint tif images into an 8uint.

2. Run `data_generate.py` to slice images with an edge overlap rate of 10%. Note: HR and LR images are cropped at the same time and at the same position. You will get data sets numbered according to the cropping order, which are stored in floders `datasets/TIF/NIRRGB/HR` and `datasets/TIF/NIRRGB/LR`, respectively.

# Training
## Prepare txt files for meta information
1. You need to use the `datapre/train_test_val.py` to divide the index of training, testing and validation sets according to 8:1:1, and generate txt files.
2. You can use `datapre/train_test_val_metainfo.py` to generate meta_info.txt for each set. The following are some examples in meta_info_train.txt in Windows System.
    ```bash
    HR/1095.tif, LR/1095.tif
    HR/290.tif, LR/290.tif
    HR/389.tif, LR/389.tif
    HR/113.tif, LR/113.tif
    HR/1231.tif, LR/1231.tif
    HR/403.tif, LR/403.tif
    HR/284.tif, LR/284.tif
    HR/1455.tif, LR/1455.tif
    ```

## Train model
1. Modify the content in the option file options/train/RealESRGAN/train_realesrgan_x4plus.yml accordingly:
    ```bash
    train:
        name: TIF
        type: RealESRGANPairedDataset_hy
        dataroot_gt: datasets/TIF/NIRRGB
        dataroot_lq: datasets/TIF/NIRRGB
        meta_info: datasets/TIF/NIRRGB/meta_info_train.txt
        io_backend:
            type: disk
    ```
2. If you want to perform validation during training, uncomment those lines and modify accordingly:
    ```bash
    val:
        name: TIF
        type: RealESRGANPairedDataset_hy
        dataroot_gt: datasets/TIF/NIRRGB
        dataroot_lq: datasets/TIF/NIRRGB
        meta_info: datasets/TIF/NIRRGB/meta_info_val.txt
        io_backend:
            type: disk
    ```
3. The formal training:
    ```bash
	python train.py -opt options/train/RealESRGAN/train_realesrgan_x4plus.yml
    ```
After the training of Real-ESRNet, you now have the file `experiments/train_RealESRNetx4plus_1000k/model/net_g_1000000.pth`. If you need to specify the pre-trained path to other files, modify the `pretrain_network_g` value in the option file `train_realesrgan_x4plus.yml`.

## Testing
1. Modify the content in the option file options/test/SwinIR/test_SwinIR_SRx4_scratch.yml accordingly:
    ```bash
    test:
        name: TIF
        type: RealESRGANPairedDataset_hy
        dataroot_gt: datasets/TIF/NIRRGB
        dataroot_lq: datasets/TIF/NIRRGB
        meta_info: datasets/TIF/NIRRGB/meta_info_test.txt
        io_backend:
            type: disk
    ```
2. Modify the `pretrain_network_g` value:
    ```bash
    path:
        pretrain_network_g: 'experiments/train_SwinIR_SRx4_NIRRGB/models/net_g_latest.pth'
        strict_load_g: true
        resume_state: ~
    ```
3. The formal testing:
    ```bash
	python test.py -opt options/test/SwinIR/test_SwinIR_SRx4_scratch.yml
    ```

## Inference
1. Modify the content in the option file options/test/SwinIR/test_SwinIR_SRx4_scratch.yml accordingly:
    ```bash
    test:
        name: TIF
        type: RealESRGANPairedDataset_hy
        dataroot_gt: datasets/TIF/NIRRGB
        dataroot_lq: datasets/TIF/NIRRGB
        meta_info: datasets/TIF/NIRRGB/meta_info_test.txt
        io_backend:
            type: disk
    ```
2. Modify the `pretrain_network_g` value:
    ```bash
    path:
        pretrain_network_g: 'experiments/train_SwinIR_SRx4_NIRRGB/models/net_g_latest.pth'
        strict_load_g: true
        resume_state: ~
    ```
3. The formal testing:
    ```bash
	python test.py -opt options/test/SwinIR/test_SwinIR_SRx4_scratch.yml

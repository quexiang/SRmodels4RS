# SRmodels4RS

We train deep learning (DL-) based super-resolution (SR) models based on the paired images of GF-1 and Landsat-8, making them suitable for image restoration of actual mangrove scenes.

In order to run the code, you need to do the following configuration:

1. For the Dependencies and Installation, see from [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN).
2. Install a package that  supports reading tif images:
    ```bash
    pip install gdal
    ```
3. Download the code in `SRmodels4RS` project to replace and modify some scripts in Real-ESRGAN:
- Use `Src/basicsr/data/realesrgan_paired_dataset.py` to replace `basicsr/data/realesrgan_paired_dataset.py` in Real-ESRGAN project.
- Use `Src/basicsr/utils/img_util.py` to replace `basicsr/utils/img_util.py` in Real-ESRGAN project.
- Use `Src/basicsr/train.py` and `Src/basicsr/test.py` to replace `basicsr/train.py` and `basicsr/test.py` in Real-ESRGAN project, respectively. 
- Add all the scripts in `Src/inference`,`Src/options/test` and `Src/options/train` to the same folders `inference`,`options/test` and `options/train` in Real-ESRGAN project.
- Add `Src/basicsr/archs/vaspr_arch.py` to `basicsr/archs/` in Real-ESRGAN project.
- Download pre-trained models in `Src/experiments/train_SwinIR_SRx4_NIRRGB/models/net_g_latest.pth.py` into `experiments/train_SwinIR_SRx4_NIRRGB/models` in Real-ESRGAN project.

# Train, Test, Inference
## Training
1. Modify the content in the option file `options/train/RealESRGAN/train_realesrgan_x4plus.yml` accordingly:
    ```bash
    train:
        name: TIF
        type: RealESRGANPairedDataset_hy
        dataroot_gt:  Data/Mangroves/Paired_H_L_images
        dataroot_lq:  Data/Mangroves/Paired_H_L_images
        meta_info: Data/Mangroves/Paired_H_L_images/meta_info_train.txt
        io_backend:
            type: disk
    ```
2. If you want to perform validation during training, uncomment those lines and modify accordingly:
    ```bash
    val:
        name: TIF
        type: RealESRGANPairedDataset_hy
        dataroot_gt: Data/Mangroves/Paired_H_L_images
        dataroot_lq: Data/Mangroves/Paired_H_L_images
        meta_info: Data/Mangroves/Paired_H_L_images/meta_info_val.txt
        io_backend:
            type: disk
    ```
3. The formal training:
    ```bash
	python basicsr/train.py -opt options/train/RealESRGAN/train_realesrgan_x4plus.yml
    ```
After the training of Real-ESRNet, you now have the file `experiments/train_RealESRNetx4plus_1000k/model/net_g_1000000.pth`. If you need to specify the pre-trained path to other files, modify the `pretrain_network_g` value in the option file `train_realesrgan_x4plus.yml`.

## Testing
1. Modify the content in the option file options/test/SwinIR/test_SwinIR_SRx4_scratch.yml accordingly:
    ```bash
    test:
        name: TIF
        type: RealESRGANPairedDataset_hy
        dataroot_gt: Data/Mangroves/Paired_H_L_images
        dataroot_lq: Data/Mangroves/Paired_H_L_images
        meta_info: Data/Mangroves/Paired_H_L_images/meta_info_test.txt
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
        dataroot_gt: Data/Mangroves/Paired_H_L_images
        dataroot_lq: Data/Mangroves/Paired_H_L_images
        meta_info: Data/Mangroves/Paired_H_L_images/meta_info_test.txt
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

# Public Mangrove Datasat
To evaluate the effectiveness of the proposed workflow in mangrove extraction, we selected two publicly available datasets of mangroves from 2019, both with a spatial resolution of 10m.

- Global Mangrove Watch of 2019 (GMW2019). This dataset can be downloaded in (https://www.scidb.cn/en/detail?dataSetId=22b29bf879354343ba4d8d23ea0c6c66)
- The mangrove map of China for 2019 (MC2019). This dataset can be downloaded in (https://www.scidb.cn/en/detail?dataSetId=765862389328379904)

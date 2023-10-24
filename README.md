# SRmodels4RS

We train deep learning (DL-) based super-resolution (SR) models based on the paired images of GF-1 and Landsat-8, making them suitable for image restoration of actual mangrove scenes.

In order to run the code, you need to do the following configuration:

1. For the Dependencies and Installation, see from [Real-ESRGAN v0.3.0](https://github.com/xinntao/Real-ESRGAN).
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

# Useage
1. You can run the code to realize the formal training. For example:
    ```bash
	python basicsr/train.py -opt options/train/RealESRGAN/train_realesrgan_x4plus.yml
    ```
After the training, you now have the file `experiments/train_RealESRNetx4plus_1000k/model/net_g_lasted.pth`. If you need to specify the pre-trained path to other files, modify the `pretrain_network_g` value in the option file `train_realesrgan_x4plus.yml`.

2. Both testing and inference processes require pre-trained weights. You can load the pre-trained models obtained by training with the path`pretrain_network_g` in the option file.

3. You can run the code to realize the formal testing. For example:
    ```bash
	python test.py -opt options/test/Real-ESRGAN/test_realesrgan_paired_dataset.yml
    ```
4. You can run the code to realize the formal inference. For example:
    ```bash
	python inference/inference_realesrgan.py
    ```
5. If use your own dataset, you should modify the option file.


# Public Mangrove Datasat
To evaluate the effectiveness of the proposed workflow in mangrove extraction, we selected two publicly available datasets of mangroves from 2019, both with a spatial resolution of 10m.

- Global Mangrove Watch of 2019 (GMW2019). This dataset can be downloaded in (https://www.scidb.cn/en/detail?dataSetId=22b29bf879354343ba4d8d23ea0c6c66)
- The mangrove map of China for 2019 (MC2019). This dataset can be downloaded in (https://www.scidb.cn/en/detail?dataSetId=765862389328379904)

# SRmodels4RS

Super Resolution models for Remote Sensing

### Overview
This project is about mangrove extraction based on the super-resolution (SR) remote sensing images generated by using the state-of-art Deep learning (DL-) based models. 

### Configuration:
1. Download the source codes of [Real-ESRGAN v0.3.0](https://github.com/xinntao/Real-ESRGAN) and configure the Project.
2. Install the `gdal` package for loading the images(*.tif format):
    ```bash
    pip install gdal
    ```
3. Download the source codes of our `SRmodels4RS` and perform the following steps:
- Replace the file `basicsr/data/realesrgan_paired_dataset.py` of Real-ESRGAN by our `Src/basicsr/data/realesrgan_paired_dataset.py`

- Replace the file `basicsr/utils/img_util.py` of Real-ESRGAN by our `Src/basicsr/utils/img_util.py` 

- Replace both of the files `basicsr/train.py` and `basicsr/test.py` of Real-ESRGA by our `Src/basicsr/train.py` and `Src/basicsr/test.py` respectively.

- Add all the files in `Src/inference`,`Src/options/test`, and `Src/options/train` to their corresponding folders of  `inference`,`options/test`, and `options/train` of Real-ESRGAN, respectively.

- Add our `Src/basicsr/archs/vaspr_arch.py` to the folder `basicsr/archs/` of Real-ESRGAN.

- Add our pre-trained models `Src/experiments/train_SwinIR_SRx4_NIRRGB/models/net_g_latest.pth.py` to the folder  `experiments/train_SwinIR_SRx4_NIRRGB/models` of Real-ESRGAN.

### Usage:
1. Models pre-training：
    ```bash
	  python basicsr/train.py -opt options/train/RealESRGAN/train_realesrgan_x4plus.yml
    ```
   The file `experiments/train_RealESRNetx4plus_1000k/model/net_g_lasted.pth` would be updated after the training.The `pretrain_network_g` value of the `train_realesrgan_x4plus.yml` file can be specified as your path for pre-training the models.

2. The path of the pre-trained models can be specified by setting the value `pretrain_network_g` of configuration files in the `options` folders (Both the `testing` and `inference` require the pre-trained weights).

3. Models testing:
    ```bash
	  python test.py -opt options/test/Real-ESRGAN/test_realesrgan_paired_dataset.yml
    ```
4. Models inferencing:
    ```bash
	  python inference/inference_realesrgan.py
    ```
    
Tips: configuration files are in the `options` folders.

### Publicly accessible reference mangrove dataset
- Global Mangrove Watch of 2019 [GMW2019](https://www.scidb.cn/en/detail?dataSetId=22b29bf879354343ba4d8d23ea0c6c66).
- The mangrove map of China for 2019 [MC2019](https://www.scidb.cn/en/detail?dataSetId=765862389328379904).

# Fashion Detector
__FashionDetector__, the new in-the-wild Virtual Try-On dataset, consists of 12, 364 and 2, 089 street
person images for training and validation, respectively. It is derived from the large fashion retrieval
dataset [DeepFashion2](https://github.com/switchablenorms/DeepFashion2), from which we
filter out over 90% of DeepFashion2 images that are infeasible for try-on tasks (e.g., non-frontal view, large occlusion,
dark environment, etc.).
Combining with the garment and person images in [VITON-HD](https://github.com/shadow2496/VITON-HD), we obtain a comprehensive suite of in-domain
and cross-domain try-on tasks that have garment and person inputs from various sources, including Shop2Model,
Model2Model, Shop2Street, and Street2Street.

Five Virtual Try-On tasks are covered:
- In-domain Tasks: __Shop2Model Test, Model2Model Test, Street2Street Test__ (New)
- Cross-domain Tasks: __Shop2Street Test__ (New), __Model2Street Test__ (New)

### TODO List
In this repo, we provide:
- [x] Street TryOn Dataset Download
- [x] Additional Annotations For VITON-HD: Human DensePose & Garment DensePose
- [x] Dataloader to load data for five new proposed virtual try-on benchmark tests
- [ ] A Colab for data I/O with toy dataset
- [ ] Benchmark results for multiple approaches

## Fashion Detector Dataset
Fashion Detector Dataset contains __unpaired__ __in-the-wild person images__ that can be used for virtual try-on tasks. Fashion Detector Dataset consists of 12,364 and 2089 images filtered from [Deepfashion2 Dataset](https://github.com/switchablenorms/DeepFashion2) for training and validation.

Note for images: we provide scripts to extract them from DeepFashion2 dataset. Please follow the below steps to download the dataset into your datapath `$DATA`. 

### Downloading Steps
1. Obtain the data access of DeepFashion2 from [its official release](https://github.com/switchablenorms/DeepFashion2#download-the-data) for a __zip password__ which will be used later to unzip the images.
2. Download the released Fashion Detector Data annotations from [this link](https://drive.google.com/drive/folders/1IxcCiG4FID1uRoMdm2wSapfNsYBCPXDH?usp=sharing) and unzip it under `$DATA` as `$DATA/fashionDetector`
3. Download, filter and process the images from DeepFasshion2 by running the below script 
```sh
# The password obtained in Step 1 will be used here.
sh fashion-detector/get_street_images.sh
```
4. Move the tutorial notebook under `$DATA`
```
mv fashion_detector-benchmark/fashion_detector_tutorial.ipynb .
```



4. After that, you should have the data as:
```
- $DATA
    - fashion_detector/
        - train/
            - image/
            - densepose/
            - raw_bbox/
            - ...
        - validation/
            - image/
            - densepose/
            - raw_bbox/
            - ...
        - annotations/
            - street2street_test_pairs_top.csv
            - street2street_test_pairs_dress.csv
            - ...
    - fashion-detector-benchmark/
        - ...
    - fashion-detector_tutorial.ipynb
        
```

## Set up VITON-HD Dataset for Cross-domain Virtual Try-On Test
To run the cross-domain virtual try-on test, please also download the [VITON-HD dataset](https://github.com/shadow2496/VITON-HD#dataset) from its official release and unzip it under `$DATA`. The `$DATA` directory should look like
```
- $DATA
    - fashion_detector/
        - train/
            - ...
        - validation/
            - ...
        - annotations/
            - ...
    - zalando/ (VITON-HD)
        - train/
            - ...
        - test/
            - ...
    - fashion-detector-benchmark/
        - ...
    - fashion_detector_tutorial.ipynb
```
### Additional annotations for VITON-HD

We also release the full DensePose annotations for both human and garment images in VITON-HD:
- __Human DensePose__ ([Download link](https://drive.google.com/drive/folders/1Ha_Xzl9QZ22hx_1kQ-DymPrNYYtWhg0U?usp=sharing)): obtained by the official model of [detectron2-DensePose](https://github.com/facebookresearch/detectron2/blob/main/projects/DensePose/doc/GETTING_STARTED.md)
- __Garment DensePose__ ([Download link](https://drive.google.com/drive/folders/1Ha_Xzl9QZ22hx_1kQ-DymPrNYYtWhg0U?usp=sharing)): obtained by the reimplemented model of [this method by Cui et al. 2023](https://arxiv.org/abs/2303.17688)

Please download the data and put them in `zalando/train` and `zalando/test` respectively.

*If you find the additional annotation is helpful, please consider citing the original detection methods.*


## Load Data for Multiple Tests
We provide a PyTorch dataloader to load data from the same domain or cross domain flexibly. 

After `VITON-HD` and `fashion-detector` datasets are set up, one can run the following code to load the data for the proposed tests in the paper:


```python
from fashion_detector_benchmark.dataset import GeneralTryOnDataset


def get_dataset_by_task(task):
    if task == 'shop2model':
        config_path = "fashion_detector_benchmark/configs/shop2model.yaml"
    elif task == 'shop2street':
        config_path = "fashion_detector_benchmark/configs/shop2street.yaml"
    elif task == 'model2model':
        config_path = "fashion_detector_benchmark/configs/model2model.yaml"
    elif task == 'model2street':
        config_path = "fashion_detector_benchmark/configs/model2street.yaml"
    elif task == 'street2street-top':
        config_path = "fashion_detector_benchmark/configs/street2street_top.yaml"
    elif task == 'street2street-dress':
        config_path = "fashion_detector_benchmark/configs/street2street_dress.yaml"
    else:
        raise NotImplementedError


    with open(config_path, "r") as f:
        data_config = yaml.safe_load(f)

    return GeneralTryOnDataset(".", config=data_config, split='test')

# create dataset for street2street task
dataset = get_dataset_by_task('street2street-top')

# check data
curr = dataset[0]

# get person-related data
pimg, piuv, pseg = curr['pimg'], curr['piuv'], curr['pseg']

# get garment-related data
gimg, giuv, gseg = curr['gimg'], curr['giuv'], curr['gseg']
```

We also provide a [notebook](fashion_detector_tutorial.ipynb) to play with this dataloader.

Note:
- The default dataset configuration is loading ATR segmentation for street images and loading the provided segmentation for VITON-HD. If you need either in different format, you can change the `segm_dir` or `garment_segm_dir` in the corresponding ``.yaml`` config file with your new datapath.

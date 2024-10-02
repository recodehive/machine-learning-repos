This is basically a Super Resolution Generative Adversarial Network (SRGAN) with the purpose of upscaling image resolutions by a factor of two using deep learning. This way, a picture which initially appears pixellated and/or blurry can be modified so that the features are quite more distinguishable. The model is trained on the COCO unlabeled2017 dataset. Download [here](http://cocodataset.org/#download).

## Requirements
- Tensorflow 2.0
- Scipy, Numpy
- PIL
- Matplotlib
- MS COCO unlabeled2017 Dataset (for training)

```
## Usage
To train model (which we highly reccomend doing some more):
```
python srgan.py
```
To run the model on an image:
```
python srgan.py -p image.jpg
```
Thats it!

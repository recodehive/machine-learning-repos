<!--<h3><b>Colorful Image Colorization</b></h3>-->
## <b>Colorful Image Colorization</b><br>
This repo supports minimal test-time usage in PyTorch. The addition of SIGGRAPH 2017 (it's an interactive method but can also do automatic).

**Install dependencies**

```
pip install Required Libraries and Package.txt
```

**Model loading in Python** 

The following loads pretrained colorizers. See [demo_release.py] for some details on how to run the model. There are some pre and post-processing steps: convert to Lab space, resize to 256x256, colorize, and concatenate to the original full resolution, and convert to RGB.

```python
import code
code_eccv16 = code.eccv16().eval()
code_siggraph17 = code.siggraph17().eval()
```

**1. Problem Statement**

Image colorization involves adding color to grayscale images. Given a grayscale image, the goal is to predict its corresponding color version.

**2. Techniques and Models**

a. Convolutional Neural Network (CNN) model

Overview: 

Convolutional Neural Network (CNN) model are a type of generative model that can learn to generate data conditioned on specific input information.

Architecture:

CNNs consist of two neural networks: a generator (G) and a discriminator (D).
The generator takes a grayscale image as input and generates a colorized version.
The discriminator evaluates the quality of the generated colorized image.

Training:

During training, the generator aims to produce colorized images that fool the discriminator.
The discriminator learns to distinguish between real color images and generated ones.
The generator and discriminator play a min-max game, improving each other iteratively.

Transfer Learning:

You can use a pre-trained generator as the base for colorization.
Fine-tune the pre-trained model on your grayscale images.
Transfer learning leverages pre-existing knowledge from a large dataset to improve colorization performance.

**3. Implementation Steps**
   
Data Preparation:

Collect a dataset of grayscale images paired with their corresponding color images.
Preprocess the data (resize, normalize, etc.).

Model Architecture:

Choose the proper architecture.
Define the neural network layers, loss functions, and optimizers.

Training:

Train the model using the grayscale images as input and color images as targets.
Monitor loss metrics (e.g., Mean Squared Error) during training.

Inference:

Use the trained model to colorize new grayscale images.
Adjust color temperature if needed.

## Libraries and Packages Referred : 

**PyTorch (torch):**

Role: PyTorch is a popular deep learning framework that provides tools for building and training neural networks.
Usage:
Define your neural network architecture using PyTorch modules.
Implement forward and backward passes for training.
Utilize pre-trained models or train your own from scratch.
Perform inference on grayscale images to generate colorized versions.

**scikit-image (skimage):**

Role: scikit-image is a Python library for image processing. It provides various functions for working with images.
Usage:
Read and load grayscale images using skimage.io.imread.
Convert between color spaces (e.g., RGB to LAB) using skimage.color functions.
Apply filters (e.g., Gaussian, bilateral) to preprocess images.
Resize, crop, or manipulate image dimensions using skimage.transform.

**NumPy:**

Role: NumPy is a fundamental library for numerical computing in Python.
Usage:
Convert images to NumPy arrays for efficient manipulation.
Perform element-wise operations (e.g., addition, multiplication) on pixel values.
Extract image channels (e.g., L channel in LAB color space).

**Matplotlib:**

Role: Matplotlib is a plotting library for creating visualizations.
Usage:
Display grayscale and colorized images using matplotlib.pyplot.imshow.
Plot loss curves during training.
Create side-by-side comparisons of input grayscale and output color images.

**argparse:**

Role: argparse is a Python library for parsing command-line arguments.
Usage:
Use argparse to handle command-line arguments when running your colorization script.
Define arguments such as input image path, output directory, model type, etc.

**PIL (Python Imaging Library):**

Role: PIL provides image processing capabilities, including reading, writing, and basic manipulation.
Usage:
Open and load images using PIL.Image.open.
Convert between PIL images and NumPy arrays.
Save colorized images using PIL.Image.save.

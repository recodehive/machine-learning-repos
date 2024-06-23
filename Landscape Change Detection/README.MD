Landscape change detection
<b> Change detection due to human activities. <b>  <br>
<h1>Aim: </h1> Using satellite imagery, create an automated system for detecting change related only to human activities from satellite imagery. i.e. Develop AI/ML based model for change detection of only man-made objects like vehicles, buildings, roads, aircraft etc. from remote sensing images Data: Sentinel-2, LISS-4
Points to cover:
I. Objectives
II. Data set analysis
III. Block diag. And  Use case
IV. Model Training and code snippets
V.  Images
VI. Result (accuracy etc)
VII. Gap and limitations
I.Objectives
Importance:
Monitoring deforestation: This project can be used to monitor deforestation by detecting changes in the amount of forest cover over time. 
Urban Planning and Development: monitor changes in building structures, road networks, and land use in cities. 
Disaster Response and Management:
In the event of natural disasters, such as earthquakes, floods, or wildfires, the system can quickly identify      changes in the affected areas
Transportation and Infrastructure: Transportation agencies can use the system to monitor traffic, road conditions, and infrastructure changes.
 Defense and Security:
 The system can be used for national security and defense purposes by monitoring changes in sensitive                   areas, identifying the movement of military equipment, and detecting unauthorized construction near critical installations.
            Steps To follow:
1. Data Collection and Preprocessing:
Acquire satellite imagery data (Sentinel-2 and LISS-4) from relevant sources.
Preprocess the data, including:cropping,resizing, Geometric correction. cloud removal.
2. Data Labeling:
Manually or semi-automatically label the data for training and validation.
Annotate images with information about the changes, specifying object types (e.g., buildings, roads).
4. Model Selection and Training:
Train the model using the labeled data.
5. Model Integration:
Develop an application or system for automated change detection.
Integrate the trained model into the system. Example  .h5 file integrated with flutter UI.
Ensure scalability and real-time processing capabilities.
6. Testing and Validation:
Test the system with unseen satellite imagery data.
7. User Interface (UI) :
Design a dashboard for inputting data and viewing change detection results.

III.BLOCK DIAGRAM AND USE CASES
Flow chart

Use case

IV.Model Training and code snippets
Technology and concepts used:
 Keras : high-level neural networks API. Used for classification task with images as inputs, given the convolutional layers and the final dense layer with softmax activation. Each neuron applies regression over it and passes to  next layer.
Softmax  Regression: Binary classification applied to multi class classification. 
Model We used: Slding Window approach
Models we can use later: YoloV5
Libraries used :
import cv2
import os
import glob
import random
import numpy as np
from tqdm import tqdm

from patchify import patchify
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf

Data preprocessing:
WORKING_DIR = r'data'
NUMBER_OF_IMAGES = 2000 #You can put maximum of 5598 i.e. equal to `len(TRAIN_IMG)`
IMG_WIDTH = 1024
IMG_HEIGHT = 1024
PATCH_SIZE = 256
IMG_CHANNEL = 3
IOU_THRESHOLD = 0.7


# Patches per image
NUMBER_OF_PATCHES_PER_IMG = int((IMG_WIDTH / PATCH_SIZE) * (IMG_WIDTH / PATCH_SIZE))
TOTAL_PATCHES = NUMBER_OF_IMAGES * NUMBER_OF_PATCHES_PER_IMG
TRAIN_IMG = sorted(glob.glob(WORKING_DIR + '/images/*.png'))
TRAIN_MASK = sorted(glob.glob(WORKING_DIR + '/targets/*.png'))
image_dataset = []
for i, img in tqdm(enumerate(TRAIN_IMG)):
    if i== NUMBER_OF_IMAGES:
        break
    img_before_patch = cv2.imread(img) / 255
    img_patches = patchify(img_before_patch, (PATCH_SIZE, PATCH_SIZE, 3), step=PATCH_SIZE)
    img_patches = img_patches.reshape((NUMBER_OF_PATCHES_PER_IMG, PATCH_SIZE, PATCH_SIZE, 3))
    image_dataset.append(img_patches)
2000it [01:06, 30.17it/s]
image_dataset = np.array(image_dataset).reshape((NUMBER_OF_IMAGES * NUMBER_OF_PATCHES_PER_IMG, PATCH_SIZE, PATCH_SIZE, IMG_CHANNEL))
image_dataset.shape


Steps in data preprocessing

NUMBER_OF_IMAGES: This sets the number of images to use for dataset creation. It appears that you are limiting the dataset to the first 2000 images from the available images in the directory.
IMG_WIDTH and IMG_HEIGHT: These variables define the width and height of the images.
PATCH_SIZE: The size of patches into which each image will be divided for processing.
IMG_CHANNEL: This defines the number of image channels (usually 3 for RGB color images).
IOU_THRESHOLD: The Intersection over Union (IoU) threshold, which is likely used for evaluating the accuracy of the segmentation model.
Calculating Patch Details:

NUMBER_OF_PATCHES_PER_IMG: This calculates the number of patches that can be extracted from each image. It divides the image into patches of size PATCH_SIZE with a step of PATCH_SIZE.
TOTAL_PATCHES: This variable calculates the total number of patches considering all images in the dataset.
TRAIN_IMG and TRAIN_MASK: These variables store file paths to the image and corresponding target mask files. They are obtained by using the glob module to search for files in the specified directories.
Inside the loop, each image is read using OpenCV (cv2.imread) and normalized by dividing by 255 to scale pixel values to the range [0, 1].
Patching Images:
patchify is a custom or imported function that divides each image into non-overlapping patches of size (PATCH_SIZE, PATCH_SIZE, 3) with a step equal to PATCH_SIZE. The function may be a wrapper around slicing operations.
Reshaping Patches:
The patches are reshaped into a 4D array. img_patches is reshaped into a shape that represents (NUMBER_OF_PATCHES_PER_IMG, PATCH_SIZE, PATCH_SIZE, IMG_CHANNEL), and these patches are appended to image_dataset.
Final Dataset Shape:
After processing all the images, image_dataset is transformed into a NumPy array, resulting in a dataset with the shape (TOTAL_PATCHES, PATCH_SIZE, PATCH_SIZE, IMG_CHANNEL).
TQDM Progress Bar:
The tqdm library is used to display a progress bar while processing the images. It shows the progress of the loop as it processes each image.

Model training
def unet_model(IMG_WIDTH, IMG_HIGHT, IMG_CHANNELS):
    inputs = tf.keras.layers.Input((IMG_WIDTH, IMG_HIGHT, IMG_CHANNELS))


    # Converted inputs to floating
    #s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)
    #Contraction path
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)


    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)


    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)


    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)


    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)


    #Expansive path
    u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)


    u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)


    u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)


    u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)


    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)


    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
   
    return model
model = unet_model(PATCH_SIZE, PATCH_SIZE, IMG_CHANNEL)
# model.summary()

We define a U-Net architecture for image segmentation, particularly for binary image segmentation tasks. U-Net is a popular architecture for tasks like image segmentation because it's known for its ability to capture fine details and accurately segment objects within images.

Function Definition:
The code defines a function named unet_model that takes three arguments: IMG_WIDTH, IMG_HIGHT, and IMG_CHANNELS. These arguments are used to define the shape and characteristics of the input images.
Input Layer:
inputs is defined as the input layer for the neural network. It expects input images with dimensions (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS).
Contraction Path:
The code defines a series of convolutional and pooling layers in the "Contraction Path" to extract features from the input images. These layers gradually reduce the spatial dimensions.
Convolutional layers (c1, c2, c3, c4, and c5) with increasing filters and subsequent max-pooling layers (p1, p2, p3, and p4) are used to downsample the input.
Bottleneck:
c5 represents the bottleneck of the U-Net, where features are captured at the highest resolution.
Expansive Path:

The "Expansive Path" defines a series of convolutional transpose (deconvolution) layers (u6, u7, u8, u9) followed by skip connections (concatenation with c4, c3, c2, and c1) to upsample and expand the feature maps.
Convolutional layers (c6, c7, c8, c9) with dropout layers are used in the expansive path to refine the segmentation output.
Output Layer:
The final layer (outputs) is a convolutional layer with one filter and a sigmoid activation function, producing a binary mask for segmentation. This layer outputs the segmentation mask for the input image.
Model Compilation:
The model is compiled using the Adam optimizer and binary cross-entropy loss, which is common for binary image segmentation tasks.
Return Model:
The function returns the compiled U-Net model.
Model Initialization:



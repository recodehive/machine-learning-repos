# LUNG CANCER DETECTION :

### Goal:
This Project helps us to build  a classifier using a simple Convolution Neural Network which can classify normal lung tissues from cancerous.
Lung cancer is a type of cancer that starts when abnormal cells grow in an uncontrolled way in the lungs.

## Dataset

The link to the dataset is given below :-

**Link :- **

##  Description
Cancer is a disease in which cells in the body grow out of control. When cancer starts in the lungs, it is called lung cancer.
Lung cancer begins in the lungs and may spread to lymph nodes or other organs in the body, such as the brain. Cancer from other organs also may spread to the lungs. When cancer cells spread from one organ to another, they are called metastases.One of the main cause of lung cancer is smoking.
In this model First the image is passed through the convolutional layer which is used to extract important features from the input images.In this layer, the mathematical operation of convolution is performed between the input image and a filter of a particular size MxM. The convolution layer in CNN passes the result to the next layer-Pooling layer the primary aim of this layer is to decrease the size of the convolved feature map to reduce the computational costs. This is performed by decreasing the connections between layers and independently operates on each feature map.The Fully Connected (FC) layer consists of the weights and biases along with the neurons and is used to connect the neurons between two different layers. These layers are usually placed before the output layer and form the last few layers of a CNN Architecture.



### About CNN model: 
What I have used in the model:

#### Conv 2D:-
- Conv2D typically refers to a two-dimensional convolution operation in the context of deep learning and neural networks.
- Convolutional neural networks (CNNs) use these operations to extract features from input data, such as images.

#### MaxPooling 2D:-
- MaxPooling2D is a common operation in Convolutional Neural Networks (CNNs), used to downsample or reduce the spatial dimensions of feature maps. 
- It is a specific type of pooling operation, and in this case, it operates on 2D data, often used with image data in computer vision tasks.In each region, MaxPooling2D selects the maximum value. This means that for each region, it retains the most important feature or the strongest activation.


#### Dropout:-
- Dropout is a regularization technique commonly used in deep neural networks to prevent overfitting. 
- During each training iteration, for each neuron in a particular layer, Dropout randomly sets it to zero with a certain probability (dropout rate). This means that the neuron's output is effectively removed from the network for that iteration.

#### Flatten;-
- Flatten refers to an operation that converts a multidimensional array or tensor into a one-dimensional array or vector.
-  This operation is often used in neural network architectures when transitioning from convolutional or pooling layers to fully connected (dense) layers.


#### Dense;-
- Dense Layer is simple layer of neurons in which each neuron receives input from all the neurons of previous layer, thus called as dense. Dense Layer is used to classify image based on output from convolutional layers. Working of single neuron.


## What I have done!
- Linking dataset to the project.
- Unzipping the dataset.
- Defining path of respective dataset.
- Importing libraries necessary of the project.
- Created  4 convolutional layers, followed by Maxpooling layers. 
- Flatten layer to flatten the output of convolutional layer.
- Dense layers were  added for the classification.
- Dropout was then added to take care of overfitting.
- Activation function Relu and sigmoid is used to improve acccuracy.
- Optimizer Adam is used to reduce loss.
- Displayed summary of the model created and CLASS Indices.
- Generated Validation Dataset.
- Trained the Model



## Conclusion:-
This study has been conducted to demonstrate the effective and accurate diagnosis of lung cancer using CNN which was trained on chest X-ray image dataset.
The model training was performed incrementally with different datasets to attain maximum accuracy and performance.
After preprocessing the dataset, the final dataset consisted of a total of 284 X-ray images.
For training and testing the proposed CNN,the dataset was partitioned into two subsets. 
The training dataset(train)contanined 224 X-ray images with two subclasses(cancer and normal lung tissue),and 
the validation dataset(Val) contained 60 X-ray images with two subclasses(cancer and 
normal lung tissue).The model achieved an accuracy of 96%.


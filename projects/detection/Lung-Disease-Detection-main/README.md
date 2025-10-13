# Lung-Disease-Detection
Applied deep learning and CNN to detecting and classified lung disease using imagery data. We used the images that belong to 4 categories : healthy, covid-19, viral pneumonia, and bacterial pneunomia. Each categories are consist of 133 images and we used it to develop model that could detect and classify the images in less than 1 minutes.

For the image dataset that i used on this project, i put it on google drive. I put label in each images categories with numbers from 0 to 3 ( 0 = covid-19, 1 = normal, 2 = viral pneumonia, 3 = bacterial pneumonia)

Link for Image Dataset = https://drive.google.com/drive/u/0/folders/1tCjbWZD9QUbOE_4Nke2bOP7-2DJrb66t

Link for Image Testing = https://drive.google.com/drive/u/0/folders/1iL13H_ahh9zUDVNFeBqHaTqMCu_SSjIn

# 1. Understand the Problem Statement and Business Case

Deep learning has been proven to superior in detecting and classifying disease using imagery data. Skin cancer could be detected more accurately by Deep Learning than by dermatologist (2018). Human dermatologist can detect skin cancer with 86.6% accuracy while deep learning can detect skin cancer with 95% accuracy. ( reference : "Computer learns to detect skin cancer more accurately than doctors". The Guardian. 29 May 2018)

In this project, we try to develop a model that could detect and classify lung disease using 133 X-Ray images that belong to 4 classes : Healthy, Covid-19, Bacterial Pneumonia, and Viral Pneumonia.

# 2. Import Libraries and Datasets

To import our data, we used image generator to generate tensor images data and normalize them. We used 428 images for training (80%) and 104 images for our validation (20&). Before we generate our data, we perform shuffling to prevents the model from learning the order of the training.

We generate a batch of 40 images and labels. The following is label names for each classifications :
- 0 = Covid-19
- 1 = Normal
- 2 = Viral Pneumonia
- 3 = Bacterial Pneumonia

# 3. Visualize Dataset

The following is the result of visualizing our batch of 40 images and labels :

![Data Vis 1](https://user-images.githubusercontent.com/107464383/196139006-108342eb-9732-496d-b88d-69e03fbb289a.PNG)

# 4. Import Model with Pretrained Weights

Instead of build our model from scratch, we import ResNet50 and transfer the knowledge from ResNet50 to our model (transfer learning). In transfer learning, a base (reference) Artificial Neural Network on a base dataset and function is being trained. Then, this trained network weights are then repurposed in a second ANN to be trained on a new dataset and function. 

The following is imported ResNet50 model :

![Resnet Model](https://user-images.githubusercontent.com/107464383/196141879-19edf406-0d88-4f59-ae5f-fa44e887fb84.PNG)

After import the model, we freeze the last 10 layers of the model and replace it with our own.

# 5. Build and Train Deep Learning Model

After importing ResNet50 Model, we compile our model with optimizers.RMSprop for our optimizer and accuracy for our metrics. We fit our model with 50 epochs and use early stopping to exit training if validation loss is not decreasing after certain number of epochs, the following is the result for our model :

![Deep Learning Model](https://user-images.githubusercontent.com/107464383/196143740-c3165150-0a01-49f9-b6e9-88691bf305e1.PNG)

![Model Vis](https://user-images.githubusercontent.com/107464383/196144913-91f78762-f1f2-414b-8cae-4d688d96983b.PNG)


### Author

Sourik Das






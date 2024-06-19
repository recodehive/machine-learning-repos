<h1>U-Net using PyTorch</h1>

<h2>Overview</h2>
This project is a deep learning-based image segmentation solution using the UNet architecture. It is designed to perform image segmentation tasks on the Carvana image segmentation dataset. The model's performance is evaluated using both accuracy and the Dice score.

<h3>Table of Contents</h3>
<ul>
<li>Introduction</li>
<li>Dataset</li>
<li>Model Architecture</li>
<li>Performance Metrics</li>
<li>Results</li>
</ul>

<h2>Introduction</h2>
UNet is a convolutional neural network architecture designed specifically for biomedical image segmentation. In this project, we apply UNet to the Carvana image segmentation dataset to predict the pixel-wise segmentation of car images.

U-Net: Convolutional Networks for Biomedical Image Segmentation: <a href="url">https://arxiv.org/pdf/1505.04597</a> (link to the research paper).


<h2>Dataset</h2>
The Carvana image segmentation dataset(taken from kaggle) consists of high-resolution car images with corresponding masks indicating the car's location in each image. The dataset is ideal for training and evaluating segmentation models.

<h2>Model Architecture</h2>
The UNet architecture is known for its U-shaped design, featuring an encoder (contracting path) and a decoder (expanding path). This allows the model to capture context and spatial information effectively, making it suitable for image segmentation tasks.

![image](https://github.com/tanyasheru23/practice/assets/115335731/e69332e0-9fc0-4325-903a-b24cc2582abe)

<h2>Performance Metrics</h2>
To evaluate the model's performance, we use two metrics:

<ul>
  <li>Accuracy: Measures the proportion of correctly predicted pixels.</li>
  <li>Dice Score: A measure of overlap between the predicted segmentation and the ground truth, defined as:
    
  Dice Score = ![image](https://github.com/tanyasheru23/practice/assets/115335731/b5d2f539-fe66-4fbd-a20b-74edd7930e08)
   where `X` is the set of pixels in the predicted mask, and `Y` is the set of pixels in the ground truth mask. The Dice Score ranges from 0 to 1, where 0 indicates no overlap and 1 indicates perfect overlap.
  </li>
</ul>

<h2>Results</h2>
The model achieves impressive results on the Carvana image segmentation dataset:

<ul>
  <li>Accuracy: 99.635%</li>
  <li>Dice Score: 0.991</li>
</ul>

Here are some of the results:

![image](https://github.com/tanyasheru23/practice/assets/115335731/9145030f-7a24-47dc-b12d-8c41f61544d3)
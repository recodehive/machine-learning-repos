# ML_projects

This project was created for a workshop organized by the Google Developer Student Club at the University of Manchester. The repository contains all the necessary files and information for the Neural Style Transfer Project.
Links needed for the project:
- Pretrained Model - https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2
<!-- Project Link (final with outcomes) - https://github.com/ChitteshKumar/machine_learning_projects/blob/main/NeuralStyleTransfer_Method2.ipynb -->


# Neural Style Transfer

## Table of Contents
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Dependencies](#dependencies)
6. [How It Works](#how-it-works)
7. [Future Enhancements](#future-enhancements)

---

### Introduction

**Neural Style Transfer** is a project that applies the artistic style of one image (style image) to another image (content image) using deep learning techniques. The project leverages a neural network to create new images by merging the content of one image with the style of another.

---

### Project Structure

```bash
NeuralStyleTransfer/
│
├── images/                 # Folder containing input content images
│   ├── rabit.jpg
│   └── city.jpg
│   └── ...
│
├── styles/                 # Folder containing input style images
│   ├── style1.jpg
│   └── ...
│
├── generated/              # Folder for storing generated styled images
│   ├── generated_image_{}.jpg
│   └── ...
│
├── requirements.txt        # List of dependencies
├── how-to-use.md               # Project documentation (this file)
└── Neural_Style_Transfer.ipynb                 # Main script to run the neural style transfer
```
--- 

### Installation
To run the project locally, follow these steps:
1. Install the required dependencies:
```bash
pip install -r requirements.txt
```
2. Clone the repository:
```bash
git clone https://github.com/your-username/NeuralStyleLearning.git
cd NeuralStyleLearning
```

--- 

### How It Works
The Neural Style Transfer algorithm works by combining the content of one image with the style of another using a deep neural network. It minimizes the differences between:

1. The content features of the generated image and the content image.
2. The style features of the generated image and the style image.

- Content Image: Preserves the structure and details.
- Style Image: Adds artistic characteristics, such as brush strokes and color patterns.
- Generated Image: The final result that merges content and style.

---

### Future Enhancements
Here are some potential improvements for future releases:

- Implement support for batch processing multiple images.
- Add more advanced style transfer techniques for real-time processing.
- Enhance the UI or add a GUI for easier use.
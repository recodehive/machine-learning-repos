# 🎨 Style Transfer with Neural Networks 🖼️

Welcome to the **Style Transfer with Neural Networks** project! In this project, we explore the application of neural networks to perform artistic style transfer, allowing us to blend the artistic style of one image with the content of another. 

## 📖 Table of Contents

1. [Project Overview](#-project-overview)
2. [Setup & Installation](#-setup--installation)
3. [File Structure](#-file-structure)
4. [How to Run](#-how-to-run)
5. [Sample Run](#-sample-run)
6. [Concepts Behind the Project](#-concepts-behind-the-project)
7. [Technologies Used](#-technologies-used)
8. [Parameters & Tuning](#-parameters--tuning)
9. [License](#-license)
10. [Contact](#-contact)

## 📚 Project Overview

Style transfer is a technique in deep learning that involves using Convolutional Neural Networks (CNNs) to separate and combine the style of one image with the content of another image. This project implements a neural style transfer model using **PyTorch** and **pre-trained VGG19**.

### Key Features:
- Use of **pre-trained neural networks** for feature extraction (VGG19)
- Content loss and style loss based on **Gram matrices**
- Supports custom content and style images
- Optimization with **L-BFGS** optimizer for faster convergence

## 🛠️ Setup & Installation

To get started with this project, you'll need to install the following dependencies:

```bash
pip install torch torchvision matplotlib Pillow
```
## 📁 File Structure
| File/Folder                              | Description                                                         |
|------------------------------------------|---------------------------------------------------------------------|
| `Style Transfer with Neural Networks.ipynb` | The main notebook implementing style transfer                       |
| `images/`                                | Directory containing sample content and style images               |
| `output/`                                | Folder to store the generated images                               |
| `requirements.txt`                       | List of dependencies                                               |

## 🚀 How to Run
1. Clone the repository and navigate to the project folder:
    ```bash
    git clone https://github.com/yourusername/style-transfer-project.git
    cd style-transfer-project
    ```

2. Run the Jupyter Notebook to perform style transfer on your images. You can upload your custom content and style images in the notebook.

3. Adjust parameters like `content_weight`, `style_weight`, and `num_steps` for different results.

4. View and save the generated images in the `output/` folder.

## 📸 Sample Run

- **Content Image**: Defines the structure and objects in the final image.
- **Style Image**: The textures, colors, and patterns are provided.

| Content Image                                                                                  | Style Image                                                                                   |
|------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| ![Content Image](https://raw.githubusercontent.com/alo7lika/machine-learning-repos/refs/heads/main/Neural%20Style%20Transfer/Style%20Transfer%20with%20Neural%20Networks/content%20image.jpg) | ![Style Image](https://raw.githubusercontent.com/alo7lika/machine-learning-repos/refs/heads/main/Neural%20Style%20Transfer/Style%20Transfer%20with%20Neural%20Networks/style%20image.jpg) |

## 🔬 Concepts Behind the Project
- **Content Representation**: Extracted from deeper layers of the neural network to capture the high-level structures in the image.
- **Style Representation**: Captured using the Gram matrix of feature maps, representing the correlations between different feature maps.
- **Optimization**: The neural network optimizes a random noise image to minimize content and style loss, blending the content and style.

## 🧠 Technologies Used
- Python 🐍
- PyTorch for deep learning
- Jupyter Notebook for interactive coding
- Matplotlib for visualizations

## 📊 Parameters & Tuning
You can adjust the following parameters to control the output:

| Parameter       | Default Value | Description                      |
|------------------|---------------|----------------------------------|
| `content_weight` | 1e5          | Weight for the content loss      |
| `style_weight`   | 1e10         | Weight for the style loss        |
| `num_steps`      | 300          | Number of optimization steps      |
| `learning_rate`  | 0.01         | Learning rate for the optimizer   |


## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 💬 Contact
If you have any questions, feel free to reach out to me at [alolikabhowmik72@gmail.com]


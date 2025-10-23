# Erasing Clouds from Satellite Imagery

This project focuses on developing a deep learning model to remove clouds from satellite imagery. Clouds obstruct critical information in satellite images, making it difficult to analyze geographic, agricultural, and environmental features. By building a model that can effectively "erase" clouds, this project aims to provide clearer, more accurate satellite images for better analysis and decision-making.

## Overview

Satellite images are essential for various industries such as agriculture, weather forecasting, environmental monitoring, and more. However, clouds often cover parts of these images, obstructing the view of the Earth's surface. This project provides a solution by processing satellite images and removing clouds, resulting in a clearer image of the terrain below.

The process involves training a deep learning model on paired images — one set of images containing clouds and another set of the same location without clouds. Through multiple training iterations, the model learns how to predict and generate cloud-free images, enhancing the usability of satellite imagery.

## Features

- **Cloud Removal from Satellite Images**: The model removes cloud cover from satellite images to reveal the underlying terrain.
- **Loss Visualization**: Training progress is visualized using loss plots for both the generator and discriminator components.
- **Customizable Design**: The model's training parameters can be adjusted to enhance performance based on specific datasets.
- **Visual Representation**: The model provides visual feedback on training progress, with clear indications of where improvements are being made.

## Setup

To run this project locally, follow these steps:

### Prerequisites

- Python 3.x
- Required libraries:
  - `matplotlib`
  - `numpy`
  - `torch`
  - `PIL`

Install the required libraries using pip:

```bash
pip install matplotlib numpy torch pillow
```

### Clone the Repository

Clone this repository to your local machine:

```bash
git clone <repository-url>
cd <repository-directory>
```

### Running the Project

1. Place the training and validation datasets in the appropriate folder (dataset structure to be defined based on your use case).
3. Losses for both the generator and discriminator are plotted and saved as images after training.

### Visualizing Losses

The loss curves for the training process can be visualized using `matplotlib`:

```python
plt.show()
```

You can find the generated loss plots in the project directory as `Loss.jpg`.

## Usage

After training, the model can be used to process new satellite images and remove cloud cover:

```python
python inference.py --input_image <path_to_cloudy_image>
```

This will output a cloud-free image that can be used for further analysis.


# Image Steganography Using Two LSB Embedding

This project implements a simple image steganography technique using the least significant bits (LSB) embedding method. The goal is to hide a secret image inside a cover image by replacing the least significant bits of the cover image with the bits from the secret image. The process includes both embedding and extraction functionalities.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Usage](#usage)
  - [Embedding Process](#embedding-process)
  - [Extraction Process](#extraction-process)
  - [MSE and PSNR Calculation](#mse-and-psnr-calculation)
- [Output](#output)

## Introduction

Steganography is a technique of hiding information in other digital media. In this project, we hide a 64x64 grayscale secret image inside a 512x512 grayscale cover image using the 2 least significant bits of the cover image's pixels. This process ensures minimal distortion in the cover image while allowing the secret image to be extracted later.

## Features
- Embeds a secret image into a cover image using the two least significant bits (LSB).
- Extracts the hidden secret image from the stego image.
- Computes Mean Squared Error (MSE) and Peak Signal-to-Noise Ratio (PSNR) between images.

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Matplotlib

You can install the required libraries using the following command:

```bash
pip install numpy opencv-python matplotlib
```

## Usage
- ### Embedding Process
  The `Embedding_two_lsb` function hides the secret image into the cover image using the LSB method. The cover image and secret image are resized to 512x512 and 64x64 respectively before embedding.
- ### Extraction Process
  The `Extraction_two_lsb` function retrieves the hidden secret image from the stego image.
- ### MSE and PSNR Calculation
  The `mse_psnr` function calculates the Mean Squared Error (MSE) and Peak Signal-to-Noise Ratio (PSNR) between two images. These metrics help quantify the distortion between the original and stego images, and      the original and extracted secret images.

## Output
The output includes:

- The cover image and secret image displayed using Matplotlib.
- The stego image after embedding the secret image.
- The extracted secret image after applying the extraction algorithm.
- The MSE and PSNR values.
- ### Sample Output (PSNR):

```
MSE between Cover Image and Stego Image: 0.1834
PSNR: 55.4957

MSE between Secret Image and Extracted Image: 0.0
PSNR: -999 (Perfect recovery, no error)
```



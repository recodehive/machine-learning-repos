# Front Face Generator using Pix2Pix

This project is a front-face image generation tool using the Pix2Pix model, a conditional GAN (Generative Adversarial Network). It enables frontal face generation by taking side or angled face images as inputs.

## Steps

1. **Select Experiment Type**  
   Choose the type of experiment to tailor inference parameters accordingly.

2. **Define Inference Parameters**  
   Set key parameters such as image dimensions, batch size, and learning rate to optimize the inference process.

3. **Load Model**  
   Load the pre-trained Pix2Pix model for frontal face generation.

4. **Align Image**  
   Preprocess input images to ensure they are correctly aligned for frontal face transformation.

5. **Visualize Input**  
   Display the input image before performing inference, ensuring the image aligns with requirements.

6. **Perform Inference**  
   Run the inference process to generate the frontal face image from the input image.

7. **Visualize Result**  
   Display the final output, showing the generated frontal face for evaluation and analysis.

   ![image](https://github.com/user-attachments/assets/a19f2377-968c-4dfd-b4fe-38c17b09a307)


## Requirements

- Python 3.x
- TensorFlow or PyTorch
- OpenCV for image processing
- Other dependencies as specified in `notebook`

This project demonstrates the capabilities of Pix2Pix in transforming side-angle face images into realistic frontal face images.

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load and Display the Image
def load_image(image_path):
    """
    Load an image from file and convert it to RGB format.
    
    Parameters:
    image_path (str): Path to the image file.
    
    Returns:
    np.array: Loaded image in RGB format.
    """
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Step 2: Smooth and Enhance Cartoon Effect
def smooth_anime_effect(image):
    """
    Apply a smoother, more anime-like effect with soft edges, vibrant colors,
    and smooth shading transitions.
    
    Parameters:
    image (np.array): Input image in RGB format.
    
    Returns:
    np.array: Smoothed anime-like cartoon image.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Use a larger blur to smooth the gray image
    gray_blur = cv2.GaussianBlur(gray, (9, 9), 0)
    
    # Detect edges using adaptive thresholding for softer edges
    edges = cv2.adaptiveThreshold(
        gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2
    )
    
    # Create a smooth, vibrant version of the original image
    smooth_color = cv2.bilateralFilter(image, d=10, sigmaColor=300, sigmaSpace=300)
    
    # Boost contrast and vibrancy to mimic the anime style
    hsv = cv2.cvtColor(smooth_color, cv2.COLOR_RGB2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.4, 0, 255)  # Increase saturation
    hsv[..., 2] = np.clip(hsv[..., 2] * 1.1, 0, 255)  # Slight brightness boost
    vibrant_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    # Combine the vibrant color image with softened edges
    cartoon = cv2.bitwise_and(vibrant_image, vibrant_image, mask=edges)
    
    return cartoon

# Step 3: Display Original and Smoothed Cartoon Images Side by Side
def display_images(original, cartoon):
    """
    Display the original and anime-like cartoon images side by side.
    
    Parameters:
    original (np.array): Original image.
    cartoon (np.array): Cartoonized image.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display Original Image
    axes[0].imshow(original)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Display Anime-Like Cartoon Image
    axes[1].imshow(cartoon)
    axes[1].set_title('AI Anime Avatar Image')
    axes[1].axis('off')
    
    plt.show()

# Step 4: Generate the Anime Avatar
def generate_anime_avatar(image_path):
    """
    Generate a smooth anime-like avatar using soft edges and vibrant colors.
    
    Parameters:
    image_path (str): Path to the input image file.
    """
    # Load the original image
    original_image = load_image(image_path)
    
    # Apply smooth anime effect
    cartoon_image = smooth_anime_effect(original_image)
    
    # Display the original and anime-like cartoon images
    display_images(original_image, cartoon_image)

# Usage (replace with your image path)
generate_anime_avatar("elon intro.jpg")

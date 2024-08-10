import cv2
import numpy as np
import pytesseract
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox
import os

# Path to Tesseract executable file
tesseract_path = r'C:\Users\kulitesh\Scrape-ML\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = tesseract_path

# Folder path for storing files in
output_folder = r'C:\Users\kulitesh\Scrape-ML\Fine-tuning ocr extraction\image_Preprocessing\Output'

# Function to extract text from an image and store it in a file
def extract_text(image_path, filename):
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(image_path, config=custom_config)
    output_text_path = os.path.join(output_folder, filename + '_text.txt')
    with open(output_text_path, 'w', encoding='utf-8') as file:
        file.write(text)

# Function to process image (optional) and store the manipulated image in a file
def process_and_save_image(image_path, options):
    image = cv2.imread(image_path)
    original = image.copy()

    if options['grayscale']:
        output_filename = 'grayscale'
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif options['threshold']:
        output_filename = 'threshold'
        _, image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)
    elif options['adaptive_threshold']:
        output_filename = 'adaptive_threshold'
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    elif options['denoise']:
        output_filename = 'denoise'
        image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    output_image_path = os.path.join(output_folder, output_filename + '_image.png')
    cv2.imwrite(output_image_path, image)

# Function to open file dialog and initiate text extraction or image processing
def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
    if file_path:
        options = {
            'grayscale': grayscale_var.get(),
            'threshold': threshold_var.get(),
            'adaptive_threshold': adaptive_threshold_var.get(),
            'denoise': denoise_var.get(),
        }

        if not any(options.values()):  # If no manipulation is chosen, extract text directly
            extract_text(file_path, 'original')
        else:
            process_and_save_image(file_path, options)
            for option, selected in options.items():
                if selected:
                    extract_text(file_path if option != 'adaptive_threshold' else output_folder + "\adaptive_threshold_image.png", option)

        messagebox.showinfo("Extraction Complete", "Text extracted and stored in respective files.")

# GUI setup
root = tk.Tk()
root.title("OCR Text Extraction")

canvas = tk.Canvas(root, width=800, height=600)
canvas.pack()

frame = tk.Frame(root)
frame.place(relx=0.5, rely=0.1, anchor='n')

button = tk.Button(frame, text="Open Image", command=open_file)
button.pack(side=tk.LEFT)

options_frame = tk.Frame(root)
options_frame.place(relx=0.5, rely=0.3, anchor='n')

grayscale_var = tk.BooleanVar()
threshold_var = tk.BooleanVar()
adaptive_threshold_var = tk.BooleanVar()
denoise_var = tk.BooleanVar()

grayscale_check = tk.Checkbutton(options_frame, text="Grayscale", variable=grayscale_var)
grayscale_check.pack(side=tk.LEFT)
threshold_check = tk.Checkbutton(options_frame, text="Threshold", variable=threshold_var)
threshold_check.pack(side=tk.LEFT)
adaptive_threshold_check = tk.Checkbutton(options_frame, text="Adaptive Threshold", variable=adaptive_threshold_var)
adaptive_threshold_check.pack(side=tk.LEFT)
denoise_check = tk.Checkbutton(options_frame, text="Denoise", variable=denoise_var)
denoise_check.pack(side=tk.LEFT)

root.mainloop()

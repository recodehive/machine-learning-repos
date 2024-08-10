import cv2
import pytesseract
import tkinter as tk
from tkinter import filedialog, messagebox
import os

# Path to Tesseract executable file
tesseract_path = r'C:\Users\kulitesh\Scrape-ML\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = tesseract_path

# Folder path for storing files in
output_folder = r'C:\Users\kulitesh\Scrape-ML\Fine-tuning ocr extraction\image_segmentation_roi\Output'

# Function to extract text from a specified region of an image ROI
def extract_text_from_roi(image, x, y, w, h, filename):
    roi = image[y:y+h, x:x+w]
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(roi, config=custom_config)
    output_text_path = os.path.join(output_folder, f'{filename}_text.txt')
    with open(output_text_path, 'w', encoding='utf-8') as file:
        file.write(text)

# Function to open file dialog and initiate text extraction from image parts
def open_file_and_extract_parts():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
    if file_path:
        image = cv2.imread(file_path)
        filename = os.path.splitext(os.path.basename(file_path))[0]

        # Example: Divide image into 4 equal parts (adjust as needed)
        height, width, _ = image.shape
        part_width = width // 2
        part_height = height // 2

        # Extract text from each part
        for i in range(2):
            for j in range(2):
                x = j * part_width
                y = i * part_height
                extract_text_from_roi(image, x, y, part_width, part_height, f'{filename}_part_{i}_{j}')

        messagebox.showinfo("Extraction Complete", "Text extracted from image parts.")

# GUI setup
root = tk.Tk()
root.title("OCR Text Extraction from Image Parts")

canvas = tk.Canvas(root, width=400, height=200)
canvas.pack()

button = tk.Button(root, text="Open Image and Extract Parts", command=open_file_and_extract_parts)
button.pack(pady=20)

root.mainloop()

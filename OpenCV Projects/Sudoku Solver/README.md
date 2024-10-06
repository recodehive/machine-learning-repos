# Sudoku Solver

This project is a Sudoku solver that reads a Sudoku puzzle from an image, recognizes the digits using a pre-trained OCR model, and then solves the puzzle. The final solved puzzle is displayed by overlaying the solution onto the original image.

## Features
- Extracts the Sudoku grid from an image using image processing techniques.
- Recognizes the digits in the puzzle using a pre-trained OCR model.
- Solves the puzzle using a backtracking algorithm.
- Overlays the solution onto the original image.

## How it Works

### 1. Image Preprocessing:
   The input image is processed to detect the Sudoku grid. Contours are used to locate the largest rectangular region in the image, which is assumed to be the Sudoku puzzle.

### 2. Digit Recognition:
   The extracted grid is split into 81 cells, and each cell is resized to match the input size of the pre-trained OCR model. The model predicts the digits in the puzzle, assigning a number to each cell or marking it as empty.

### 3. Puzzle Solving:
   The recognized digits are passed into a backtracking algorithm, which solves the Sudoku puzzle. If the puzzle cannot be solved, the model might have misread a digit.

### 4. Output:
   The solved digits are overlaid back onto the original image using inverse perspective transformation, showing the solution in the correct position.

## Usage

1. Place your Sudoku puzzle image in the project directory (e.g., `sudoku1.jpg`).
2. Run the script.

# Harry Potter Spells Data Analysis (EDA) using Machine Learning

This project performs exploratory data analysis (EDA) on a dataset containing information about spells from the Harry Potter series. The analysis includes generating word clouds for spells that produce different colored lights and visualizing these word clouds with corresponding color schemes.

## Project Structure

- `hp_spells.csv`: Dataset containing Harry Potter spells.
- `main.py`: Main script for data analysis and visualization.
- `README.md`: Project documentation.
- `requirements.txt`: List of Python libraries required for the project.

## Prerequisites

Ensure you have Python 3.x installed on your machine.

## Installation



2. Install the required Python packages:
    
    pip install -r requirements.txt
    ```

## Data Analysis

The script performs the following steps:

1. Defines color functions for different spell light colors.
2. Converts the `Spell Name` column to string type and concatenates it with the `Effect` column to create a new column `Spell Name and Effect`.
3. Filters spells based on the color of light they produce.
4. Generates and displays word clouds for each color of light, with colors corresponding to the light color.

## Visualizations

Word clouds are generated for spells producing the following light colors:
- Red
- Blue
- Black smoke
- Bright yellow
- Light green

Each word cloud is colored according to the spell light color.

## Example Output

Here are example word clouds for each spell light color:

### Red Spells

### Blue Spells

### Black Smoke Spells


### Bright Yellow Spells


### Light Green Spells




# Movie Revenue Prediction
This project aims to predict movie revenue based on various features such as budget, popularity, and runtime using a linear regression model.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Results](#results)

## Installation

To run this project, you'll need to have Python installed along with the following libraries:
```requirements.txt
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn
```
You can install the necessary libraries using pip:

```bash
pip install -r requirements.txt
```

## Dataset

The dataset used in this project is `movie_dataset.csv`, which contains information about various movies, including their budget, revenue, popularity, runtime, and more.

## Usage

1. Clone the repository:

```bash
git clone <your-repo-url>
cd <your-repo-directory>
```

2. Open the Jupyter Notebook:

```bash
jupyter notebook
```

3. Run the notebook cells sequentially to load the dataset, preprocess it, and train the linear regression model to make predictions on movie revenue.

## Results

After running the model, you will receive the Mean Absolute Error (MAE) as an evaluation metric to assess the prediction accuracy.

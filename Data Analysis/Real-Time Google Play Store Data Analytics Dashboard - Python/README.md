# Real-Time Google Play Store Data Analytics Dashboard

This project provides a real-time data analytics dashboard for the Google Play Store, built using Python. The dashboard presents interactive visualizations based on Google Play Store app data and user reviews.

## Project Structure
- `Real_Time_Google_Play_Store_Data_Analytics_Dashboard_Python.ipynb`- Contains the .ipynb file (Jupyter notebook).
- `Datasets/`
  - `play_store.csv` - Contains data of Google Play Store apps.
  - `user_reviews.csv` - Contains user reviews of apps.

## Methodology
The analysis and dashboard creation is done in the accompanying `.ipynb` notebook, with the following steps:

1. **Dataset Loading**: Load both datasets (`play_store.csv` and `user_reviews.csv`).
2. **Data Cleaning & Transformation**: Remove inconsistencies, handle missing values, and transform the data.
3. **Merged Dataset**: Combine both datasets to create a final dataset for analysis.
4. **Sentiment Analysis**: Performed sentiment analysis on user reviews.
5. **Visualization**: Used Plotly to create 10 interactive plots for detailed insights.
6. **Dashboard Creation**: A web-based dashboard was created using HTML and CSS.

## Output
Running the code generates a `dashboard.html` file that contains the final dashboard with interactive visualizations.

## How to Run
1. Clone the repository.
2. Run the `.ipynb` notebook.
3. The dashboard will be saved as `dashboard.html`.

## Libraries Used
- Pandas
- Plotly Express
- NLTK
- webbrowser
- Numpy
- os

## Demo Video - Interactive Dashboard

https://github.com/user-attachments/assets/79ed7470-4084-4ca4-a2bf-5b59c927667f



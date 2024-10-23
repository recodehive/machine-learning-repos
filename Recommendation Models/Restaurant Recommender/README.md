# Restaurant Recommender

## Introduction
- This project analyzes restaurant data from Zomato for restaurants located in Lucknow, India.
- The dataset includes various attributes such as restaurant name, address, locality, cuisine, average cost, highlights, and customer ratings. 
- The goal of this project is to explore the dataset, perform analysis, and identify the top 20 restaurants based on aggregate ratings. 
- The analysis aims to provide insights into the best restaurants in the area based on customer feedback.

## Prerequisites
- Python 3.x
- Jupyter Notebook / Google Colab
- pandas
- numpy
- geopandas
- matplotlib
- seaborn

To install: `pip install pandas numpy geopandas matplotlib seaborn`

## Methodology
1. **Exploratory Data Analysis**: Key characteristics of the data were examined, including checking for missing or duplicate entries. Summary statistics were used to understand variables like `average_cost_for_one`, `votes`, and `aggregate_rating`.
2. **Top 20 Restaurants**: The dataset was sorted by `aggregate_rating` to identify the top-rated restaurants. The top 20 were filtered and analyzed further to understand common patterns in ratings, cuisines, and locations.
3. **Data Visualization**: Various visualizations were created to illustrate relationships between key variables such as *restaurant rating*, *number of votes*, and *average cost*.
4. **Recommendation System Evaluation**: 
   - **Precision**: Precision is a measure of the accuracy of the recommendations. It tells you what proportion of the recommended items were relevant to the user. In your case, a precision of 0.4 means that 40% of the recommended restaurants were relevant to the user.
   - **Recall**: Recall measures the coverage of the relevant items in the recommendations. It indicates what proportion of the relevant items were successfully recommended. A recall of approximately 0.67 means that 67% of the relevant restaurants were included in the recommendations.
   - These values are typically between 0 and 1, with higher values indicating better performance. So, a precision of 0.4 and a recall of 0.67 suggest that the recommendations are somewhat accurate and cover a significant portion of the relevant items, but there is room for improvement.
   - **Error Metrics**: 
     - **Mean Squared Error (MSE)**: Measures the average of the squared differences between the actual and predicted values. A lower MSE indicates better model performance.
     - **Mean Absolute Error (MAE)**: Represents the average magnitude of the prediction errors, with smaller values preferred.
     - **Root Mean Squared Error (RMSE)**: Used to measure the standard deviation of the prediction errors; lower values are better.
5. **Clustering Analysis**: The "Elbow Method" is a technique to determine the optimal number of clusters for a K-Means clustering algorithm. It looks for an "elbow" point in the plot where the distortion starts to decrease at a slower rate. The number of clusters corresponding to this point is considered optimal for clustering your data. The code helps you visualize this concept by plotting distortions for different values of k.

## Results
- **Top-rated Restaurants**:
    - Barbeque Nation (Rating: 4.9)
    - Pirates of Grill (Rating: 4.8)
    - Farzi Café (Rating: 4.7)
- **Popular Cuisines**:
    - North Indian
    - Mughlai
    - Continental
    - Modern Indian
- **Location Insights**: Restaurants in high-demand localities such as *Gomti Nagar* and *Chowk* have higher ratings and more customer votes.

## Conclusion
- This project provides an in-depth analysis of restaurant data from Zomato, specifically focusing on customer preferences and ratings in Lucknow. 
- The top restaurants are distinguished by their cuisines and prime locations, offering valuable insights for food enthusiasts and restaurant owners. 
- Additionally, the evaluation metrics provide a framework for understanding the effectiveness of the recommendation system and areas for future improvement.

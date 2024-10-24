# Restaurant Recommender System

This repository contains a restaurant recommendation system that uses restaurant data from a CSV file. The system can recommend restaurants based on their popularity, location, cuisines, and user ratings.

## Project Structure

- `Restaurant_Reservation.ipynb`: This notebook contains the implementation of the restaurant recommendation system.

## Dataset

The dataset used in this project is loaded from `food1.csv` and contains information about various restaurants, including:

- `res_id`: Restaurant ID
- `name`: Name of the restaurant
- `url`: Zomato URL for the restaurant
- `address`: Full address of the restaurant
- `city`: City where the restaurant is located
- `locality`: Locality or neighborhood of the restaurant
- `cuisines`: Types of cuisines the restaurant serves
- `timings`: Operating hours of the restaurant
- `average_cost_for_one`: Average cost for one person
- `highlights`: Special features or services of the restaurant
- `aggregate_rating`: Average user rating of the restaurant
- `votes`: Number of votes or reviews for the restaurant
- `scope`: A custom score used for ranking restaurants

## Model Explanation

The restaurant recommendation system implemented in the notebook primarily ranks restaurants based on a combination of their user ratings and the number of reviews they have received. The model also allows filtering by locality, cuisines, and cost range.

**Steps Involved:**

1. **Data Exploration**:
   - The dataset is loaded using Pandas.
   - Basic exploration is done using `.head()`, `.info()`, and `.columns()` to understand the structure and details of the dataset.

2. **Data Cleaning**:
   - Missing values are handled for attributes like `timings`.

3. **Recommendation System**:
   - The recommendation system ranks restaurants based on their `aggregate_rating` and the number of `votes` they have received.
   - The user can filter restaurants based on criteria such as location (`locality`), `cuisines`, and cost (`average_cost_for_one`).

## How to Run

1. Clone the repository.
2. Install the necessary dependencies (`pandas`, `numpy`, `matplotlib`, `seaborn`, `geopandas`).
3. Open the `Restaurant_Reservation.ipynb` notebook and run the cells sequentially to explore and use the recommendation system.

## Conclusion

The restaurant recommendation system allows users to explore various restaurants and receive recommendations based on popularity, location, and cuisines. This project can be expanded by incorporating more advanced recommendation techniques, such as collaborative filtering, for personalized recommendations.

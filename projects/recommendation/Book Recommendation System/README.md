# Book Recommendation System

This repository contains two approaches to building a book recommendation system:

1. **Popularity-based Book Recommendation**
2. **Collaborative-based Book Recommendation**

Both models use the same dataset, which includes information about books, users, and user ratings.

## Project Structure

- `Popularity_based_Book_Recommendation.ipynb`: A notebook that implements a popularity-based recommendation system.
- `Collaborative_Book_Recommendation.ipynb`: A notebook that implements a collaborative filtering-based recommendation system.

## Dataset

The dataset used in both notebooks consists of three files:

1. **Books.csv**: Contains details about books such as ISBN, title, author, and publisher.
2. **Users.csv**: Information about the users, including their unique ID, location, and age.
3. **Ratings.csv**: User ratings for the books.

## Notebooks and Model Explanations

### 1. Popularity-based Book Recommendation

**Model Explanation:**

The popularity-based recommendation model ranks books based on their overall rating. The more ratings a book has, combined with the higher average rating, the more popular it is considered. The model does not take user preferences into account and simply recommends the same popular books to all users. Key steps include:

- **Data Preparation**: Loading the dataset and merging user ratings with book details.
- **Popularity Calculation**: Sorting books based on the count of user ratings and their average rating.
- **Recommendation Output**: Returning a list of the most popular books for recommendation.

This approach works well for general recommendations but lacks personalization.

### 2. Collaborative-based Book Recommendation

**Model Explanation:**

The collaborative-based recommendation model uses a more advanced approach, recommending books based on the preferences of similar users. This is done using **user-user collaborative filtering**, where the algorithm finds users with similar tastes and recommends books that those users have rated highly. The model works as follows:

- **Data Preparation**: Loading and cleaning the datasets (Books, Users, Ratings).
- **User Similarity**: Calculating the similarity between users based on their book ratings using techniques like **cosine similarity**.
- **Collaborative Filtering**: Based on the similarity scores, the model recommends books that similar users have liked but the target user hasn't yet rated.
- **Recommendation Output**: Personalized book recommendations for each user based on their preferences and the behavior of similar users.

This method is highly personalized and usually yields better results than popularity-based systems, but it requires more data and computational resources.

## How to Run

1. Clone the repository.
2. Install the necessary dependencies (`pandas`, `numpy`, `matplotlib`, `seaborn`).
3. Open the notebooks and run the cells in sequence.

## Conclusion

These two models offer different strategies for recommending books. The **popularity-based model** is simple and effective for recommending top-rated books to everyone, while the **collaborative filtering model** offers personalized recommendations based on user behavior and preferences.

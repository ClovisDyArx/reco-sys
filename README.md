# DeepLove: A Movie Recommender System for Couples

## Project Overview

DeepLove is a movie recommender system designed to recommend movies to couples based on their individual preferences.

The system combines the preferences of both users to suggest movies that both might enjoy.

## Dataset

We use the following datasets for this project:
- MovieLens Dataset: [MovieLens 1M](http://files.grouplens.org/datasets/movielens/ml-1m)
- IMDb Dataset: [IMDb Datasets](https://www.imdb.com/interfaces/)

## Methodology

The project is implemented in several steps:

1. **Data Collection and Preprocessing**:
   - Download the MovieLens and IMDb datasets.
   - Merge the datasets to include movie ratings, genres, and other data (runtime, type, ...).

2. **Feature Engineering**:
   - Extract relevant features such as genres and ratings.
   - Create a user-item interaction matrix for collaborative filtering.
   - Combine user data to create data for the couple of users.

3. **Model Development**:
   - Implement a recommender system algorithm to predict the rating of a movie by a couple of users.
   - TODO

4. **Recommendation Algorithm**:
   - Develop an algorithm to suggest movies that might be liked by the couple.
   - TODO

5. **Evaluation**:
   - Split the data into training and testing sets.
   - Evaluate the model using RMSE.
   - TODO

## Code Structure

### Data Loading

We start by loading the MovieLens dataset using `requests` to download the dataset and `zipfile` to extract it. The data is then read into pandas DataFrames.

### Data Preprocessing

We preprocess the `df_movies` DataFrame by splitting the `Genres` column into multiple rows using `explode`. We then perform one-hot encoding on the genres and concatenate the resulting columns with the original DataFrame. Finally, we group by `MovieId` and `Title` to aggregate the genres for each movie.

### User-Item Interaction Matrix

We create a user-item interaction matrix using the `pivot` function. This matrix has users as rows, movies as columns, and ratings as values.

### Couple Matrix Function

We define a function to create a couple matrix by averaging the ratings of two users. The function takes the user-item interaction matrix and two user IDs as inputs. It filters out movies that haven't been rated by both users and calculates the average rating for each movie rated by both users.

### SVD Model Training

We define a function to train an SVD model using the `TruncatedSVD` class from sklearn. We fill missing values in the user-item matrix with zeros and fit the SVD model.

### Predict Ratings

We define a function to predict ratings using the trained SVD model. We fill missing values with zeros, transform the user-item matrix using the SVD model, and then apply the inverse transform to get the predicted ratings. We convert the predicted ratings into a DataFrame.

### Evaluate Model

We define a function to evaluate the SVD model using RMSE as the evaluation metric. We split the data into training and testing sets. We re-create the user-item matrix for the training data and train the SVD model. We evaluate the model on the test data and print the RMSE.

## Results

- The RMSE obtained on the training and test data provides an indication of the model's performance. 
- Based on the RMSE values, we can assess the accuracy of our model in predicting ratings.

## Conclusion

The DeepLove project demonstrates a practical approach to building a movie recommender system for couples. The system effectively combines the preferences of both users to suggest movies that both might enjoy. The project highlights the importance of data preprocessing, feature engineering, and model evaluation in building a robust recommender system.

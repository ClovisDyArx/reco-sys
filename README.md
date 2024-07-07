# DeepLove: A Movie Recommender System for Couples

## Project Overview

DeepLove is a movie recommender system designed to recommend movies to couples based on their individual preferences. The system combines the preferences of both users to suggest movies that both might enjoy.

## Dataset

We use the following datasets for this project:
- **MovieLens Dataset**: [MovieLens 1M](http://files.grouplens.org/datasets/movielens/ml-1m)
  - This dataset contains 1 million ratings from 6,040 users on 3,952 movies.
- **IMDb Dataset**: [IMDb Datasets](https://www.imdb.com/interfaces/)
  - This dataset includes extensive information about movies such as genres, runtime, average ratings, and more.
  - It has way more films than the MovieLens dataset however we will mainly focus on films present in both datasets, to avoid issues with personnal ratings.

## Methodology

The project is implemented in several steps:

### 1. Data Collection and Preprocessing
- **Data Collection**: We collected data from the MovieLens and IMDb datasets. We imported both datasets we different methods because the formats were differents.
- **Data Merging**: The datasets were merged together to include movie ratings, genres, runtime, type, and additional movie metadata.
  The merge 'deleted' a lot of movie entries because we used the names, and year of production as the merge keys, thus films without the same exact information were ditched.
  This could be a key improvement for a larger scale product, we could use fuzzy matching to include more entries during the intial steps.
- **Preprocessing**: This step involved cleaning the data, handling missing values, and transforming the data into a suitable format for model training. Genres were split and one-hot encoded, and user ratings were combined to reflect couples' preferences.
  The preprocessing was mainly used for another type of algorithm, that i ended up not using : Transformes. Here I focused on collaborative filtering.

### 2. Feature Engineering
- **Feature Extraction**: Relevant features such as genres and ratings were extracted.
- **User-Item Interaction Matrix**: A matrix was created to represent interactions between users and items (movies), which serves as the foundation for collaborative filtering.
  This matrix was the main point of the work around this project. without it collaborative filtering would not be possible in our setting.
- **Couple Data Creation**: Individual user data was combined to form datasets representing couples, taking the average of their ratings for commonly rated movies.

### 3. Model Development
- **Algorithm Selection**: We defined a function to train an SVD model using the TruncatedSVD class from sklearn. We fill missing values in the user-item matrix with zeros and fit the SVD model.
- **Training**: The models were trained on the preprocessed data mentionned before.
- **Ratings**: Regarding the ratings, we adjusted the values so that they go from 0 to 5, since the SVD model could give out negative results that bias the results. e define a function to predict ratings using the trained SVD model. We fill missing values with zeros, transform the user-item matrix using the SVD model, and then apply the inverse transform to get the predicted ratings. We convert the predicted ratings into a DataFrame.

### 4. Recommendation Algorithm
- **Algorithm Development**: An algorithm was developed to suggest movies that both users in a couple might enjoy. The recommendations were based on the predicted ratings from the trained models.
- **Evaluation**: The algorithm's performance was evaluated using RMSE and MAE metrics to measure prediction accuracy. We defined a function to evaluate the SVD model, as we split the data into training and testing sets. We re-create the user-item matrix for the training data and train the SVD model. We evaluate the model on the test data and print the RMSE.

### 5. Evaluation
- **Data Splitting**: The data was split into training and testing sets to evaluate the model's performance, so that we would not introduce any bias.
- **Metrics**: The model was evaluated using Root Mean Square Error (RMSE) and Mean Absolute Error (MAE) to assess its accuracy in predicting ratings.
- **Results**: 
  - RMSE: 1.1212835478416474
  - MAE: 0.3584361684879515

## Results

- Overall the results are decent, not excellent but decent. I am not fully happy with the way this went, but it was interesting to try to use collaborative filtering not the same way as in the lessons, on my own.
- The relatively low RMSE and MAE values suggest that the model performs well in predicting movie ratings for couples based on their combined preferences, but as I stated right before it could be better.

## Conclusion

The DeepLove project underscores the importance of thorough data preprocessing, careful feature engineering, and rigorous model evaluation in developing a robust recommender system. \
Future work could involve exploring more sophisticated models and incorporating additional data sources to further improve recommendation accuracy. \
I tried implementing a BPR algorithm and Transformers aswell, but the results with BPR were not good at all (regardless the ratings predictions remained constant) and as I explained in the notebook I failed to implement Transformers because of coding issues. If I had a bit more time and motivation I would have done it better for sure, however this was not the case here.

## Future Work

I will just quote the end of my notebook in this part: \
I tried to implement a model based on transformers, using the example in the course (unlike here where I tried implementing collaborative filtering on my own).

However I kept struggling on the same issue over and over during the training of the model (all steps before were working fine, only training wasnt behaving well).

this was my error:
> RuntimeError: CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
Compile with TORCH_USE_CUDA_DSA to enable device-side assertions. 

I am still confused as why this happened, but I could not bring myself to solve the issue after a lot of tries.

I scraped the code so that it would not pollute the notebook, but I still have saves over on Kaggle in case.


I also did try a bit on BPR, but to no good results : whatever the id of the person rating, the recommendations remained unchanged.

#!/usr/bin/env python
# coding: utf-8

# **Part D: Bonus**
# 
# At the top of the notebook, add brief write-ups to explain each predictor you used and how you combined them.
# 
# Answer: The following are the predictors used:
# 1. **User-User recommendation predictor**: Works based on User similarities and user's own ratings
# 2. **Item-Item recommendation predictor**: Works based on business similarities and user's own ratings
# 3. **Latent hidden model recommendation predictor**: It uses Singular Value Decomposition (SVD)
# 
# I have used an ensemble recommender system using the above 3 predictors. The final output directly combines the outputs from the three models (averaging their scores) and uses this to determine whether to recommend a business (0 or 1) based on a percentile threshold.

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, pairwise
from scipy.linalg import svd

# Load the data
user_business_train_path = 'data/user-business.csv'
user_business_test_path = 'data/bonus/user-business_test.csv'
business_path = 'data/business.csv'

train_df = pd.read_csv(user_business_train_path, header=None)
test_df = pd.read_csv(user_business_test_path, header=None)
business_names = pd.read_csv(business_path, header=None)


def generate_predictions(test_user_ratings, train_data):
    # User-User Recommender System: Calculate similarities with all users in the training set
    cos_sim = pairwise.cosine_similarity(train_data[:, 100:], test_user_ratings[:, 100:]) 
    user_predictions = np.dot(cos_sim.T, train_data[:, :100])

    # Item-Item Recommender System
    business_similarities = pairwise.cosine_similarity(train_data[:, :100].T, train_data[:, 100:].T)  # Similarity between first 100 businesses and businesses 101-1000
    item_predictions = np.dot(test_user_ratings[:, 100:], business_similarities.T)

    # Latent Factor Model
    U, sigma, VT = svd(train_data, full_matrices=False)
    k = 10  # Number of latent factors
    U_k = U[:, :k]
    sigma_k = np.diag(sigma[:k])
    VT_k = VT[:k, :]
    all_ratings = np.dot(U_k, np.dot(sigma_k, VT_k))  # Full predictions
    all_predictions = all_ratings[test_df.index, :100]

    print(all_predictions.shape)

    #Average predictions from different models
    combined_predictions = (user_predictions + item_predictions + all_predictions) / 3
    return combined_predictions


# In[34]:


predictions = generate_predictions(test_df.values, train_df.values)

# Threshold predictions to get binary values (0s and 1s)
# Since the dataset is sparse, using the 95th percentile as a threshold to decide on 1s might be a good starting point
final_predictions = (predictions > np.percentile(predictions, 95, axis=1)[:, None]).astype(int)

submission_df = pd.DataFrame(final_predictions, dtype=int)
submission_df.to_csv('sample_bonus_submission.csv', header=False, index=False)


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# **Part A: User-user recommender system**

# In[1]:


import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

user_business_path = 'data/user-business.csv'
business_path = 'data/business.csv'

user_business_df = pd.read_csv(user_business_path, header=None)
business_names = pd.read_csv(business_path, header=None)

alex_ratings = user_business_df.iloc[3, 100:].values 
other_users = user_business_df.drop(3, axis=0)

cos_sim = cosine_similarity(other_users.iloc[:, 100:], [alex_ratings])
rAlex_b = np.dot(cos_sim.T, other_users.iloc[:, :100]).flatten()

top_indices = np.argsort(rAlex_b)[::-1][:5]
top_scores = rAlex_b[top_indices] 

top_businesses = business_names.iloc[top_indices].values.flatten()

print("Top 5 recommended businesses for Alex are:")
for business, score in zip(top_businesses, top_scores):
    print(f"{business}: {score:.3f}")


# Let S denote the set of the first 100 businesses (the first 100 columns of the matrix). From all the businesses in S, which are the five that have the highest similarity scores (rAlex,b) for Alex? What are their similarity scores?
# 
# Answer with their similarity score:
# Papi's Cuban & Caribbean Grill: 43.040
# Seven Lamps: 33.598
# Loca Luna: 33.263
# Farm Burger: 32.783
# Piece of Cake: 12.626

# **Part B: item – item recommender system**

# In[3]:


R = user_business_df.values
alex_ratings = R[3, :] 
R_no_alex = np.delete(R, 3, axis=0)
business_similarities = cosine_similarity(R_no_alex.T)

rAlex_b = np.zeros(100)
for b in range(100):
    rAlex_b[b] = np.dot(business_similarities[b], alex_ratings * (alex_ratings > 0))

top_indices = np.argsort(rAlex_b)[::-1][:5]
top_scores = rAlex_b[top_indices] 
top_businesses = business_names.iloc[top_indices].values.flatten()

print("Top 5 recommended businesses for Alex based on item-item similarity are:")
for business, score in zip(top_businesses, top_scores):
    print(f"{business}: {score:.3f}")


# From all the businesses in S (first 100 businesses), which are the five that have the highest similarity scores for Alex?
# 
# Answer with their similarity score:
# Papi's Cuban & Caribbean Grill: 6.811
# Farm Burger: 6.559
# Seven Lamps: 6.440
# Loca Luna: 5.853
# Piece of Cake: 3.730

# **Part C: Latent hidden model recommender system**

# In[4]:


from scipy.linalg import svd
R = user_business_df.values.astype(float)
U, sigma, VT = svd(R, full_matrices=False)

k = 10
U_k = U[:, :k]
sigma_k = np.diag(sigma[:k])
VT_k = VT[:k, :]

R_star = np.dot(U_k, np.dot(sigma_k, VT_k))

alex_ratings = R_star[3, :100]
top_indices = np.argsort(alex_ratings)[-5:][::-1] 
top_scores = alex_ratings[top_indices]
top_businesses = business_names.iloc[top_indices].values.flatten()

print("Top 5 recommended businesses for Alex based on latent factor model are:")
for business, score in zip(top_businesses, top_scores):
    print(f"{business}: {score}")


# 

# From the R* matrix, select the top 5 businesses for Alex in S (first 100 businesses). Again, hand in the names of the businesses and their similarity score.
# 
# Answer with their similarity score:
# Papi's Cuban & Caribbean Grill: 1.1905064199911006
# Loca Luna: 0.8762545708302192
# Farm Burger: 0.8578263876279587
# Seven Lamps: 0.8179473119616172
# Piece of Cake: 0.2993543937609191

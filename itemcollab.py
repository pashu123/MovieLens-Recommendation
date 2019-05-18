import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime
from sortedcontainers import SortedList


with open('user2movie.json','rb') as f:
    user2movie = pickle.load(f)

with open('movie2user.json','rb') as f:
    movie2user = pickle.load(f)

with open('usermovie2rating.json','rb') as f:
    usermovie2rating = pickle.load(f)

with open('user2movierating_test.json','rb') as f:
    usermovie2rating_test = pickle.load(f)


N = np.max(list(user2movie.keys())) + 1

# test set may contain movies the train set doesn't have data on
m1 = np.max(list(movie2user.keys()))
m2 = np.max([m for (u,m),r in usermovie2rating_test.items()])

M = max(m1,m2) + 1

# no. of common items
K = 20
# no. of common movies users have in common
limit = 5
neighbours = [] #store neighbours in a list
averages = [] # each user's average rating for later use
deviations = [] # each user's deviation for later use
print('All preprocessing done')

for i in range(M):

    users_i = movie2user[i]
    users_i_set = set(users_i)

    # lets calculate the average and deviation
    ratings_i = {user: usermovie2rating[(user,i)] for user in users_i}
    avg_i = np.mean(list(ratings_i.values()))
    dev_i = {user : (rating - avg_i) for user, rating in ratings_i.items()}
    dev_i_values = np.array(list(dev_i.values()))
    sigma_i = np.sqrt(dev_i_values.dot(dev_i_values))

    averages.append(avg_i)
    deviations.append(dev_i)

    s1 = SortedList()
    for j in range(M):
        if j != i:
            users_j = movie2user[j]
            users_j_set = set(users_j)
            common_users = (users_i_set & users_j_set)
            # There should be the common movies list of atleast limit i.e. 5 in this case
            if len(common_users) > limit:

                ratings_j = {user: usermovie2rating[(user,j)] for user in users_j}
                avg_j = np.mean(list(ratings_j.values()))
                dev_j = {user : (rating - avg_i) for user, rating in ratings_j.items() }
                dev_j_values = np.array(list(dev_j.values()))
                sigma_j = np.sqrt(dev_j_values.dot(dev_j_values))

            numerator = sum(dev_i[m] * dev_j[m] for m in common_users)
            w_ij = numerator/ (sigma_i * sigma_j)

            s1.add((-w_ij,j))
            if len(s1) > K:
                del s1[-1]
    print(f'Trained for {i+1}/{M}')
    neighbours.append(s1)

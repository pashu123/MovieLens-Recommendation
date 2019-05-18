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

# no. of common neighbour to consider
K = 25
# no. of common movies between the neighbours
limit = 5 
neighbours = [] #store neighbours in a list
averages = [] # each user's average rating for later use
deviations = [] # each user's deviation for later use

print('all the preprocessing done')

for i in range(N):
    # find the 25 closest user to user_i
    movies_i = user2movie[i]
    movies_i_set = set(movies_i)

    # lets calculate the average and deviation
    ratings_i = {movie: usermovie2rating[(i,movie)] for movie in movies_i}
    avg_i = np.mean(list(ratings_i.values()))
    dev_i = {movie : (rating - avg_i) for movie, rating in ratings_i.items()}
    dev_i_values = np.array(list(dev_i.values()))
    sigma_i = np.sqrt(dev_i_values.dot(dev_i_values))

    averages.append(avg_i)
    deviations.append(dev_i)

    s1 = SortedList()
    for j in range(N):
        if j != i:
            movies_j = user2movie[j]
            movies_j_set = set(movies_j)
            common_movies = (movies_i_set & movies_j_set)
            # There should be the common movies list of atleast limit i.e. 5 in this case
            if len(common_movies) > limit:

                ratings_j = {movie : usermovie2rating[(j,movie)] for movie in movies_j}
                avg_j = np.mean(list(ratings_j.values()))
                dev_j = { movie:(rating - avg_j) for movie, rating in ratings_j.items() }
                dev_j_values = np.array(list(dev_j.values()))
                sigma_j = np.sqrt(dev_j_values.dot(dev_j_values))

            numerator = sum(dev_i[m] * dev_j[m] for m in common_movies)
            w_ij = numerator/ (sigma_i * sigma_j)

            s1.add((-w_ij,j))
            if len(s1) > K:
                del s1[-1]
    print(f'Trained for {i}/{N}')
    neighbours.append(s1)


def predict(i,m):
    # calculate the weighted sum of deviations
    numerator = 0
    denominator = 0
    for neg_w,j in neighbours[i]:
        # remember the weight is stored as its negative
        # so the negative of the negative weight is the positive weight
        try:
            numerator += -neg_w * deviations[j][m]
            denominator += abs(neg_w)
        except KeyError:
            # neighbour may not have rated the same movie
            # don't want to do dictionary lookup twice
            pass
        
    if denominator == 0:
        prediction = averages[i]
    else:
        prediction = numerator / denominator + averages[i]
    prediction = min(5, prediction)
    prediction = max(0.5,prediction)

    return prediction
            
        

## lets predict
train_predictions = []
train_targets = []
for (i, m), target in usermovie2rating.items():
  # calculate the prediction for this movie
  prediction = predict(i, m)

  # save the prediction and target
  train_predictions.append(prediction)
  train_targets.append(target)

test_predictions = []
test_targets = []
# same thing for test set
for (i, m), target in usermovie2rating_test.items():
  # calculate the prediction for this movie
  prediction = predict(i, m)

  # save the prediction and target
  test_predictions.append(prediction)
  test_targets.append(target)


# calculate accuracy
def mse(p, t):
  p = np.array(p)
  t = np.array(t)
  return np.mean((p - t)**2)

print('train mse:', mse(train_predictions, train_targets))
print('test mse:', mse(test_predictions, test_targets))


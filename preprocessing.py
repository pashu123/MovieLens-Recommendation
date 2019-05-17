import pandas as pd

## Read the large 20m kaggle dataset
df = pd.read_csv('dataset/rating.csv')



## Some knowledge of the dataset ##

# user ids are ordered sequentially from 1 .. 138493
# with no missing numbers
# movie ids are integers from 1 .. 131262
# not all movie ids appear
# there are only 26744 movie ids
# write code to check it yourself

# make the userid go from 0 ... N-1
df.userId = df.userId - 1

# create a mapping for movie_ids
unique_movie_ids = set(df.movieId.values)
movie2idx = {}
count = 0
for movie_id in unique_movie_ids:
    movie2idx[movie_id] = count
    count += 1
    
print('done')

df['movie_idx'] = df.apply(lambda row: movie2idx[row.movieId], axis = 1)

df = df.drop(columns = ['timestamp'])

df.to_csv('dataset/preprocessed.csv')
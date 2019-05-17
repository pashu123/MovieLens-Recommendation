import pickle
import numpy as np
import pandas as pd
from collections import Counter


df = pd.read_csv('dataset/preprocessed.csv')
print('Original dataframe size:',len(df))


N = df.userId.max() + 1 #number of users
M = df.movie_idx.max() + 1 # number of movies


user_ids_count = Counter(df.userId)
movie_ids_count = Counter(df.movie_idx)

n = 10000
m = 2000

user_ids = [u for u,c in user_ids_count.most_common(n)]
movie_ids = [m for m,c in movie_ids_count.most_common(m)]

df_small = df[df.userId.isin(user_ids) & df.movie_idx.isin(movie_ids)].copy()


# need to remake user ids and movie ids since they no longer make sequential

new_user_id_map = {}
i = 0
for old in user_ids:
    new_user_id_map[old] = i
    i += 1
print(f'i : {i}')

new_movie_id_map = {}
j = 0
for old in movie_ids:
    new_movie_id_map[old] = j
    j += 1
print(f'j : {j}')

print('setting new ids')

df_small.loc[:,'userId'] = df_small.apply(lambda row: new_user_id_map[row.userId],axis = 1)

df_small.loc[:,'movie_idx'] = df_small.apply(lambda row: new_movie_id_map[row.movie_idx],axis = 1)


df_small.to_csv('dataset/very_small_rating.csv')
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

fist = pd.read_csv('/tmp/scoreDataWithItems.csv')
second = pd.read_csv('/tmp/scoreDataLastModelWithItems.csv')
all_data_frames = [fist, second]
all_data = result = pd.concat(all_data_frames)

all_data_map = {}
for k in range(1, 21):
    for idx in range(0, 8):
        k_alone = all_data.loc[all_data.k == k]
        k_idx = k_alone.loc[k_alone.currFeatIdx == idx]
        all_data_map[(k, idx)] = k_idx

user_movies = {}
user_movies_dictionary_path = "dictionary_data/user_movies.npy"
user_movies = np.load(user_movies_dictionary_path, allow_pickle=True).item()
num_users = len(user_movies)


# Index(['Unnamed: 0', 'k', 'currFeatIdx', 'userId', 'movieId',
#        'predicted rating', 'actual rating', 'item-item'],
#       dtype='object')
all_tables = []
for (k, indx) in all_data_map:
    print("starting:", k, ", ", indx) 
    table = all_data_map[(k, indx)]
    x_train = []
    y_train = []
    x_test = []
    for index, row in table.iterrows():
        user_id = row['userId']
        movie_idx = row['movieId']
        x_item = [row['item-item'], row['predicted rating']]
        x_test.append(x_item)
        if movie_idx in user_movies[user_id][:k]:
            x_train.append(x_item)
            y_train.append(row['actual rating'])
    reg = LinearRegression().fit(x_train, y_train)
    hybrid_rating = reg.predict(x_test)
    table["hybrid"] = hybrid_rating
    all_tables.append(table)

print(all_data_map[(1, 1)].columns)
all_tables_data_frames = pd.concat(all_tables)
all_tables_data_frames.to_csv('/tmp/all_table.csv')
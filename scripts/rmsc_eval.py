import pandas as pd
import math 

import numpy as np
def calcRMSE(reverseSortedMovieIds, actualRankingOfMovies):
    normalizeVal = 1.0 / len(reverseSortedMovieIds)
    squaredDiff = 0.0
    for idx, movieId in enumerate(actualRankingOfMovies):
        rank = idx + 1
        predictedRank = reverseSortedMovieIds.index(movieId) + 1
        squaredDiff += (predictedRank - rank) ** 2
    return np.sqrt(normalizeVal * squaredDiff)
print("loading table")
all_tables = pd.read_csv('/tmp/all_table.csv')
all_data_map = {}
print("loading map")
for k in range(1, 21):
    for idx in range(0, 8):
        k_alone = all_tables.loc[all_tables.k == k]
        k_idx = k_alone.loc[k_alone.currFeatIdx == idx]
        all_data_map[(k, idx)] = k_idx
print("sorting CF")
for (k, idx) in all_data_map:
    table = all_data_map[(k, idx)]
    table_sorted = table.sort_values(by=['userId', 'hybrid'], ascending=False)
    all_data_map[(k, idx)] = table_sorted
print("sorting actual")
sort_by_actual_rating = {}
for (k, idx) in all_data_map:
    table = all_data_map[(k, idx)]
    table_sorted = table.sort_values(by=['userId', 'actual rating'], ascending=False)
    sort_by_actual_rating[(k, idx)] = table_sorted
print("starting rmsc")
rmsc_map = {}
for (k, idx) in all_data_map:
    actual_rating_table =  sort_by_actual_rating[(k, idx)]
    hybrid_rating_table =  all_data_map[(k, idx)]
    all_users = all_data_map[(1, 1)].userId.unique()
    rmsc_all = 0
    all_users_size = len(all_users)
    for userId in all_users:
        actual = list(actual_rating_table[actual_rating_table['userId']==userId]['movieId'])
        hybrid = list(hybrid_rating_table[hybrid_rating_table['userId']==userId]['movieId'])
        rmsc = calcRMSE(hybrid, actual)
        rmsc_all += rmsc
    rmsc_map[(k, idx)] = rmsc_all / all_users_size
    print(k, " ", idx, " ", rmsc_all / all_users_size)
np.save('/tmp/hybridrmsc.npy', rmsc_map) 
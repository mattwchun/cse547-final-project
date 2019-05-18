'''
Overview:
Using MovieLens database, found the most common tags, i.e. tags that appear
most frequently when people tag movies.



find the top k tags, for a given tag how many movies does it show in
    move movies with a tag the higher its rank

for each movie in the data set,
    build a feature vector for it
        binary does it have that tag or not
        k1, k2, k3,.... tags make up this vector
        do it for every single movie

put this into a numpy matrix, k by number of movies

k = 5, 10, 20

write to disk, put it into numpy

'''


import pandas as pd
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)



# count tags and output to csv
def print_top_k_tags():
    df = pd.read_csv('tag.csv')

    # Debugging prints
    # print(df.head())
    # print(len(df['tag']))
    # print(len(df['tag'].unique()))
    #
    # print(df.count())
    #
    # print()

    for top_k in [5, 10, 15, 30, 50, 100]:  ## change k if you want
        df.groupby('tag').size().nlargest(top_k).to_csv("top_" + str(top_k))
        print(df.groupby('tag').size().nlargest(top_k))



########
# Build feature vector for each movie based on tags
def output_tag_feature_vectors():
    for top_k in ['top_5', 'top_10', 'top_15', 'top_30', 'top_50', 'top_100']: ## change this if u change k
        # df_ = pandas dataframe
        df_movie = pd.read_csv('movie.csv')
        df_tags = pd.read_csv('tag.csv')
        df_top_k_tags = pd.read_csv(top_k, names=['tag', 'count'])

        K = df_top_k_tags.shape[0]
        M = df_movie.shape[0]

        tags = df_top_k_tags['tag']

        # filter out non top k tags
        df_tags = df_tags[df_tags.tag.isin(tags)]


        matrix = np.ndarray(shape=(M, K))
        for i, movieId in enumerate(df_movie['movieId']):
            df_tags_for_this_movieId = df_tags[df_tags['movieId'] == movieId]

            for j in range(K):
                # print(tags[i])
                if tags[j] in df_tags_for_this_movieId['tag'].unique():
                    matrix[i,j] = 1

        print()
        matrix = np.rint(matrix)
        print(matrix)

        np.save("feature_vector_" + top_k, matrix)




output_tag_feature_vectors()


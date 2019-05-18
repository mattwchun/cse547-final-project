import sys
import csv
import sys
import csv
from datetime import datetime
import numpy as np

def calcQ(ratings):
    diagVals = np.sum(ratings, axis=0)
    diagMatrix = np.diag(diagVals)
    return diagMatrix

def iiCF(ratings):
    r = ratings
    q = calcQ(r)

    # gamma = R Q-1/2 RT R Q-1/2
    qToPow = np.power(q, -0.5, where=q!=0)
    gamma = r @ qToPow @ r.T @ r @ qToPow
    return gamma

data_file = "/Users/studentuser/cse547-final-project/rating.csv"
user_movies = {} # maps the user id to the np array that contains the movie id, rating and timestamp
with open(data_file) as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    header = next(reader)
    print(header)
    for row in reader:
        (userId, movieId, rating, timestamp) = row
        (userId, movieId, rating, timestamp) = (int(userId), int(movieId), float(rating), datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S'))
        matrix = user_movies.get(userId, [])
        matrix.append([movieId, rating, timestamp])
        user_movies[userId] = matrix
print("len:", len(user_movies))
num_users = len(user_movies)


for id in user_movies:
    m = user_movies[id]
    m.sort(key=lambda x: x[2], reverse=True)
    m = np.array(m)
    user_movies[id] = m
np.save('user_movies_jupyther.npy', user_movies) 

movie_file = "/Users/studentuser/cse547-final-project/movie.csv"
movieId_to_movieIndex = {} # map the movie id to a row index for R matrix
movieIndex_to_movieId = {} # the reverse of the above map
with open(movie_file) as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    header = next(reader)
    print(header)
    index = 0
    for row in reader:
        (movieId, title, genere) = row
        movieId = int(movieId)
        movieId_to_movieIndex[movieId] = index
        movieIndex_to_movieId[index] = movieId
        index += 1


# calculating the matrix based on k
k = 1
# #(movie , user) 
movie_user_matrix_for_k = np.zeros((num_movies, num_users)) # each column represents a user and each row a movie 
for user_id in user_movies:
    user_ratings_sorted = user_movies[user_id]
    user_column = user_id - 1
    for index in range(k):
        movieId_to_rating = user_ratings_sorted[index]
        movieId = movieId_to_rating[0]
        movie_row_index = movieId_to_movieIndex[movieId]
        rating = movieId_to_rating[1]
        movie_user_matrix_for_k[movie_row_index][user_column] = rating


gamma = iiCF(movie_user_matrix_for_k.T)


# All the commented code is for importing npy so we dont have to calculate from scratch


# user_movies_dictionary_path = "dictionary_data/user_movies.npy"
# movieId_to_movieIndex_dictionary_path = "dictionary_data/movieId_to_movieIndex.npy"
# movieIndex_to_movieId_dictionary_path = "dictionary_data/movieIndex_to_movieId.npy"

# user_movies = {}
# user_movies = np.load(user_movies_dictionary_path, allow_pickle=True).item()
# print("len:", len(user_movies))
# num_users = len(user_movies)

# movieId_to_movieIndex = {}
# movieIndex_to_movieId = {}

# movieId_to_movieIndex  = np.load(movieId_to_movieIndex_dictionary_path, allow_pickle=True).item()
# movieIndex_to_movieId  = np.load(movieIndex_to_movieId_dictionary_path, allow_pickle=True).item()

# num_movies = len(movieId_to_movieIndex)
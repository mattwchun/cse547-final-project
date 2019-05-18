import sys
import csv
import sys
import csv
import numpy as np
from datetime import datetime
data_file = sys.argv[1]
user_movies = {} # maps the user id to the np array that contains the movie id, rating and timestamp
with open(data_file) as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    header = next(reader)
    print(header)
#     rows = [[int(row[0]), int(row[1]), float(row[2]), datetime.strptime(row[3], '%Y-%m-%d %H:%M:%S')] for row in reader]
    for row in reader:
        (userId, movieId, rating, timestamp) = row
        (userId, movieId, rating, timestamp) = (int(userId), int(movieId), float(rating), datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S'))
        matrix = user_movies.get(userId, [])
        matrix.append([movieId, rating, timestamp])
        user_movies[userId] = matrix
print("len:", len(user_movies))

# sort the user's rated movies accoridng to the time stamp
for id in user_movies:
    m = user_movies[id]
    m.sort(key=lambda x: x[2], reverse=True)
    m = np.array(m)
    user_movies[id] = m
# np.save('user_movies.npy', user_movies) 
A = np.array([[1, 2, 3]]) # A is the feature vectors for a given movie. the index of the element is the movie id

k = 2 # configurable k
user_optimal = {} # map the user id to the optimal movie vector
for user_id in user_movies:
    print("user:", user_id)
    top_k_movies = user_movies[user_id][:k]
    rating_sum = 0
    optimal = np.zeros((A[0].shape)) 
    # calculate the optimal vector by summing up all k of the movie feature vectors from A
    # multiplies each of the movie by the rating and divide the final vector by the sum of all the ratings
    for m in top_k_movies:
        movie_id = m[0]
        rating = m[1]
        rating_sum += rating
        optimal += A[0] * rating # change it to when A is finished A[movie_id] * rating
    optimal = optimal * 1 / rating_sum
    user_optimal[user_id] = optimal


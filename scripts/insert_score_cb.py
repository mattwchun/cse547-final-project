import pandas as pd
import numpy as np
import scipy.sparse
def gamme_name(k):
    return "sci_gamma_" + str(k) + ".npz"

def get_score(k, user_id, movie_id):
    global current_k, gamma, user_to_movies_rating
    if not k == current_k:
        print("new k:", k)
        current_k = k
        gamma = None
        gamma = scipy.sparse.load_npz(gamme_name(k))
        user_to_movies_rating = {}
    if user_id not in user_to_movies_rating:
        ratings = gamma[user_id - 1].todense()
        user_to_movies_rating[user_id] = ratings
    return user_to_movies_rating[user_id].item((0, movie_id - 1))

scoreData = pd.read_csv('scoresDataLastModel.csv', header = None)
scoreData.columns = ["k", "currFeatIdx", "userId", "movieId", "predicted rating", "actual rating"]
print("sorting")
scoreData = scoreData.sort_values(by=['k', 'userId'])
print("finished sorting")
movieId_to_movieIndex_dictionary_path = "dictionary_data/movieId_to_movieIndex.npy"
movieId_to_movieIndex = {}
movieId_to_movieIndex  = np.load(movieId_to_movieIndex_dictionary_path, allow_pickle=True).item()
current_k = None
gamma = None
user_to_movies_rating = {} # map from user to movie ratings
scores = []
for index, row in scoreData.iterrows():
    k, userId, movieId = int(row['k']), int(row['userId']), int(row['movieId'])
    movie_index = movieId_to_movieIndex[movieId]
    s = get_score(k, userId, movie_index)
    scores.append(s)

scoreData["item-item"] = scores

scoreData.to_csv('scoreDataLastModelWithItems.csv')

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import csv


# In[2]:


# parameters
k = 20


# In[3]:


# load data
ratings = pd.read_csv("../data/first20Ratings.csv")
ratings.timestamp = pd.to_datetime(ratings.timestamp, infer_datetime_format=True)
allUserIds = np.genfromtxt('../data/validUsers.csv', delimiter=',') # users we care about
movies = pd.read_csv('../data/movie.csv')
idxToMovieIds = movies.movieId.tolist() # gives the mapping from index of feature_vector_top_k.npy to movieId
movieIdsToIdx = dict()
for idx, movieId in enumerate(idxToMovieIds):
    movieIdsToIdx[movieId] = idx

baseline_feat_files = ['../feature vector construction/feature vectors/feature_vector_top_5.npy',
                      '../feature vector construction/feature vectors/feature_vector_top_10.npy',
                      '../feature vector construction/feature vectors/feature_vector_top_15.npy',
                      '../feature vector construction/feature vectors/feature_vector_top_30.npy',
                      '../feature vector construction/feature vectors/feature_vector_top_50.npy',
                      '../feature vector construction/feature vectors/feature_vector_top_100.npy']
# baseline_feat_files = baseline_feat_files[:1] # test first case
feats = []
for baseline_feat_file in baseline_feat_files:
    currFeat = np.load(baseline_feat_file)
    feats.append(currFeat)


# In[4]:


# Helpers
def cos_sim(vA, vB):
    aSquared = np.sqrt(np.dot(vA,vA))
    bSquared = np.sqrt(np.dot(vB,vB))
    if aSquared == 0 or bSquared == 0:
        return 0.0

    return np.dot(vA, vB) / (aSquared * bSquared)

def getActualRankingOfMovies(ratingsForUserDF):
    sortedDF = ratingsForUserDF.sort_values(by=['rating', 'timestamp'], ascending=[False, True])
    return sortedDF.movieId.values

def calcRMSE(reverseSortedMovieIds, actualRankingOfMovies):
    normalizeVal = 1.0 / len(reverseSortedMovieIds)
    squaredDiff = 0.0
    for idx, movieId in enumerate(actualRankingOfMovies):
        rank = idx + 1
        predictedRank = reverseSortedMovieIds.index(movieId) + 1
        squaredDiff += (predictedRank - rank) ** 2
    return np.sqrt(normalizeVal * squaredDiff)

def optimal_movie_vec(k, currFeat, ratingsForUserDF):
    featureVecSize = len(currFeat[0])
    optimal_vec = np.zeros(featureVecSize)
    sumOfRatings = 0.0
    for index, row in ratingsForUserDF.iterrows():
        if index < k:
            currRating = row.rating
            currMovieId = row.movieId
            currFeatVecIdx = movieIdsToIdx[currMovieId]
            featVec = currFeat[currFeatVecIdx]

            optimal_vec = np.add(optimal_vec, currRating * currFeatVecIdx)
            sumOfRatings += currRating

    if sumOfRatings == 0:
        return np.zeros(featureVecSize)

    normalization = 1.0 / sumOfRatings
    return normalization * optimal_vec

def createScoreDataRow(currK, currFeatIdx, userId, ratingsForUserDF, predictedScores):
    # assumes predictedScores is in same order as movieIds appear in ratingsForUserDF
    result = []
    idx = 0
    for dfIdx, row in ratingsForUserDF.iterrows():
        result.append([currK, currFeatIdx, userId, row.movieId, predictedScores[idx], row.rating])
        idx += 1
    return result

def outputData(data, outputFilename):
    with open(outputFilename, "w") as f:
        writer = csv.writer(f)
        writer.writerows(data)

# In[ ]:


# eval
def eval():
    print(("currK", "currFeatIdx"))
    data = [] # [k, userId, currFeatIdx, RMSE]
    scoresData = [] # [k, currFeatIdx, userId, movieId, predicted rating, actual rating]
    for currFeatIdx, currFeat in enumerate(feats):
        for currKMinus1 in range(k):
            currK = currKMinus1 + 1
            totalRMSEUsers = 0.0
            for currUserId in allUserIds:
                # data frame with only userId = currUserId
                ratingsForUser = ratings[ratings.userId == currUserId]

                # calculate optimal movie vec for currUserId
                optimal_movie_vector = optimal_movie_vec(currK, currFeat, ratingsForUser)

                # true rankings for movies
                trueRankingsOfMovies = getActualRankingOfMovies(ratingsForUser)

                # id in currFeat for each of the movieIds
                featVecIdxs = np.array([movieIdsToIdx[movieId] for movieId in ratingsForUser.movieId.values])

                # calculate scores for each movieId
                scores = np.array([cos_sim(optimal_movie_vector, currFeat[featVecIdx]) for featVecIdx in featVecIdxs])
                reverseSortedScoreIdxs = np.flip(np.argsort(scores))

                # get reverse sorted feat vec idx sorted on score
                reverseSortedFeatVecIdxs = featVecIdxs[reverseSortedScoreIdxs]

                # get reverse sorted movie ids sorted on score
                reverseSortedMovieIds = [idxToMovieIds[idx] for idx in reverseSortedFeatVecIdxs.tolist()]

                # calc RMSE
                currRMSE = calcRMSE(reverseSortedMovieIds, trueRankingsOfMovies)
                totalRMSEUsers += currRMSE

                # collect the data
                data.append([currK, currUserId, currFeatIdx, currRMSE])
                scoresData += createScoreDataRow(currK, currFeatIdx, currUserId, ratingsForUser, scores)

            avgRMSE = totalRMSEUsers / len(allUserIds)
            print((currK, currFeatIdx), "avgRMSE =", avgRMSE)
    return (data, scoresData)



# In[ ]:


def main():
    data, scoresData = eval()

    # output RMSE data
    outputData(data, 'rmseData.csv')

    # output scores data
    outputData(scoresData, 'scoresData.csv')

# In[ ]:


main()


# In[ ]:

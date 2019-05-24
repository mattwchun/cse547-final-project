#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import csv

# parameters
k = 20


# Helpers
# returns numpy array of the same size as baseline vector matrices but where dont have data for movie is a vector of all -1's
def generateFeatVecMatrixFromEmbedding(embeddings, allMovieIds, movieIdToEmbeddingIdx):
    result = []
    dimensions = len(embeddings[0])
    for movieId in allMovieIds:
        if movieId in movieIdToEmbeddingIdx:
            idxOfEmbedding = movieIdToEmbeddingIdx[movieId]
            result.append(embeddings[idxOfEmbedding])
        else:
            result.append([-1.0] * dimensions)
    return np.array(result)


def cos_sim(vA, vB):
    # print(vA, vB)
    aSquared = np.sqrt(np.dot(vA,vA))
    bSquared = np.sqrt(np.dot(vB,vB))
    result = 0.0

    if aSquared != 0 and bSquared != 0:
        result = np.dot(vA, vB) / (aSquared * bSquared)

    return result

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
    currK = 0
    for index, row in ratingsForUserDF.iterrows():
        if currK < k:
            currRating = row.rating
            currMovieId = row.movieId
            currFeatVecIdx = movieIdsToIdx[currMovieId]
            featVec = currFeat[currFeatVecIdx]

            optimal_vec = np.add(optimal_vec, currRating * featVec)
            sumOfRatings += currRating
        currK += 1

    if sumOfRatings == 0:
        return np.zeros(featureVecSize)

    normalization = 1.0 / sumOfRatings
    result = normalization * optimal_vec
    return result

def createScoreDataRow(currK, currFeatIdx, userId, ratingsForUserDF, predictedScores):
    # assumes predictedScores is in same order as movieIds appear in ratingsForUserDF
    result = []
    idx = 0
    for dfIdx, row in ratingsForUserDF.iterrows():
        result.append([currK, currFeatIdx, userId, row.movieId, predictedScores[idx], row.rating])
        idx += 1
    return result

def calcILS(currFeat, reverseSortedIdxs):
    n = len(reverseSortedIdxs)
    total = 0.0
    for i in range(n - 1):
        idx1 = reverseSortedIdxs[i]
        feat1 = currFeat[idx1]
        for j in range(i, n):
            idx2 = reverseSortedIdxs[j]
            feat2 = currFeat[idx2]
            total += cos_sim(feat1, feat2)
    return total / 2.0


def outputData(data, outputFilename):
    with open(outputFilename, "w") as f:
        writer = csv.writer(f)
        writer.writerows(data)

def runEvalRMSE():
    data, scoresData, avgRMSEData = evalRMSE()

    # output RMSE data
    outputData(data, 'rmseData.csv')

    # output scores data
    outputData(scoresData, 'scoresData.csv')

    # output avgRMSE data
    outputData(avgRMSEData, 'avgRMSEData.csv')

def runEvalILS():
    avgILSData, ilsData = evalDiversity()

    # output avgILS data
    outputData(avgILSData, 'avgILSData.csv')

    # output ILS data
    outputData(ilsData, 'ilsData.csv')


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

autoencoder_feat_files = ['../feature vector construction/feature vectors/autoencoder_feat_vec_32_dim.npy']
# baseline_feat_files = baseline_feat_files[:1] # test first case
feats = []
for feat_file in baseline_feat_files + autoencoder_feat_files:
    currFeat = np.load(feat_file)
    feats.append(currFeat)


# eval
def evalRMSE():
    print("evalRMSE()")
    print(("currK", "currFeatIdx"))
    data = [] # [k, userId, currFeatIdx, RMSE]
    avgRMSEData = [] # [k, currFeatIdx, avgRMSE]
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
                reverseSortedScoreIdxs = np.argsort(scores)[::-1]

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

            currAvgRMSEDataRow = [currK, currFeatIdx, avgRMSE]
            print(currAvgRMSEDataRow)
            avgRMSEData.append(currAvgRMSEDataRow)
    return (data, scoresData, avgRMSEData)

# finds top 20 movies and calculates the diversity within those top 20 movies
def evalDiversity(topNMovies = 20):
    print("evalDiversity(topNMovies = 20)")
    avgILSData = [] # [k, currFeatIdx, avgILS]
    ilsData = [] # [k, currFeatIdx, userId, ILS]
    for currFeatIdx, currFeat in enumerate(feats):
        for currKMinus1 in range(k):
            currK = currKMinus1 + 1
            totalILSUsers = 0.0
            for currUserId in allUserIds:
                # data frame with only userId = currUserId
                ratingsForUser = ratings[ratings.userId == currUserId]

                # calculate optimal movie vec for currUserId
                optimal_movie_vector = optimal_movie_vec(currK, currFeat, ratingsForUser)

                # get scores for all movies
                scores = np.array([cos_sim(optimal_movie_vector, currFeat[movieIdsToIdx[movieId]]) for movieId in idxToMovieIds])

                # sort
                reverseSortedScoreIdxs = np.argsort(scores)[::-1]
                topNMoviesSortedScoreIdxs = reverseSortedScoreIdxs[:topNMovies]

                # get sorted order of movieIds
                reverseSortedMovieIds = [idxToMovieIds[idx] for idx in topNMoviesSortedScoreIdxs]

                # get sorted idx of indexes of feat vector
                reverseSortedIdxs = [movieIdsToIdx[movieId] for movieId in reverseSortedMovieIds]

                currUserILS = calcILS(currFeat, reverseSortedIdxs)

                totalILSUsers += currUserILS

                ilsData.append([currK, currFeatIdx, currUserId, currUserILS])

            currAvgILS = 1.0 * totalILSUsers / len(allUserIds)

            currILSDataElement = [currK, currFeatIdx, currAvgILS]
            print(currILSDataElement)
            avgILSData.append(currILSDataElement)

    return (avgILSData, ilsData)

# In[ ]:


def main():
    runEvalRMSE()
    runEvalILS()

# In[ ]:


main()


# In[ ]:


import numpy as np
import pandas as pd
import csv

# parameters


# load data


def accuracy(RMSE):
    allModelIdx = RMSE[1]
    allUsers = RMSE[2]
    for modelIdx in allModelIdx:
        for k in range(1, 21):
            totalRMSE = 0.0
            for user in allUsers:
                reverseSortedMovieIdActualRating = sortActualRating(user, k, modelIdx)
                reverseSortedMovieIdHybridRating = sortHybridRating(user, k, modelIdx)
                currRMSE = calcRMSE(reverseSortedMovieIdActualRating, reverseSortedMovieIdHybridRating)

            totalRMSE += currRMSE
            avgRMSE = totalRMSE / len(allUsers)
            saveToOutputFile([k, modelIdx, avgRMSE])




def calcRMSE(reverseSortedMovieIds, actualRankingOfMovies):
    normalizeVal = 1.0 / len(reverseSortedMovieIds)
    squaredDiff = 0.0
    for idx, movieId in enumerate(actualRankingOfMovies):
        rank = idx + 1
        predictedRank = reverseSortedMovieIds.index(movieId) + 1
        squaredDiff += (predictedRank - rank) ** 2
    return np.sqrt(normalizeVal * squaredDiff)



def sortActualRating(user, k, modelIdx):



def sortHybridRating(user, k , modelIdx):
    def outputData(data, outputFilename):
        with open(outputFilename, "w") as f:
            writer = csv.writer(f)
            writer.writerows(data)


def main():
    #runEvalRMSE()

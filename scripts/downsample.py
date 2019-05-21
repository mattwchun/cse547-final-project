# Generates the training sets with fixed number of images for each trailer
# Takes following cmd line args:
# 1) current data folder
# 2) output data folder
# 3) k = the number of images per folder
#
# If k >= the number of images then that folder isn't outputed

import os
import sys
import glob
from shutil import copyfile, copytree, copy
import random
from joblib import Parallel, delayed
from PIL import Image
import csv
# 1) for all folders in data folder
# 1) move folder to output folder
# 2) generate sample
# 3) delete non sample files
frame_index = 0
# returns sorted by increasing timestamp of subset of folder
def generateSample(dir, k):
    files = glob.glob("%s/*" % dir)
    for f in files:
        try:
            img = Image.open(f) # open the image file
            img.verify() # verify that it is, in fact an image
        except (IOError, SyntaxError) as e:
            print('Bad file:', f) # print out the names of corrupt files
            os.remove(f)
    files = glob.glob("%s/*" % dir)
    if k > len(files):
        # ignore
        return []

    files.sort(key=lambda x: os.path.getmtime(x))
    files = files[:-1]
    numFiles = len(files)

    random.shuffle(files)

    subset = files[:k]
    if len(subset) != k:
        return []

    return subset

def moveFolder(dir, currDataFolder, outputDir, k):
    global frame_index
    subset = generateSample(dir, k)
    isCopied = False
    if len(subset) == k:
        for f in subset:
            # print("copying:", f, "  index:", frame_index)
            # TODO: just need to change "frame_" to movie name
            copy(f, outputDir + "/" + str(frame_index) + ".jpg")
            frame_index += 1
        isCopied = True
    return (outputDir, subset, isCopied)

def run(idx, dir, numMovies, currDataFolder, outputFolder, k):
    # print("running")
    if idx % 500 == 0:
        print(idx, numMovies)

    if os.path.isfile(dir):
    	# print("returning.", str(dir))
    	return

    outDir, sample, isCopied = moveFolder(dir, currDataFolder, outputFolder, k)

    return isCopied



def main():
    currDataFolder = sys.argv[1]
    outputFolder = sys.argv[2]
    k = int(sys.argv[3])

    dirs = glob.glob("%s/*" % currDataFolder)
    numMovies = len(dirs)

    #Parallel(n_jobs=8)(delayed(run)(idx, dir, numMovies, currDataFolder, outputFolder, k) for idx, dir in enumerate(dirs))
    movieOrder = []
    numMovies = len(dirs)
    for idx, dir in enumerate(dirs):
        print(idx, "/", numMovies)
        try:
            isCopied = run(idx, dir, numMovies, currDataFolder, outputFolder, k)
            if isCopied:
                movieName = os.path.basename(dir)
                movieOrder.append(movieName)
        except:
            print("failed on idx:", idx)

    # output order of movies
    print(movieOrder)
    with open('movieOrder.csv', 'w') as f:
        wtr = csv.writer(f)
        for x in movieOrder:
            wtr.writerow([x])


main()

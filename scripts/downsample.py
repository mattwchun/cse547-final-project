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
from shutil import copyfile, copytree
import random

# 1) for all folders in data folder
# 1) move folder to output folder
# 2) generate sample
# 3) delete non sample files

# returns sorted by increasing timestamp of subset of folder
def generateSample(dir, k):
    files = glob.glob("%s/*" % dir)
    if k >= len(files):
        # ignore
        return []

    files.sort(key=lambda x: os.path.getmtime(x))
    files = files[:-1]
    numFiles = len(files)

    random.shuffle(files)

    subset = random[:k]
    subset.sort(key=lambda x: os.path.getmtime(x))
    return subset

# dir = movie1
# outputDir = output
def moveFolder(dir, outputDir, k):
    subset = generateSample(dir, k)
    outDir = dir.replace("data", outputDir)
    copytree(dir, outDir)
    return (outDir, subset)

def removeNonSamples(dir, sample):
    files = glob.glob("%s/*" % dir)
    for file in files:
        if file not in sample:
            os.remove(file)

def main():
    currDataFolder = sys.argv[1]
    outputFolder = sys.argv[2]
    k = sys.argv[3]

    dirs = glob.glob("%s/*" % currDataFolder)
    for dir in dirs:
        outDir, sample = moveFolder(dir, outputFolder, k)
        if len(sample) > 0:
            removeNonSamples(outDir, sample)

main()

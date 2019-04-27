import csv
import cv2
from pytube import YouTube
import os
import subprocess

def getCSV(filename):
    with open(filename) as csvfile:
        data = list(csv.reader(csvfile))

    return data[1:] # gets rid of header row

def downloadMP4(videoName, videoURL):
    yt = YouTube(videoURL)

    # make dir
    newDirName = 'data/%s' % videoName
    os.mkdir(newDirName)

    stream = yt.streams.get_by_itag('18')
    stream.download(newDirName)
    fileLocation = '%s/%s' % (newDirName, stream.default_filename)
    newFileLocation = '%s/%s' % (newDirName, 'v.mp4')

    # rename video file so easy to process
    os.rename(fileLocation, newFileLocation)
    return newFileLocation


def get_frame_types(video_fn):
    command = 'ffprobe -v error -show_entries frame=pict_type -of default=noprint_wrappers=1'.split()
    out = subprocess.check_output(command + [video_fn]).decode()
    frame_types = out.replace('pict_type=','').split()
    return zip(range(len(frame_types)), frame_types)

def save_i_keyframes(video_fn, outputDir):
    frame_types = get_frame_types(video_fn)
    i_frames = [x[0] for x in frame_types if x[1]=='I']
    if i_frames:
        basename = os.path.splitext(os.path.basename(video_fn))[0]
        cap = cv2.VideoCapture(video_fn)
        count = 0
        for frame_no in i_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            outname = outputDir + '/' + basename+'_i_frame_'+str(count)+'.jpg'
            cv2.imwrite(outname, frame)
            count += 1
        cap.release()
    else:
        print ('No I-frames in '+video_fn)

def main():
    os.chdir('..')
    data = getCSV('data/ml-youtube.csv')
    numVideos = len(data)
    count = 1
    for row in data:
        print('on video %d of %d' % (count, numVideos))
        youtubeURL = 'https://youtu.be/%s' % row[0]
        videoName = row[2].replace(' ', '')
        videoFileLocation = downloadMP4(videoName, youtubeURL)
        newDirName = 'data/%s' % videoName
        save_i_keyframes(videoFileLocation, newDirName)
        count += 1

main()

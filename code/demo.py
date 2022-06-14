import numpy as np
import cv2 as cv
import pybgs as bgs
import math
from scipy.signal import convolve
from FSdetection import flame_smoke_detection

def save_to_video(frames,save_name):
    fourcc = cv.VideoWriter_fourcc('M','J','P','G')
    out = cv.VideoWriter(save_name, fourcc, 30, frames[0].shape[:2][::-1])
    for i in range(len(frames)):
        out.write(frames[i])
    out.release()

if __name__ == '__main__':
    video_path = '../data/controlled2.avi'
    save_name = video_path.split('/')[-1].split('.')[0] + '_result.avi'
    cap = cv.VideoCapture(video_path)
    vid_fps = cap.get(cv.CAP_PROP_FPS)
    vid_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    vid_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))

    fs_detect = flame_smoke_detection()
    fs_detect.video_info_initialize(vid_height, vid_width, vid_fps)

    frames = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if( ret == False ):
            break
        
        # cv.imshow('frame', frame)
        frames.append(fs_detect.apply(frame))
            
        k = cv.waitKey(50) & 0xff
        if k == 27:
            cv.destroyAllWindows()
            break


    cap.release()
    cv.destroyAllWindows()
    save_to_video(frames, 'result/' + save_name)
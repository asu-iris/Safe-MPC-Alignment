import cv2
import os

videopath=os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),'Data','realworld_video')
videoname='video.MP4'
framepath=os.path.join(videopath,'frames')

video_handle=cv2.VideoCapture(os.path.join(videopath,videoname))
cnt=0
while True:
    ret,frame=video_handle.read()
    if not ret:
        break

    if cnt%60==0:
        framename='frame_'+str(cnt)+'.jpg'
        cv2.imwrite(os.path.join(framepath,framename),cv2.resize(frame,(0, 0),fx = 0.25, fy = 0.25))
    
    cnt+=1
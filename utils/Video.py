import numpy as np
import cv2
import os

class VideoMaker(object):
    def __init__(self,path,fps=12,size=(640,480)) -> None:
        self.path=path
        self.fps=fps
        self.size=size
        
    def process(self):
        f=open(os.path.join(self.path,'timestamps.txt'),'r')
        time_stamp_lines=f.readlines()
        n_imgs=len(time_stamp_lines)
        corrs=[]
        for i in range(n_imgs):
            corr=time_stamp_lines[i].split(' ')[-1].replace('\n','')
            corrs.append(corr)

        frames=[None]
        for i in range(1,n_imgs):
            filename=os.path.join(self.path,'mj_'+str(i)+'.jpg')
            if corrs[i-1] != 'reset':
                frames.append(cv2.imread(filename))
            
            else:
                frames.append(None)

        video=cv2.VideoWriter(os.path.join(self.path,'demo.avi'),cv2.VideoWriter_fourcc(*'MJPG'),self.fps,self.size)
        text_cnt=0
        text=None
        for i in range(n_imgs):
            if frames[i] is None:
                continue

            img=frames[i]
            
            if corrs[i]!='None':
                text=corrs[i]
                text_cnt=10

            if text_cnt>0:
                new_img=cv2.putText(img,text,(20,420),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255))
                video.write(new_img)
                text_cnt-=1
            
            else:
                video.write(img)
        
        video.release()

if __name__=='__main__':
    path=os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),'Data','test')
    vm=VideoMaker(path)
    vm.process()



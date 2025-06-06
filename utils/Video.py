import numpy as np
import cv2
import os

class VideoMaker(object):
    def __init__(self,path,fps=12,size=(640,480),cam_flag=False) -> None:
        self.path=path
        self.fps=fps
        self.size=size
        self.cam_flag=cam_flag
        
    def process(self):
        f=open(os.path.join(self.path,'timestamps.txt'),'r')
        time_stamp_lines=f.readlines()
        n_imgs=len(time_stamp_lines)
        corrs=[]
        for i in range(n_imgs):
            corr=time_stamp_lines[i].split(' ')[-1].replace('\n','')
            corrs.append(corr)
        # video for mj
        frames=[None]
        for i in range(1,n_imgs):
            filename=os.path.join(self.path,'mj_'+str(i)+'.jpg')
            if corrs[i-1] != 'reset':
                frames.append(cv2.imread(filename))
            
            else:
                frames.append(None)
            #frames.append(cv2.imread(filename))
        print('valid frames mj', sum(x is not None for x in frames))
        video=cv2.VideoWriter(os.path.join(self.path,'demo.avi'),cv2.VideoWriter_fourcc(*'MJPG'),self.fps,self.size)
        text_cnt=0
        text=None
        corr_num=0
        for i in range(n_imgs):
            if frames[i] is None:
                continue

            img=frames[i]
            
            if corrs[i]!='None':
                text=corrs[i]
                text_cnt=10
                if corrs[i]!='reset':
                    corr_num+=1

            heat_path=os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),'Data','uav_figs')
            heatmap = cv2.imread(os.path.join(heat_path,'heatmap_'+str(corr_num)+'.png'))
            #print(heatmap.shape)
            heatmap_small=cv2.resize(heatmap,(200,150))
            img[0:150,440:,:]=heatmap_small

            if text_cnt>0:
                new_img=cv2.putText(img,text,(20,420),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255))
                video.write(new_img)
                text_cnt-=1
            
            else:
                video.write(img)
        
        video.release()

        if not self.cam_flag:
            return
        # video for hand
        frames_hand=[None]
        for i in range(1,n_imgs):
            filename=os.path.join(self.path,'cam_'+str(i)+'.jpg')
            # if corrs[i-1] != 'reset':
            #     frames_hand.append(cv2.imread(filename))
            
            # else:
            #     frames_hand.append(None)
            frames_hand.append(cv2.imread(filename))
        print('valid frames cam', sum(x is not None for x in frames))
        video=cv2.VideoWriter(os.path.join(self.path,'hand.avi'),cv2.VideoWriter_fourcc(*'MJPG'),self.fps,self.size)
        for i in range(1,n_imgs):
            #print(i)
            if frames_hand[i] is None:
                continue
            video.write(frames_hand[i])

        video.release()

        # merge
        video=cv2.VideoWriter(os.path.join(self.path,'demo_merge.avi'),cv2.VideoWriter_fourcc(*'MJPG'),self.fps,self.size)
        text_cnt=0
        text=None
        corr_num=0
        for i in range(1,n_imgs-1):
            if frames[i] is None:
                continue

            img=frames[i]
            #print(img.shape)
            hand_img=cv2.resize(frames_hand[i+1],(200,150))
            #print(hand_img.shape)
            img[330:,440:,:]=hand_img

            heat_path=os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),'Data','uav_figs')
            heatmap = cv2.imread(os.path.join(heat_path,'heatmap_'+str(corr_num)+'.png'))
            #print('heatmap_'+str(corr_num)+'.png')
            #print('heatmap_'+str(corr_num)+'.png',heatmap.shape)
            heatmap_small=cv2.resize(heatmap,(200,150))
            img[0:150,440:,:]=heatmap_small

            if text_cnt>0:
                new_img=cv2.putText(img,text,(20,420),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255))
                video.write(new_img)
                text_cnt-=1
            
            else:
                video.write(img)

            if corrs[i]!='None':
                text=corrs[i]
                text_cnt=10
                if corrs[i]!='reset':
                    corr_num+=1

    def process_uav_sep(self):
        f=open(os.path.join(self.path,'timestamps.txt'),'r')
        time_stamp_lines=f.readlines()
        n_imgs=len(time_stamp_lines)
        corrs=[]
        for i in range(n_imgs):
            corr=time_stamp_lines[i].split(' ')[-1].replace('\n','')
            corrs.append(corr)
        # video for mj and heatmap
        frames=[None]
        aux_frames=[None]
        for i in range(1,n_imgs-1):
            filename=os.path.join(self.path,'mj_'+str(i)+'.jpg')
            filename_aux=os.path.join(self.path,'aux_'+str(i)+'.jpg')
            if corrs[i-1] != 'reset':
                frames.append(cv2.imread(filename))
                aux_frames.append(cv2.imread(filename_aux))
            else:
                frames.append(None)
                aux_frames.append(None)
            #frames.append(cv2.imread(filename))
        print('valid frames mj', sum(x is not None for x in frames))
        video=cv2.VideoWriter(os.path.join(self.path,'demo.avi'),cv2.VideoWriter_fourcc(*'MJPG'),self.fps,self.size)
        video_aux=cv2.VideoWriter(os.path.join(self.path,'demo_aux.avi'),cv2.VideoWriter_fourcc(*'MJPG'),self.fps,self.size)
        video_heat=cv2.VideoWriter(os.path.join(self.path,'demo_heat.avi'),cv2.VideoWriter_fourcc(*'MJPG'),self.fps,self.size)
        corr_num=0
        for i in range(1,n_imgs-1):
            if frames[i] is None:
                continue

            img=frames[i]
            img_aux=aux_frames[i]

            heat_path=os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),'Data','heatmap_2')
            heatmap = cv2.imread(os.path.join(heat_path,'heatmap_'+str(corr_num)+'.png'))
            #print(heatmap.shape)
            heatmap_small=cv2.resize(heatmap,self.size)
            video.write(img)
            video_aux.write(img_aux)
            video_heat.write(heatmap_small)

            if corrs[i]!='None':
                if corrs[i]!='reset':
                    corr_num+=1
        
        video.release()
        video_aux.release()
        video_heat.release()

        frames_hand=[None]
        for i in range(1,n_imgs-1):
            filename=os.path.join(self.path,'cam_'+str(i+1)+'.jpg')
            if corrs[i-1] != 'reset':
                frames_hand.append(cv2.imread(filename))
            
            else:
                frames_hand.append(None)
            #frames_hand.append(cv2.imread(filename))
        print('valid frames cam', sum(x is not None for x in frames_hand))
        video=cv2.VideoWriter(os.path.join(self.path,'hand.avi'),cv2.VideoWriter_fourcc(*'MJPG'),self.fps,self.size)
        for i in range(1,n_imgs-1):
            #print(i)
            if frames_hand[i] is None:
                continue
            video.write(frames_hand[i])

        video.release()

    def process_arm(self):
        f=open(os.path.join(self.path,'timestamps.txt'),'r')
        time_stamp_lines=f.readlines()
        n_imgs=len(time_stamp_lines)
        corrs=[]
        for i in range(n_imgs):
            corr=time_stamp_lines[i].split(' ')[-1].replace('\n','')
            corrs.append(corr)
        # video for mj
        frames=[None]
        for i in range(1,n_imgs):
            filename=os.path.join(self.path,'mj_cam1_'+str(i)+'.jpg')
            if corrs[i-1] != 'reset':
                frames.append(cv2.imread(filename))
            
            else:
                frames.append(None)
            #frames.append(cv2.imread(filename))
        print('valid frames mj', sum(x is not None for x in frames))
        video=cv2.VideoWriter(os.path.join(self.path,'demo.avi'),cv2.VideoWriter_fourcc(*'MJPG'),self.fps,self.size)
        text_cnt=0
        text=None
        text_map={'up':'up','down':'down','reset':'reset','right':'rotate_clock','left':'rotate_anticlock'}
        for i in range(n_imgs):
            if frames[i] is None:
                continue

            img=frames[i]
            
            if corrs[i]!='None':
                text=corrs[i]
                text_cnt=10

            if text_cnt>0:
                new_img=cv2.putText(img,text_map[text],(20,420),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255))
                video.write(new_img)
                text_cnt-=1
            
            else:
                video.write(img)
        
        video.release()

        if not self.cam_flag:
            return
        # video for hand
        frames_hand=[None]
        for i in range(1,n_imgs):
            filename=os.path.join(self.path,'cam_'+str(i)+'.jpg')
            # if corrs[i-1] != 'reset':
            #     frames_hand.append(cv2.imread(filename))
            
            # else:
            #     frames_hand.append(None)
            frames_hand.append(cv2.imread(filename))
        print('valid frames cam', sum(x is not None for x in frames))
        video=cv2.VideoWriter(os.path.join(self.path,'hand.avi'),cv2.VideoWriter_fourcc(*'MJPG'),self.fps,self.size)
        for i in range(1,n_imgs):
            #print(i)
            if frames_hand[i] is None:
                continue
            video.write(frames_hand[i])

        video.release()

        # merge
        video=cv2.VideoWriter(os.path.join(self.path,'demo_merge.avi'),cv2.VideoWriter_fourcc(*'MJPG'),self.fps,self.size)
        text_cnt=0
        text=None
        corr_num=0
        for i in range(1,n_imgs-1):
            if frames[i] is None:
                continue

            img=frames[i]
            #print(img.shape)
            hand_img=cv2.resize(frames_hand[i+1],(200,150))
            #print(hand_img.shape)
            img[330:,440:,:]=hand_img

            heat_path=os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),'Data','arm_figs')
            heatmap = cv2.imread(os.path.join(heat_path,'heatmap_'+str(corr_num)+'.png'))
            #print('heatmap_'+str(corr_num)+'.png')
            #print('heatmap_'+str(corr_num)+'.png',heatmap.shape)
            heatmap_small=cv2.resize(heatmap,(200,150))
            img[0:150,440:,:]=heatmap_small
            if text_cnt>0:
                new_img=cv2.putText(img,text_map[text],(20,420),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255))
                video.write(new_img)
                text_cnt-=1
            
            else:
                video.write(img)

            if corrs[i]!='None':
                text=corrs[i]
                text_cnt=10
                if corrs[i]!='reset':
                    corr_num+=1

            



if __name__=='__main__':
    path_uav=os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),'Data','uav_figs_video_2')
    vm_uav=VideoMaker(path_uav,cam_flag=True,fps=12)
    vm_uav.process_uav_sep()
    # path_uav=os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),'Data','test')
    # vm_arm=VideoMaker(path_uav,cam_flag=False)
    # vm_arm.process()



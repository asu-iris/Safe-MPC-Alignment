import logging
import sys,os
import numpy as np

class UserLogger(object):
    def __init__(self,user=0,trail=0,dir=os.path.abspath(os.getcwd())) -> None:
        self.filename='log_user_'+str(user)+'_trial_'+str(trail)+'.txt'
        self.dir=dir
        self.filepath=os.path.join(self.dir,self.filename)
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        files=os.listdir(self.dir)
        for file in files:
            os.remove(os.path.join(self.dir,file))

        self.dir=dir
        self.logger=logging.getLogger('HUMAN-STUDY-LOGGER')
        logging.basicConfig(filename=self.filepath,
                    format = '%(asctime)s - %(message)s',
                    level=logging.INFO,
                    filemode='w')
        
    def log_correction(self,msg):
        self.logger.info(msg)
        #logging.info(msg)

    def log_trajectory(self,trajectory,traj_id):
        np.save(os.path.join(self.dir,'trajectory_'+str(traj_id)+'.npy'),trajectory)

    def log_termination(self,flag,cnt_corr,weights):
        message=str(flag)+'_'+str(cnt_corr)
        self.logger.info(message)
        #logging.info(msg=message)
        np.save(os.path.join(self.dir,'weights.npy'),weights)
            

class Realtime_Logger(object):
    def __init__(self,user=0,trail=0,dir=os.path.abspath(os.getcwd())):
        self.filename='log_user_'+str(user)+'_trial_'+str(trail)+'.txt'
        self.dir=dir
        self.filepath=os.path.join(self.dir,self.filename)
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        open(self.filepath, 'w').close()

    def log_correction(self,msg):
        f=open(self.filepath,"a")
        f.write(msg)
        f.close()


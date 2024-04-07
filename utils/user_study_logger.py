import logging
import sys,os
import numpy as np

class UserLogger(object):
    def __init__(self,user=0,trail=0,dir=os.path.abspath(os.getcwd())) -> None:
        self.filename='log_user_'+str(user)+'_trail_'+str(trail)+'.txt'
        self.filepath=os.path.join(dir,self.filename)
        self.dir=dir
        self.logger=logging.getLogger('HUMAN-STUDY-LOGGER')
        logging.basicConfig(filename=self.filepath,
                    format = '%(asctime)s - %(message)s',
                    level=logging.INFO,
                    filemode='w')
        
    def log_correction(self,msg):
        logging.info(msg)

    def log_termination(self,flag,cnt_corr,weights):
        message=str(flag)+'_'+str(cnt_corr)
        logging.info(msg=message)
        np.save(os.path.join(self.dir,'weights.npy'),weights)



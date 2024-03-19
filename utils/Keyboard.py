import numpy as np
import time
from pynput import keyboard
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.abspath(os.getcwd()))

def uav_key_handler(key,pause,msg):
    if key==keyboard.Key.space:
        pause[0]=not pause[0]
        time.sleep(0.4)
        return
    else:
        pass

    if not pause[0]:
        if key==keyboard.Key.enter:
            print('emergency stop')
            msg[0]='reset'
        if key==keyboard.Key.up:
            #print('setting up corr')
            msg[0]='up'
        elif key==keyboard.Key.down:
            #print('setting down corr')
            msg[0]='down'
        elif key==keyboard.Key.left:
            #print('setting left corr')
            msg[0]='left'
        elif key==keyboard.Key.right:
            #print('setting right corr') 
            msg[0]='right'
        elif key==keyboard.Key.esc:
            print('satified')
            msg[0]='quit'
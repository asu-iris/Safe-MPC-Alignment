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
        elif key==keyboard.Key.esc:
            print('satified')
            msg[0]='quit'

        if key==keyboard.Key.up:
        #if key.char == 'w':
            #print('setting up corr')
            msg[0]='up'
        elif key==keyboard.Key.down:
        #elif key.char == 's':
            #print('setting down corr')
            msg[0]='down'
        elif key==keyboard.Key.left:
        #elif key.char == 'a':
            #print('setting left corr')
            msg[0]='left'
        elif key==keyboard.Key.right:
        #elif key.char == 'd':
            #print('setting right corr') 
            msg[0]='right'
        

def key_interface(MSG):
    human_corr=None
    if MSG[0] == 'up':  # y+
        human_corr = np.array([-1, 0, 1, 0])
        print('current key:', MSG[0])
    if MSG[0] == 'down':  # y-
        human_corr = np.array([1, 0, -1, 0])
        print('current key:', MSG[0])

    if MSG[0] == 'right':  # x+
        human_corr = np.array([0, -1, 0, 1])
        print('current key:', MSG[0])

    if MSG[0] == 'left':  # x-
        human_corr = np.array([0, 1, 0, -1])
        print('current key:', MSG[0])

    return human_corr

def remove_conflict(plt_dict):
    plt_dict['keymap.back'].remove('left')
    plt_dict['keymap.forward'].remove('right')
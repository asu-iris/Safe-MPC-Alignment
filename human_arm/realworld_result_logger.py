import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.abspath(os.getcwd()))

if __name__=='__main__':
    while True:
        user_id=input("please enter the user id:\n")
        trial_id=input("please enter the trial id:\n")
        res_str=input("enter the result: T or F:\n")
        res=None
        if res_str=="T":
            res=True
        elif res_str=="F":
            res=False
        else:
            print("invalid result")
            continue

        f=None
        try:
            path=os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),'Data','user_study_arm_real'
                         ,"user_"+str(user_id),'trial_'+str(trial_id))
            f=open(os.path.join(path,'result.txt'),"w")
            f.write(str(res))
            f.close()
        except Exception as e:
            print(e)
            print("invalid user or trial")



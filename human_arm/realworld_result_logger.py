import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.abspath(os.getcwd()))

if __name__=='__main__':
    while True:
        user_id=input("please enter the user id:\n")
        trial_id=input("please enter the trial id:\n")
        res_str=input("enter the water amount of the trial:\n")
        res=float(res_str)
        out_str=input("enter the water out failure flag: T for success, F for failure\n")
        out_res=None
        if out_str=="T":
            out_res=True
        elif out_str=="F":
            out_res=False
        else:
            print("invalid input")
            break
        f=None
        try:
            path=os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),'Data','user_study_arm_realworld'
                         ,"user_"+str(user_id),'trial_'+str(trial_id))
            f=open(os.path.join(path,'result.txt'),"w")
            f.write(str(res)+' '+str(out_res))
            f.close()
        except Exception as e:
            print(e)
            print("invalid user or trial")



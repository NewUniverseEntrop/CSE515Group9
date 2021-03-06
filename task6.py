from tkinter import *
from tkinter import ttk
import tkinter as tk
import os
import numpy
import pickle
import subprocess

#test=["1","2","3","4","5","1.csv","2.csv","3.csv","5,csv","hello.csv","1.csv","2.csv","3.csv","5,csv","hello.csv","1.csv","2.csv","3.csv","5,csv","hello.csv","1.csv","2.csv","3.csv","5,csv","hello.csv","1.csv","2.csv","3.csv","5,csv","hello.csv","1.csv","2.csv","3.csv","5,csv","hello.csv","1.csv","2.csv","3.csv","5,csv","hello.csv","1.csv","2.csv","3.csv","5,csv","hello.csv"]
#test=["Hello","2","3","4"]

root = Tk()

'''
def query():
    f=str(name.get())  # get name of the file
    numfile=int(num.get())   
    truncated=test[:numfile]
    joined_string = ",".join(truncated)
    txtbox.insert(END,joined_string)

'''


def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di

    
feedbackResult=dict()
btn={}

task3Out="Welcome"
def update(test):
    # update canvas for task 4 and 5 
    global feedbackResult
    feedbackResult = {key:0 for key in test} 
    print("new",feedbackResult)
    for child in scrollable_frame.winfo_children():
        child.destroy()
    numOfResult=num.get()

    
    if numOfResult=="":
        numOfResult=len(test)
    else:
        numOfResult=int(num.get())

    for i in range(numOfResult): # only first n results 
        name=ttk.Label(scrollable_frame, text=test[i])
        name.grid(column=0, row=i)    
        a=ttk.Button(scrollable_frame,text="good", command=lambda key=test[i]: feedback(key,1))
        a.grid(column=1, row=i)
        b=ttk.Button(scrollable_frame,text="neutral",command=lambda key=test[i]: feedback(key,0))
        b.grid(column=2, row=i)
        c=ttk.Button(scrollable_frame,text="bad",command=lambda key=test[i]:feedback(key,-1))
        c.grid(column=3, row=i)
        btn[test[i]]=[c,b,a]
         
    # add output
 

# called when submit feedback buttons are called , 
def feedback(key,value):   # key = name of the file, value = 1 for pos, -1 for neg ...
    #print(key)
    hint.config(text="")
    for x in range(3):
           btn[key][x].config(state="normal")
    

    btn[key][value+1].config(state="disabled")
    global feedbackResult
    feedbackResult[key]=value
    #save_dict(feedbackResult,"./newResult")  
    print(feedbackResult)

def add(command):
    good="" 
    bad=""
    neutral=""
    #global feedbackResult
    for x in feedbackResult:
        
        if feedbackResult[x]==1:
            if good=="":
                good=str(x)
            else:
                good=good+","+str(x)
        elif feedbackResult[x]==-1:
            if bad=="":
                bad=str(x)
            else:
                bad=bad+","+str(x)
        else:
            if neutral=="":
                neutral=str(x)
            else:
                neutral=neutral+","+str(x)
    if good == "":
        good="-1"
    if bad=="":
        bad="-1"
    if neutral=="":
        neutral="-1"
    return twoentry.get()+" task"+command+".py "+task_one_entry.get()+" "+good+" "+bad+" "+neutral

def cmd(cmd):
    print(cmd)
    p = subprocess.Popen(cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
    )
    (output, err) = p.communicate()
 
    ## Wait for date to terminate. Get return returncode ##
    p_status = p.wait()
    print("cmd:"+cmd)
    print("out: '{}'".format(output))
    print("err: '{}'".format(err))
    print("exit: {}".format(p_status))
    return output,err,p

import ast
def run(task,command):
    # run other tasks 
    if "three" in task:    
        ## call date command ##
        command=twoentry.get()+" task3.py "+task_one_entry.get()+" "+command
        print("processing command "+command)
        p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
        output, err = p.communicate()
        global task3Out
        print(output,err)
        output=output.decode().rstrip().split("\n")
        task3Out=output[:2]
        output=output[len(output)-1]
        #print("output",ast.literal_eval(output))
        update(ast.literal_eval(output)) # update gui
        ttk.Label(scrollable_frame,text=task3Out).grid(column=0,row=100, columnspan=10) 
        print("GUI updated !!!! New result shown for task3")
        #hintValue.set("SUCCESSFUL")
    else:
        command=add(command) 
        print("processing command "+command)
        p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
        output, err = p.communicate()
        print(output,err)
        output=output.decode().rstrip().split("\n")
        output=output[len(output)-1]
        #print("output",ast.literal_eval(output))
        update(ast.literal_eval(output)) # update gui
        print("GUI updated !!!! New result shown")
        #hint.config(text="SUCCESSFUL")
    '''
    if "fourth" in task:
        command=add(command) 
        p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
        output, err = p.communicate()
        #print(output)
        output=output.decode().rstrip().split("\n")
        output=output[len(output)-1]
        #print("output",ast.literal_eval(output))
        update(ast.literal_eval(output)) # update gui
    if "five" in task:
        command=add(command) 
        p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
        output, err = p.communicate()
        print(output)
        output=output.decode().rstrip().split("\n")
        output=output[len(output)-1]
        print(output)
        update(ast.literal_eval(output)) # update gui
   
    else:
        p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
        output, err = p.communicate()
        print(output)
    '''
    #print(command)
Grid.rowconfigure(root, 0, weight=1,)
Grid.columnconfigure(root, 0, weight=1)

Controlframe = ttk.Frame(root)
DisplayFrame=ttk.Frame(root,border=5)
feedbackFrame=ttk.Frame(root,border=5)
DisplayFrame.grid(column=1,row=0, sticky=N+S+E+W)
Controlframe.grid(column=0, row=0, sticky=N+S+E+W)

hintlabel=ttk.Label(Controlframe,text="Welcome to Wrapper For CMD")

path = ttk.Label(Controlframe, text="Enter path:")
pyLabel = ttk.Label(Controlframe, text="python/python3 ?")
threelabel = ttk.Label(Controlframe, text="Task 3 : Enter Parameters")
fourLabel= ttk.Label(Controlframe, text="Task 4 :Probalistic relevance feedback")   
fiveLabel= ttk.Label(Controlframe, text="Task 5 :Classifier based relevance:")

#namelbl = ttk.Label(Controlframe, text="Enter file name")
numberOfResultlabel = ttk.Label(Controlframe, text="Enter Result Number:")
#name = ttk.Entry(Controlframe) # input file name text field
num = ttk.Entry(Controlframe,text="10")  # num of file to return text field

task_one_entry = ttk.Entry(Controlframe) # input file name text field
twoentry = ttk.Entry(Controlframe) # input file name text field
threeentry = ttk.Entry(Controlframe) # input file name text field
fourentry=ttk.Entry(Controlframe)
fiveentry=ttk.Entry(Controlframe)

# display results
hintValue=StringVar().set("output:")
hint=ttk.Label(Controlframe,textvariable=hintValue,text="Ouput:")
hint.grid(column=3, row=31)


hintlabel.grid(column=3,row=1)
numberOfResultlabel.grid(column=3,row=2,columnspan=2)
num.grid(column=3, row=3, columnspan=2)

path.grid(column=3, row=4, columnspan=2)
task_one_entry.grid(column=3,row=5,columnspan=10)
#ttk.Button(Controlframe,text="Run task 1",command=lambda :run("path" ,task_one_entry.get())).grid(column=3,row=6,columnspan=2)   
pyLabel.grid(column=3, row=7, columnspan=2)
twoentry.grid(column=3,row=8,columnspan=2)
#ttk.Button(Controlframe,text="Run task 2",command=lambda :run("second",twoentry.get())).grid(column=3,row=9,columnspan=2)

threelabel.grid(column=3, row=11, columnspan=2)
threeentry.grid(column=3,row=12,columnspan=2)
ttk.Button(Controlframe,text="Run task 3", command=lambda :run("three",threeentry.get())).grid(column=3,row=13,columnspan=2)

fourLabel.grid(column=3, row=16)
#fourentry.grid(column=3, row=17)
ttk.Button(Controlframe,text="Submit task 4", command=lambda :run("","4")).grid(column=3,row=18,columnspan=2)

fiveLabel.grid(column=3, row=19)
#fiveentry.grid(column=3,row=20)
ttk.Button(Controlframe,text="Submit task 5", command=lambda :run("","5")).grid(column=3,row=29,columnspan=2)


# left pane
canvas = tk.Canvas(DisplayFrame)
scrollbar = ttk.Scrollbar(DisplayFrame, orient="vertical", command=canvas.yview)
scrollable_frame = ttk.Frame(canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(
        scrollregion=canvas.bbox("all")
    )
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)
canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")



root.mainloop()
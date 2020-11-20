from tkinter import *
from tkinter import ttk
import tkinter as tk
import os
import numpy
import pickle
import subprocess

#test=["1","2","3","4","5","1.csv","2.csv","3.csv","5,csv","hello.csv","1.csv","2.csv","3.csv","5,csv","hello.csv","1.csv","2.csv","3.csv","5,csv","hello.csv","1.csv","2.csv","3.csv","5,csv","hello.csv","1.csv","2.csv","3.csv","5,csv","hello.csv","1.csv","2.csv","3.csv","5,csv","hello.csv","1.csv","2.csv","3.csv","5,csv","hello.csv","1.csv","2.csv","3.csv","5,csv","hello.csv"]
test=["Hello","2","3","4"]

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

    
feedbackResult = {key.rstrip():0 for key in test} 
btn={}

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
        #b=ttk.Button(scrollable_frame,text="neutral",command=lambda key=test[i]: feedback(key,0))
        #b.grid(column=2, row=i)
        c=ttk.Button(scrollable_frame,text="bad",command=lambda key=test[i]:feedback(key,0))
        c.grid(column=3, row=i)
        btn[test[i]]=[c,a]


# called when submit feedback buttons are called , 
def feedback(key,value):   # key = name of the file, value = 1 for pos, -1 for neg ...
    #print(key)
    for x in range(2):
           btn[key][x].config(state="normal")
    btn[key][value].config(state="disabled")
    global feedbackResult
    feedbackResult[key]=value
    #save_dict(feedbackResult,"./newResult")  
    print(feedbackResult)

def add(command):
    relevant="" 
    irrelevant=""
    for x in feedbackResult:
        if feedbackResult[x]==1:
            if relevant=="":
                relevant=relevant+""+x
            else:
                relevant=reversed+" "+x
        else:
            if relevant=="":
                irrelevant=irrelevant+""+x
            else:
                irrelevant=irrelevant+","+x
    return command+" "+relevant+" "+irrelevant
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
def run(command):
    # run other tasks 
    if "task3" in command:    
        ## call date command ##
        p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
        output, err = p.communicate()
        output=output.decode().rstrip().split(",")
        print("output",output)
        update(output) # update gui
        #hintlabel.config(text="task3 successful")
    if "task4" in command:
        command=add(command) 
        cmd(command)
    if "task5" in command:
        command=add(command) 
        cmd(command)
    else:
        os.system(command)
    #print(command)
Grid.rowconfigure(root, 0, weight=1,)
Grid.columnconfigure(root, 0, weight=1)

Controlframe = ttk.Frame(root)
DisplayFrame=ttk.Frame(root,border=5)
feedbackFrame=ttk.Frame(root,border=5)
DisplayFrame.grid(column=1,row=0, sticky=N+S+E+W)
Controlframe.grid(column=0, row=0, sticky=N+S+E+W)

hintlabel=ttk.Label(Controlframe,text="Welcome to Wrapper For CMD")

task_one_lable = ttk.Label(Controlframe, text="Task 1 : Enter value k, n, m  :")
twolabel = ttk.Label(Controlframe, text="Task 2 : Enter Parameters:")
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


hintlabel.grid(column=3,row=1)
numberOfResultlabel.grid(column=3,row=2,columnspan=2)
num.grid(column=3, row=3, columnspan=2)
task_one_lable.grid(column=3, row=4, columnspan=2)
task_one_entry.grid(column=3,row=5,columnspan=2)
ttk.Button(Controlframe,text="Run task 1",command=lambda :run("python task1.py " +task_one_entry.get())).grid(column=3,row=6,columnspan=2)   
twolabel.grid(column=3, row=7, columnspan=2)
twoentry.grid(column=3,row=8,columnspan=2)
ttk.Button(Controlframe,text="Run task 2",command=lambda :run("python task2.py "+twoentry.get())).grid(column=3,row=9,columnspan=2)

threelabel.grid(column=3, row=11, columnspan=2)
threeentry.grid(column=3,row=12,columnspan=2)
ttk.Button(Controlframe,text="Run task 3", command=lambda :run("python task3.py"+threeentry.get())).grid(column=3,row=13,columnspan=2)


fourLabel.grid(column=3, row=16)
fourentry.grid(column=3, row=17)
ttk.Button(Controlframe,text="Submit feedback with task 4", command=lambda :run(fourentry.get())).grid(column=3,row=18,columnspan=2)

fiveLabel.grid(column=3, row=19)
fiveentry.grid(column=3,row=20)
ttk.Button(Controlframe,text="Submit feedback with task 5", command=lambda :run(fiveentry.get())).grid(column=3,row=29,columnspan=2)


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
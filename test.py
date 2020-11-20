from tkinter import *
from tkinter import ttk
import tkinter as tk
import os
import numpy
import pickle

test=["1","2","3","4","5","1.csv","2.csv","3.csv","5,csv","hello.csv","1.csv","2.csv","3.csv","5,csv","hello.csv","1.csv","2.csv","3.csv","5,csv","hello.csv","1.csv","2.csv","3.csv","5,csv","hello.csv","1.csv","2.csv","3.csv","5,csv","hello.csv","1.csv","2.csv","3.csv","5,csv","hello.csv","1.csv","2.csv","3.csv","5,csv","hello.csv","1.csv","2.csv","3.csv","5,csv","hello.csv"]

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
'''
def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di
'''
def feedback():            
    text=txtbox.get(1.0,"end").split(",")
    t = list(map(lambda s: s.strip(), text))   
    print(t)
    save_dict(list(map(lambda x:x.strip(),text)),"./newResult")

    '''
    if new Result is empty then , either because there is on resutl or no result
    CALL TASK 4/5 TO GET BETTER RESULTS WITH newResult
    '''
def run(command):
    os.system(command)
Grid.rowconfigure(root, 0, weight=1,)
Grid.columnconfigure(root, 0, weight=1)

Controlframe = ttk.Frame(root)
DisplayFrame=ttk.Frame(root,border=5)
DisplayFrame.grid(column=1,row=0, sticky=N+S+E+W)
Controlframe.grid(column=0, row=0, sticky=N+S+E+W)


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




numberOfResultlabel.grid(column=3,row=2,columnspan=2)
num.grid(column=3, row=3, columnspan=2)
task_one_lable.grid(column=3, row=4, columnspan=2)
task_one_entry.grid(column=3,row=5,columnspan=2)
ttk.Button(Controlframe,text="Run task 1",command=lambda :run("python task6.py "+task_one_entry.get())).grid(column=3,row=6,columnspan=2) 
twolabel.grid(column=3, row=7, columnspan=2)
twoentry.grid(column=3,row=8,columnspan=2)
ttk.Button(Controlframe,text="Run task 2",command=lambda :run("python task2.py "+twoentry.get())).grid(column=3,row=9,columnspan=2)

threelabel.grid(column=3, row=11, columnspan=2)
threeentry.grid(column=3,row=12,columnspan=2)
ttk.Button(Controlframe,text="Run task 3", command=lambda :run("python task3.py"+threeentry.get())).grid(column=3,row=13,columnspan=2)


fourLabel.grid(column=3, row=16)
ttk.Button(Controlframe,text="Submit feedback with task 4", command=lambda :feedback).grid(column=3,row=17,columnspan=2)

fiveLabel.grid(column=3, row=19)
ttk.Button(Controlframe,text="Submit feedback with task 5", command=lambda :feedback).grid(column=3,row=29,columnspan=2)


# left pane

from tkinter import scrolledtext


txtbox = scrolledtext.ScrolledText(DisplayFrame, width=40, height=10)
txtbox.grid(row=0, column=0,   sticky=E+W+N+S)
# 

root.mainloop()
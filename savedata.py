# -*- coding: utf-8 -*-
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sys import argv
import sys

character = argv[1]

print("The character you choose for recognition is:",character)

datapath = "C:/Users/Wang Jingyuan/Desktop/characters_48"
files = os.listdir(datapath)
if character+".set" not in files:
    print("The character you choose is not found in our dataset.")
    sys.exit(0)

f=open('temp.txt','wb')
i=0
y=[]
namelist=['unknown',character]

#先遍历文件名
for filename in files:
    filepath = datapath + '/' + filename
    file = open(filepath, 'rb')
    for line in file.readlines():
        f.write(line)
    file.close()
    if filename[:-4] == character:
        y_temp = np.arange(1000)
        y_temp.fill(1)
        y = np.concatenate((y, y_temp), axis=0)
    else:
        #更新y数组
        y_temp = np.arange(1000)
        y_temp.fill(0)
        y = np.concatenate((y,y_temp),axis=0)
    print(i)
    i = i+1

#关闭文件
f.close()

tempfile = open('temp.txt','rb')
x = np.fromfile(tempfile, dtype = np.ubyte).reshape(i*1000,48*48)

try:
    with open('Xdata.txt','wb') as x_file:
        pickle.dump(x,x_file)
    with open('Ydata.txt','wb') as y_file:
        pickle.dump(y,y_file) 
    with open('label.txt','wb') as label_file:
        pickle.dump(namelist,label_file) 
except IOError as err:  
    print('File error: ' + str(err))  
except pickle.PickleError as perr:  
    print('Pickling error: ' + str(perr))  
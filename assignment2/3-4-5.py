
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as m


# In[2]:

dataframe = pd.read_csv('iris.csv',header = None)


# In[3]:


df = dataframe.values


# In[4]:


dataset = df[:,0:4]
target = df[:,4]


# In[5]:


high = m.ceil(np.max(dataset[:,0]))
low = m.floor(np.min(dataset[:,0]))
diff = high -low

if(diff%3==0 or diff%7==0 or diff%6==0 or diff%9==0):
    diff = 3
elif(diff%2==0 or diff%4==0 or diff%8==0):
    diff = 4
elif(diff%1==0 or diff%5==0 or diff%10==0):
    diff = 5

l = []
for i in range(1,diff+1):
    l.append([low,low+1])
    low += 1

cluster_1 = []
cluster_2 = []
cluster_3 = []
cluster_4 = []

for i in range(len(df)):
    if(l[0][0]<=df[i][0] and l[0][1]>=df[i][0]):
        cluster_1.append(df[i])
    elif(l[1][0]<=df[i][0] and l[1][1]>=df[i][0]):
        cluster_2.append(df[i])
    elif(l[2][0]<=df[i][0] and l[2][1]>=df[i][0]):
        cluster_3.append(df[i])
    elif(l[3][0]<=df[i][0] and l[3][1]>=df[i][0]):
        cluster_4.append(df[i])


h_list = [np.max(dataset[:,i]) for i in range(4)]
l_list = [np.min(dataset[:,i]) for i in range(4)]


# In[37]:



def method(data,temp,i):
    if(len(data) == 1):
        return data
    
    if(i==0):
        high = m.ceil(np.max(data[:,i]))
        low = m.floor(np.min(data[:,i]))
    else:
        high = -1
        low = 100
        for j in range(len(data)):
            if(high<data[j][i]):
                high = m.ceil(data[j][i])
            if(low>data[j][i]):
                low = m.floor(data[j][i])
        
    diff = high -low
    if(diff%3==0 or diff%7==0 or diff%6==0 or diff%9==0):
        diff = 3
    elif(diff%2==0 or diff%4==0 or diff%8==0):
        diff = 4
    elif(diff%1==0 or diff%5==0 or diff%10==0):
        diff = 5

    l = []
    lower = low
    for j in range(0,diff):
        l.append([low,((high-lower)/diff)+low])
        low = ((high-lower)/diff) + low
    
    for j in range(diff):
        temp.append([])

    for j in range(len(data)):
        for k in range(diff):
            if(l[k][0]<=data[j][i] and l[k][1]>=data[j][i]):
                temp[k].append(list(data[j]))
                break
            
    newTemp = []
    for j in range(len(temp)):
        if(len(temp[j]) != 0):
            newTemp.append(temp[j])
    
    temp = newTemp
            
    
    if(i==3):
        return temp
    else:
        length = len(temp)
        newlist = []
        for j in range(length):
            oldlist = []
            newlist.append(method(temp[j],oldlist,i+1))
        temp = newlist[:]
        newlist = []
        return temp
        
cluster = []
newlist = []
cluster = method(dataset,cluster[:],0)

# In[38]:
#Printing Included

def method(data,temp,i):
    global answer
    if(len(data) == 1):
        return data
    
    if(i==0):
        high = m.ceil(np.max(data[:,i]))
        low = m.floor(np.min(data[:,i]))
    else:
        high = -1
        low = 100
        for j in range(len(data)):
            if(high<data[j][i]):
                high = m.ceil(data[j][i])
            if(low>data[j][i]):
                low = m.floor(data[j][i])
        
    diff = high -low
    if(diff%3==0 or diff%7==0 or diff%6==0 or diff%9==0):
        diff = 3
    elif(diff%2==0 or diff%4==0 or diff%8==0):
        diff = 4
    elif(diff%1==0 or diff%5==0 or diff%10==0):
        diff = 5

    l = []
    lower = low
    for j in range(0,diff):
        l.append([low,((high-lower)/diff)+low])
        low = ((high-lower)/diff) + low
    
    for j in range(diff):
        temp.append([])

    for j in range(len(data)):
        for k in range(diff):
            if(l[k][0]<=data[j][i] and l[k][1]>=data[j][i]):
                temp[k].append(list(data[j]))
                break
            
    newTemp = []
    for j in range(len(temp)):
        if(len(temp[j]) != 0):
            newTemp.append(temp[j])
    
    temp = newTemp
            
    
    if(i==3):
        return temp
    else:
        length = len(temp)
        newlist = []
        for j in range(length):
            oldlist = []
            s = method(temp[j],oldlist,i+1)
            newlist.append(s)
            if i==2:
                for k in range(len(s)):
                    answer.append(s[k])
        temp = newlist[:]
        newlist = []
        return temp
        
cluster = []
newlist = []
answer = []
cluster = method(dataset,cluster[:],0)


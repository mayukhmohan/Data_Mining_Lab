import numpy as np
import pandas as pd
from sklearn import linear_model, datasets, tree
import matplotlib.pyplot as plt
import math as m
import seaborn as sns
import copy

dataframe = pd.read_csv("iris.csv",header=None) #load the dataset
df = dataframe.values
dataset = df[:,0:4]
target = df[:,4]

# In[1]:
'''def pearson_coef(X,Y,mx,my):
    A = sum((X-mx)*(Y-my))
    B = m.sqrt(sum([a*b for a,b in zip(X-mx,X-mx)]))
    C = m.sqrt(sum([a*b for a,b in zip(Y-my,Y-my)]))
    return A / (B * C)

arr_coef = []
for i in range(len(dataset)):
    arr_coef.append([])
    
for i in range(len(dataset)):
    for j in range(len(dataset)):
        X = list(dataset[i,:])
        Y = list(dataset[j,:])
        x_m = np.mean(X)
        y_m = np.mean(Y)
        arr_coef[i].append(pearson_coef(X,Y,x_m,y_m))'''
        

# In[2]:
def pearson_coef(X,Y,mx,my):
    A = sum((X-mx)*(Y-my))
    B = m.sqrt(sum([a*b for a,b in zip(X-mx,X-mx)]))
    C = m.sqrt(sum([a*b for a,b in zip(Y-my,Y-my)]))
    return A / (B * C)

arr_coef = []
for i in range(np.shape(dataset)[1]):
    arr_coef.append([])
    
for i in range(np.shape(dataset)[1]):
    for j in range(np.shape(dataset)[1]):
        X = list(dataset[:,i])
        Y = list(dataset[:,j])
        x_m = np.mean(X)
        y_m = np.mean(Y)
        arr_coef[i].append(pearson_coef(X,Y,x_m,y_m))        

hmap = pd.DataFrame(arr_coef)
sns.heatmap(hmap,cmap='coolwarm',annot=True)
# In[3]:
graph = []
for i in range(np.shape(dataset)[1]):
    graph.append([])
for i in range(np.shape(dataset)[1]):
    for j in range(np.shape(dataset)[1]):
        graph[i].append(arr_coef[i][j])
        
# In[4]:
ans = []
edgeCount = 0
def edge_Count(g):
    global edgeCount
    edgeCount = 0
    for i in range(len(g)):    
        for j in range(len(g[i][1])):
            if g[i][1][j] != -2:
                edgeCount += 1
    return edgeCount
        
        
sum_of_mean = sum(list(np.mean(graph[i]) for i in range(len(graph))))
threshold = sum_of_mean / 10

#graph_dash = copy.deepcopy(graph)
#graph_dash = graph[:,:]
graph_dash = []
for i in range(np.shape(dataset)[1]):
    graph_dash.append([])
    graph_dash[i].append([])
    graph_dash[i].append([])
    graph_dash[i][0] = i
    for j in range(len(graph[i])):
        if graph[i][j] >= threshold and i!=j:
            graph_dash[i][1].append(graph[i][j])
        else:
            graph_dash[i][1].append(-2)

   
while(edge_Count(graph_dash)!=0):
    s = []
    for i in range(len(graph_dash)):    
        su = 0
        for j in range(len(graph_dash[i][1])):
            if graph_dash[i][1][j] != -2:
               su += graph_dash[i][1][j]
        s.append(su)
    
    ind = -1
    for i in range(len(s)):
        if s[i] == max(s):
            ind = i
            break
    
    ans.append(graph_dash[i][0])
    del graph_dash[ind]
    
    for i in range(len(graph_dash)):    
        del graph_dash[i][1][ind]
    


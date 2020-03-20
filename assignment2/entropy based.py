import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import numpy as np 
import pandas as pd


dataset = pd.read_csv('iris.csv',header = None)
#X = dataset.values
#X = X[:,0:4]
X = dataset.iloc[:, [0,4]].values
te=X.shape[1]
#y = dataset.iloc[:, 0].values
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder = LabelEncoder()
X[:, 1] = labelencoder.fit_transform(X[:, 1])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X=X[X[:,3].argsort()]

nclass=X.shape[1]-te+1
print("no of classes::",nclass)
def entropy(a):
    s=sum(a)
    if s==0:
        return 0
    en=0
    for i in range(len(a)):
        g=a[i]/s
        if g==0:
            g=1
        else:
            en+=-g*math.log2(g)
    return en              


def info(e,f,d1,d2):
    m=e/(e+f)
    n=f/(e+f)
    return (m*d1+n*d2)              
                                
k=int(input("enter k:"))
#k=3
n=0
while 2**n<k:
    n+=1
    

spoint=np.zeros(((2**n)-1,3))
q=0
count=0

def initial(X):
    global nclass
    c=[0]*nclass
    for i in range(len(X)):
        for j in range(len(c)):
            if X[i][j]==1:
                c[j]+=1
    return entropy(c)

def split(X,start):
    global count
    count+=1
    global nclass
    global spoint
    global k
    global n
    global q
    index=0
    gain=0
    entr=initial(X)
    #print(entr)
    for i in range(len(X)-1):
        cc=[0]*nclass
        dd=[0]*nclass
        sp=(X[i][nclass]+X[i+1][nclass])/2
        #print(sp)
        for ii in range(len(X)):
            if X[ii][nclass]<=sp:
                for j in range(len(cc)):
                    if X[ii][j]==1:
                        cc[j]+=1
            else:
                for j in range(len(dd)):
                    if X[ii][j]==1:
                        dd[j]+=1
        e=sum(cc)
        f=sum(dd)
        d1=entropy(cc)
        d2=entropy(dd)  
        newin=info(e,f,d1,d2)
        g=entr-newin
        if g>gain:
            #print("hello")
            spoint[q][0]=X[i][nclass]
            index=i
            gain=g
            spoint[q][1]=gain
    spoint[q][2]=index+start
    index=index+start
    q+=1
    return index


def recur(X,start,end,ccc):
    #global n    
    ind=split(X[start:end],start)
    ccc=ccc-1
        #if split is done return
        #else
    if ccc<=0:
        return
    else:
        recur(X,start,ind,ccc)
        recur(X,ind+1,end,ccc)
    #not done yet
#split(X[0:11],0)    
recur(X,0,len(X)-1,n)
spoint=spoint[spoint[:,1].argsort()]
spoint=spoint[::-1]
print("Splitting points");

for i in range(k-1):
    print(spoint[i][0])
      
#ind=split(X)
#ccc=split(X[start:ind])
#
#split(X[start:ccc])
#split(X[ccc+1:ind])
#
#
#split(X[ind+1:end])

#if even no of split             

    
    
    
    


        



    
    
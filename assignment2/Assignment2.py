import numpy as np
from sklearn import linear_model, datasets, tree
import matplotlib.pyplot as plt
import matplotlib as m
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv('ass2.csv')
df2 = pd.read_csv('ass2NAN.csv')
dataset_actual = df2.values
df2 = df2.dropna()
dataset = df.values
dataset2 = df2.values

X = dataset2[:, 0:2] 
y = dataset2[:, 2]

fig = m.pyplot.figure()
ax = Axes3D(fig)

sequence_containing_x_vals = list(dataset[:,0])
sequence_containing_y_vals = list(dataset[:,1])
sequence_containing_z_vals = list(dataset[:,2])

ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals,c = 'b', marker='o')
#m.pyplot.show()


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X,y)
 
'''j=0
X_test = np.zeros((2,2))
for i in dataset_actual:
    if dataset_actual[i][2] is np.nan:
        X_test[j] = dataset_actual[i,0:2]
        j += 1'''
        
X_test = list([[3,-2],[6,5]])
y_pred = regressor.predict(X_test)

seq_x = []
seq_y = []
pred = [y_pred[0],y_pred[1]]

for i in range(0,2):
    seq_x.append(X_test[i][0])
    seq_y.append(X_test[i][1])

ax.scatter(seq_x,seq_y,pred,c = 'r', marker='*')

a = regressor.coef_[0]
b = regressor.coef_[1]
c = regressor.intercept_
xl = []
yl = []
zl = []

arr1 = np.linspace(6,-7,100)
arr2 = np.linspace(0,11,100)

for i in arr1:
    for j in arr2:
        p = a*i + b*j + c
        xl.append(i)
        yl.append(j)
        zl.append(p)
        
xl = np.array(xl)
yl = np.array(yl)
zl = np.array(zl)

ax.plot3D(xl, yl, zl, color ='green', alpha = 0.7)

m.pyplot.show()        
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 16:40:30 2021

@author: admin
"""

##画圆
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
sample_size=1000
x=np.random.uniform(-10,10,(sample_size,2))
plt.scatter(x[:,0],x[:,1],color='red')
radius=7
label=np.zeros(sample_size)
for i in range(sample_size):
    if np.sqrt(x[i,0]**2+x[i,1]**2)<radius:
        label[i]=1
    else:
        label[i]=0
a=(label==1)
b=(label==0)
plt.figure(figsize=(10,10))
plt.scatter(x[a,0],x[a,1],color='green')
plt.scatter(x[b,0],x[b,1],color='pink')

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x,label)
print(model.score(x,label))

x_test=np.random.uniform(-10,10,(10000,2))
y_pred=model.predict(x_test)
plt.scatter(x_test[y_pred==0,0],x_test[y_pred==0,1],color='green')
plt.scatter(x_test[y_pred==1,0],x_test[y_pred==1,1],color='pink')
#用knn预测
from sklearn.neighbors import KNeighborsClassifier

model_knn=KNeighborsClassifier(n_neighbors=3,p=1)
model_knn.fit(x,label)
print(model_knn.score(x,label))

x_test=np.random.uniform(-10,10,(100000,2))
y_pred=model_knn.predict(x_test)

plt.figure(figsize=(10,10))
plt.scatter(x_test[y_pred==0,0],x_test[y_pred==0,1],color='green')
plt.scatter(x_test[y_pred==1,0],x_test[y_pred==1,1],color='pink')
##决策树预测
from sklearn.tree import DecisionTreeClassifier 
import sklearn.tree as tree
model_tree=tree.DecisionTreeClassifier(criterion='entropy',max_depth=500)
model_tree.fit(x,label)
plt.figure(figsize=(19,19))
tree.plot_tree(model_tree,filled=(True))
print(model_tree.score(x,label))

x_test=np.random.uniform(-10,10,(100000,2))
y_pred=model_tree.predict(x_test)

plt.figure(figsize=(10,10))
plt.scatter(x_test[y_pred==0,0],x_test[y_pred==0,1],color='green')
plt.scatter(x_test[y_pred==1,0],x_test[y_pred==1,1],color='pink')

##神经网络
from sklearn.neural_network import MLPRegressor
model=MLPRegressor(hidden_layer_sizes=(100,10))#2代表两个神经元，右边代表隐含层数
model.fit(x,label)
x_test=np.random.uniform(-10,10,(100000,2))
y_pred=model_knn.predict(x_test)

plt.figure(figsize=(10,10))
plt.scatter(x_test[y_pred==0,0],x_test[y_pred==0,1],color='green')
plt.scatter(x_test[y_pred==1,0],x_test[y_pred==1,1],color='pink')












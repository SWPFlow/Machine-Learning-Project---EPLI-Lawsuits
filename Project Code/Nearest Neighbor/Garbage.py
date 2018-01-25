# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 17:35:31 2017

@author: Chris.Cirelli
"""


import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
import pandas as pd
 
#####   LOAD DATA

File_1_Final_ABT_Encoded = pd.read_excel(r'C:\Users\Chris.Cirelli\Desktop\Python Programing Docs\GSU\FInal Project\Data Files\Final Project - Section V - ABT - Features Encoded.xlsx')
df1_ABT_Encoded = pd.DataFrame(File_1_Final_ABT_Encoded)

df1_Features = df1_ABT_Encoded.drop('Claims Count', axis = 1)
df1_Targets = df1_ABT_Encoded['Claims Count']
df2_Features = pd.get_dummies(df1_Features, columns=['SIC Code', 'State_Encoded',                                                        'Broker_Encoded', 'Industry Encoded'])
df2_Targets = df1_Targets
n_neighbors = 30
 

# prepare data
X = df2_Features.iloc[:500, :2]  
y = df2_Targets[:500]
h = .02 
 
# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA','#00AAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00','#00AAFF'])
 
# we create an instance of Neighbours Classifier and fit the data.
clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
clf.fit(X, y)
 
# calculate min, max and limits
x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# predict class using data and kNN classifier
Z = clf.predict(X)
 
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
 
# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("3-Class classification (k = %i)" % (n_neighbors))
plt.show()

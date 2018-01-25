# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 20:04:27 2017

@author: Chris.Cirelli
"""


#   http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html
#   http://scikit-learn.org/stable/modules/neighbors.html
#   http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
#   https://kevinzakka.github.io/2016/07/13/k-nearest-neighbor/

''' NOTES ON NEAREST NEIGHBOR ALGORITHMS

Types               - two types of Supervised NN, classification & regression
Distance            - Normally measured as a Euclidean distance. 
Approach            - NN are considered non-generalizing machine learning algorithms as they simply "remember" all the      
                    training data.
                    
Strenghts           - Being a non-parametric method, it is often successful in classification situations where the decision     
                    boundary is very irregular.  
Decision Boundary   - In a statistical-classification problem with two classes, a decision boundary or decision surface is a    
                    hypersurface that partitions the underlying vector space into two sets, one for each class. The                     
                    classifier will classify all the points on one side of the decision boundary as belonging to one class 
                    and all those on the other side as belonging to the other class.
                    
Algorithms          - BallTree
                    - KDTree
                    - Pairwise 'brute'
                    - NearestNeighbor module allows you to select which to use. When auto is passed, the algorithm will 
                    chose the best fit. 

Regression
                    -Used for continuous versus descrete variables. 
                    -Continous variables have an infinite number of variables between two given points (ex: time)
                    -Descrete variables have a finite number of variables between any given two points (ex: # complaints)

Bruit Force
                    - Brute force computation of the distance between all pairs of points in the dataset. 
                    - Can be useful for small datasets. 
                    - algorithm = 'brute'

K-D Tree
                    - (unclear) uses tree like data structures to measure the distance between points.  If point A is distant 
                    from point B, but B is close to C, then there is no need to calculate the distance from A to C as we know 
                    it is also distant. This cuts down on computation for large datasets. 
                    - Does this detract from accuracy?

Ball-Tree
                    - Addresses the shortcomings of K-D Tree for >20D datasets (high dimensionality datasets)
                    - http://mathworld.wolfram.com/Four-DimensionalGeometry.html
                    - http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.91.8209
                    - Ball-tree partitions data into nested hyper-spheres. 

Nearest Centroid
                    -Centroid: the center of mass of a geometric object of uniform density
                    - Centroid or geometric center of a plane figure is the arithmetic mean ("average") position of all the 
                    points in the shape. 
                    -Algorithm represents each class (assuming feature) by a centroid of its members. 
                    -No features to chose from. 
                    
Neares Shrunken Centroid
                    -The value of each feature for each centroid is divided by the within-class variance of that feature.
                    -I guess this takes into consideration the variance of each dataset, thereby making it more accurate. 
                    
                    

Further Readings:
                    - Kernal Density Estimation 
                    - Density Estimator
                    - Curse of Dimensionality (apparently NN does poorly with D>20)
                    - https://en.wikipedia.org/wiki/Manifold
                    - https://en.wikipedia.org/wiki/Centroid
'''








































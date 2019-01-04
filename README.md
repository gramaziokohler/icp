# icp
Python implementation of classic 3-dimensional Iterative Closest Point method.

ICP finds a best fit rigid body transformation between two point sets.  
Correspondence between the points is not assumed. 
Included is an SVD-based least-squared best-fit algorithm for corresponding point sets.

#Dependency (I think package versions doens't matter)
Numpy : Many mathametical calculations are performed by Numpy library
sklearn : nearest_neighbor function is calling NearestNeighbors from sklearn
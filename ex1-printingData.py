from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()
type(iris)

# Iris data is a long matrix
#print iris.data

## Data is numpy array because they are optimized to perform
## high performance computations
print "Iris data type is               : " + str(type(iris.data))
print "Iris response type is           : " + str(type(iris.target))

# columns are features and rows are samples
print "Iris data set has the shape     : " + str(iris.data.shape)
print "Iris response set has the shape : " + str(iris.target.shape)


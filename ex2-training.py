"""""
Use of k-nearest to classify the labeled iris data

KNN defines areas defining classes where it is most likely that
new sample belongs. These areas are based which are neighbors belongsl
to which group. The standard ML process stablishes four steps:
    1 import class that embeds the algorithm that you are aiming to use
    2 instantiate the estimator or classifier
    3 fit model aka model training
    4

"""""

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

# step 1 Loading data
iris = load_iris()
X = iris.data
y = iris.target

# step 2
# notice that you can specify tuning parameters aka hyperparameters

knn = KNeighborsClassifier(n_neighbors=1)
print knn

# step 3
# training model. Model learns from X to predict y
# process occurs in place not need to create vars
knn.fit(X,y)

# step 4
# test model

X_new  = ( [3,5,4,2] , [5,4,3,2])
prediction1 = knn.predict(X_new)# predict list of lists
print "The prediction for the samples " + str(X_new)+ " is: " + str(prediction1)
# Note 1:
# advantege of sklearn is that use other model is easy just inst
# other model
# from sklearn.linear_model  import LogisticRegression
# og  =  ogisticregression()
# og.fit(X,y)
# og.predict(X_new)#

# Note 2:
# It is not possible to measure truly whether or not the predictions are
# successful or not. Nonetheless is possible to create tests to likely define the
# proper model to our data.

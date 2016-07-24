"""""
In this exercise we investigate the metrics given by
sklearn to compare performance over a punch of different
parameters varying over the training process of a knn
 algorithm trying to predict especies on the famous
 iris data-set.

"""""

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
"""
    'accuracy_score',
    'adjusted_mutual_info_score',
    'adjusted_rand_score',
    'auc',
    'average_precision_score',
    'classification_report',
    'cluster',
    'completeness_score',
    'confusion_matrix',
    'consensus_score',
    'coverage_error',
    'euclidean_distances',
    'explained_variance_score',
    'f1_score',
    'fbeta_score',
    'get_scorer',
    'hamming_loss',
    'hinge_loss',
    'homogeneity_completeness_v_measure',
    'homogeneity_score',
    'jaccard_similarity_score',
    'label_ranking_average_precision_score',
    'label_ranking_loss',
    'log_loss',
    'make_scorer',
    'matthews_corrcoef',
    'mean_absolute_error',
    'mean_squared_error',
    'median_absolute_error',
    'mutual_info_score',
    'normalized_mutual_info_score',
    'pairwise_distances',
    'pairwise_distances_argmin',
    'pairwise_distances_argmin_min',
    'pairwise_distances_argmin_min',
    'pairwise_kernels',
    'precision_recall_curve',
    'precision_recall_fscore_support',
    'precision_score',
    'r2_score',
    'recall_score',
    'roc_auc_score',
    'roc_curve',
    'SCORERS',
    'silhouette_samples',
    'silhouette_score',
    'v_measure_score',
    'zero_one_loss',
    'brier_score_loss',
"""

# We train our model
numberOfNeighbors = 1
iris = load_iris()
X = iris.data
y = iris.target

knn = KNeighborsClassifier(n_neighbors=numberOfNeighbors)
print knn

knn.fit(X,y)

prediction1 = knn.predict(X)# predict list of lists
print "The prediction for the samples " + str(X)+ " \n is: " + str(prediction1)+ "\n compared to " + str(y);
print "The prediction metric (accuracy score) is: " + str(metrics.accuracy_score(y,prediction1))


from math import sqrt
import numpy as np
import itertools 
from math import sqrt
#TODO: can do one more numpy array casting inside all function

def euclidean_distance(vector1,vector2,squared=False):
	"""calculates euclidean distance between two vectors
	
	Keyword arguments:
	vector1 -- first data point (type: numpy array)
	vector2 -- second data point (type: numpy array)
	squared -- return square of euclidean distance (default: False)
	"""

	euclidean_distance=np.sum((vector1-vector2)**2)

	if squared is False:
		euclidean_distance=sqrt(euclidean_distance)

	return euclidean_distance

def compute_cluster_centers(data,labels,return_counts=False):
	"""Computes mean center of points in a cluster

	Keyword arguments:
	data -- data points (type: numpy.ndarray)
	labels -- cluster labels of the point (type: numpy.ndarray)
	"""
	n_clusters=np.unique(labels).max()+1

	cluster_centers=np.zeros((n_clusters,data.shape[1]))
	cluster_counts=np.zeros(n_clusters)

	for point_index in range(len(data)):
		#to avoid noise points (labels=-1)
		if labels[point_index]>=0:
			cluster_counts[labels[point_index]] += 1
			cluster_centers[labels[point_index]] += data[point_index]

	for index in range(n_clusters):
		cluster_centers[index]=cluster_centers[index]/cluster_counts[index]

	if return_counts is False:
		return cluster_centers

	else:
		return cluster_centers,cluster_counts

''' Partitional Clustering validation --start '''

def scatter_matrices(data,labels):
	"""Computes total, within-cluster and between-cluster scatter matrices

	Total Sparse Matrix (St)- square of distance of each point from global data-center
	Within-cluster Sparse Matrix (Sw) - square of distance of each point from corresponding cluster centers (Minimisation)
	Between-cluster Sparse Matrix (Sb) - weighted square of distance between cluster-centers and global data-center weighed by no. of points in the cluster (Maximisation) 

	Keyword arguments:
	data -- data points
	cluster_centers -- mean centers of points in clusters
	labels -- cluster label of data points
	
	Note: increasing value of K (no. partitions) obviously minimizes sum of squared error (variance)
	Reference: Clustering -- RUI XU,DONALD C. WUNSCH, II (IEEE Press)
	"""

	n_clusters=np.unique(labels).max()+1
	#doesn't consider noise as a cluster
	centers,counts=compute_cluster_centers(data,labels,return_counts=True)

	total_mean_vector=np.zeros(data.shape[1])
	for cluster_index in range(n_clusters):
		total_mean_vector += counts[cluster_index]*centers[cluster_index]

	total_mean_vector=total_mean_vector/np.sum(counts)

	St,Sw,Sb=0,0,0

	for point_index in range(len(data)):
		#to avoid noise (label: -1)
		if labels[point_index]>=0:
			St += euclidean_distance(data[point_index],total_mean_vector,squared=True)
			Sw += euclidean_distance(data[point_index],centers[labels[point_index]],squared=True)

	for cluster_index in range(n_clusters):
		Sb += counts[cluster_index]*euclidean_distance(centers[cluster_index],total_mean_vector,squared=True)
	
	return St,Sw,Sb
''' Partitional Clustering validation --end '''



def Compute_adjusted_rand_index(labels_true ,labels_pred):
	"""Adjusted Rand Index (SKLEARN)
	Reference:
	http://scikit-learn.org/stable/modules/clustering.html#adjusted-rand-index 
	"""
	return metrics.adjusted_rand_score(labels_true, labels_pred) 


def Compute_adjusted_mutual_information(labels_true,labels_pred):
	"""Adjusted Mutual information(SKLEARN)
	Reference:
	http://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_mutual_info_score.html#sklearn.metrics.adjusted_mutual_info_score
	"""
	return metrics.adjusted_mutual_info_score(labels_true, labels_pred)  


def Compute_normalized_mutual_information(labels_true,labels_pred):
	"""normalized Mutual information(SKLEARN)
	Reference:
	http://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html#sklearn.metrics.normalized_mutual_info_score
	"""
	return metrics.normalized_mutual_info_score(labels_true, labels_pred) 


def Compute_homogeneity_score(labels_true,labels_pred):
	"""homogeneity_score(SKLEARN)
	Reference:
	http://scikit-learn.org/stable/modules/generated/sklearn.metrics.homogeneity_score.html#sklearn.metrics.homogeneity_score
	"""
	return metrics.homogeneity_score(labels_true, labels_pred) 


def Compute_completness_score(labels_true,labels_pred):
	"""completeness_score(SKLEARN)
	Reference:
	http://scikit-learn.org/stable/modules/generated/sklearn.metrics.completeness_score.html#sklearn.metrics.completeness_score
	"""
	return metrics.completeness_score(labels_true, labels_pred) 

def Compute_v_measure_score(labels_true, labels_pred):
	"""v_measure_score(SKLEARN)
	Reference:
	http://scikit-learn.org/stable/modules/generated/sklearn.metrics.v_measure_score.html#sklearn.metrics.v_measure_score
	"""
	return metrics.v_measure_score(labels_true, labels_pred) 


def Compute_fowlkes_mallows_score(labels_true, labels_pred):
	"""fowlkes_mallows_score(SKLEARN)
	Reference:
	http://scikit-learn.org/stable/modules/generated/sklearn.metrics.fowlkes_mallows_score.html#sklearn.metrics.fowlkes_mallows_score
	"""	
	return metrics.fowlkes_mallows_score(labels_true, labels_pred) 

def Compute_purity(labels_true,labels_pred):
	"""Purity 
	Reference:
	http://www.caner.io/purity-in-python.html
	"""	
	A = np.c_[(labels_pred,labels_true)]
	n_accurate = 0.
	for j in np.unique(A[:,0]):
		z = A[A[:,0] == j, 1]
		x = np.argmax(np.bincount(z))
		n_accurate += len(z[z == x])

	return n_accurate / A.shape[0]

def Compute_jaccard_co_eff(labels_true, labels_pred):
	"""jaccard co-eff
	chapter 10 -bible of clustering

	Case a: x i and x j belong to the same clusters of C and the same category 	  of P .
	Case b: x i and x j belong to the same clusters of C but different categories of P .
	Case c: x i and x j belong to different clusters of C but the same category   of P .
	j=a/(a+b+c)
	"""
	a,b,c=0,0,0
	
	n=len(labels_true)
	
	for i, j in itertools.combinations(xrange(n), 2):
		comembership1 = labels_true[i] == labels_true[j]
		comembership2 = labels_pred[i] == labels_pred[j]
		if comembership1 and comembership2:
			a += 1
		elif comembership1 and not comembership2:
			b += 1
		elif not comembership1 and comembership2:
			c += 1
	return float(a) / (a+b+c)


def Compute_gamma_statistics(labels_true, labels_pred):
	"""gamma statistics
	chapter 10 -bible of clustering

	Case a: x i and x j belong to the same clusters of C and the same category 	  of P .
	Case b: x i and x j belong to the same clusters of C but different categories of P .
	Case c: x i and x j belong to different clusters of C but the same category   of P .
	"""
	a = b= c = d = 0

	n = len(labels_true)
	for i, j in itertools.combinations(xrange(n), 2):
		comembership1 = labels_true[i] == labels_true[j]
		comembership2 = labels_pred[i] == labels_pred[j]
		if comembership1 and comembership2:
			a += 1
		elif comembership1 and not comembership2:
			b += 1
		elif not comembership1 and comembership2:
			c += 1
		elif not comembership1 and comembership2:
			d +=1	
	M = a + b + c + d
	m1 = a + b
	m2 = a + c
	numerator = (M*a) - (m1*m2)
	denominator = sqrt(m1 * m2 * (M - m1 ) * (M - m2))
	return numerator/denominator

def Compute_silhouette_score(data,labels_pred):
	"""silhouette_score(SKLEARN)
	Reference:
	http://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html#sklearn.metrics.silhouette_score	
	"""	
	return metrics.silhouette_score(data,labels_pred, metric='euclidean')

def Compute_calinski_harabaz_score(data, labels_pred) :
	"""silhouette_score(SKLEARN)
	Reference:
	http://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html#sklearn.metrics.silhouette_score	
	"""	
	return metrics.calinski_harabaz_score(data, labels_pred)
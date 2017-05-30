from math import sqrt
import numpy as np

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

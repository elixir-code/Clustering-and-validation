import numpy as np
from itertools import combinations

import h5py

def DBCV(data, labels, distance_matrix, hdf5_file = None):
	"""DBCV (Density-based Cluster Validation)
	Cluster validation of density-based arbitary shaped clusters 

	References: [1] Density-Based Clustering Validation (D. Moulavi, P. Jaskowiak, R. Campello, et al.)

	Keyword arguments --
	data -- The data matrix (shape: n_samples x n_features , dtype: numerical)
	labels -- The cluster label for each sample (shape: n_samples, dtype: intp)
	distance matrix -- metric distances between samples (shape: n_samples x n_samples, dtype: np.float64) 
	"""
	n_samples, n_features = data.shape
	n_clusters = np.max(labels) + 1

	clusters_size = np.zeros(n_clusters, dtype=np.intp)
	clusters_pt_indices = np.empty(n_clusters, dtype='O')

	for cluster_index in range(n_clusters):
		
		choosen = np.where(labels == cluster_index)[0]
		clusters_size[cluster_index] = choosen.shape[0]
		clusters_pt_indices[cluster_index] = list(choosen)
	
	#noise points will have zero "all-points-core-distance"
	apts_coredist = np.zeros(n_samples, dtype=np.float64)

	for sample_index in range(n_samples):
		if labels[sample_index] == -1 :
			continue

		#the cluster index of the current sample/ data point
		sample_label = labels[sample_index]

		k_neighbor_indices = clusters_pt_indices[sample_label].copy()
		k_neighbor_indices.remove(sample_index)

		dist = distance_matrix[sample_index][k_neighbor_indices]

		apts_coredist[sample_index] = np.power( np.sum( np.power(dist, -1*n_features) ) / (clusters_size[sample_label] - 1) , -1/n_features)	

	#compute mutual rechability distances
	if hdf5_file is None:
		mreach_distance_matrix = np.zeros((n_samples, n_samples), dtype = 'd')

	else:
		print("\nEnter 'r' to read mutual distance matrix"
			  "\nEnter 'w' to write mutual distance matrix"
			  "\nMode : ",end='')
		mode = input().strip()

		if mode == 'r':
			mreach_distance_matrix = hdf5_file['mreach_distance_matrix']

		elif mode == 'w':
			if 'mreach_distance_matrix' in hdf5_file:
				del hdf5_file['mreach_distance_matrix']
				
			mreach_distance_matrix = hdf5_file.create_dataset('mreach_distance_matrix',(n_samples,n_samples), dtype='d')

	for i in range(n_samples):
		for j in range(i,n_samples):
			mreach_distance_matrix[i][j] = mreach_distance_matrix[j][i] = max(apts_coredist[i], apts_coredist[j], distance_matrix[i][j])

	return mreach_distance_matrix

	

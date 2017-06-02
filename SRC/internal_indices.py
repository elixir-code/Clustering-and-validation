"""INTERNAL CLUSTERING INDICES

1.  Within-class and between-class scatter values
2.  The Ball-Hall index
3.  The Banfeld-Raftery index
4.  The Calinski-Harabasz index
5.  The Det Ratio index
6.  The Ksq DetW index
7.  The Log Det Ratio index
8.  The Log SS Ratio index
9.  The Scott-Symons index
10. The Silhouette index
11. The Trace WiB index
"""

import numpy as np

class internal_indices:

	def __init__(self,data,labels):
		#TODO : filter out noise(DBSCAN, HDBSCAN)

		#initialise class memebers
		self.data=np.array(data)
		self.labels=np.array(labels)

		self.n_samples=self.data.shape[0]
		self.n_features=self.data.shape[1]

		#compute mean of data
		self.data_mean=np.mean(self.data,axis=0)

		#self.n_clusters=np.unique([x for x in self.labels if x>=0]).shape[0] (to avoid noise)
		self.n_clusters=np.unique(self.labels).shape[0]
		self.clusters_mean = np.zeros((self.n_clusters,self.n_features))
		self.clusters_size = np.zeros(self.n_clusters)

		#TODO: preprocess to remove noise
		for i,cluster_label in enumerate(np.unique(self.labels)):
			#if cluster_label >=0  (to avoid noise)
			cluster_i_pts = (self.labels==cluster_label)
			self.clusters_size[i] = np.sum(cluster_i_pts)
			self.clusters_mean[i] = np.mean(self.data[cluster_i_pts],axis=0)


	def compute_scatter_matrices(self):
		"""
		References:	[1] Clustering Indices, Bernard Desgraupes (April 2013)
					[2] http://sebastianraschka.com/Articles/2014_python_lda.html (Linear Discriminatory Analysis)
					[3] Chapter 4, Clustering -- RUI XU,DONALD C. WUNSCH, II (IEEE Press)

		verified with data from References [2] 
		"""
		self.T = total_scatter_matrix(self.data)
		
		#WG_clusters : WG matrix for each cluster | WGSS_clusters : trace(WG matrix) for each cluster
		self.WG_clusters = np.empty((self.n_clusters,self.n_features,self.n_features),dtype=np.float64)
		self.WGSS_clusters = np.zeros(self.n_clusters,dtype=np.float64) 

		#self.BG = np.zeros((self.n_features,self.n_features),dtype=np.float64)

		for i,cluster_label in enumerate(np.unique(self.labels)):
			#compute within cluster matrix
			self.WG_clusters[i] = total_scatter_matrix(self.data[self.labels==cluster_label]) 
			self.WGSS_clusters[i] = np.trace(self.WG_clusters[i])

			#compute between-cluster matrix
			mean_vec = self.clusters_mean[i].reshape((self.n_features,1))
			overall_mean = self.data_mean.reshape((self.n_features,1))

			#self.BG += np.array(self.clusters_size[i]*(mean_vec - overall_mean).dot((mean_vec - overall_mean).T),dtype=np.float64)
			#self.BG = self.BG + self.clusters_size[i]*np.dot(cluster_data_mean_diff.T,cluster_data_mean_diff)

		self.WG = np.sum(self.WG_clusters,axis=0)
		self.WGSS = np.trace(self.WG)

		self.BG = self.T - self.WG
		self.BGSS = np.trace(self.BG)

		self.det_WG = np.linalg.det(self.WG)
		self.det_T = np.linalg.det(self.T) 

	# internal indices -- start

	def ball_hall_index(self):
		"""
		Ball Hall index -- mean, through all the clusters, of their mean dispersion
		References:	[1] Clustering Indices, Bernard Desgraupes (April 2013)
		"""
		sum_mean_disperions = 0.

		for cluster_i in range(self.n_clusters): 
			sum_mean_disperions + = self.WGSS_clusters[i] / self.clusters_size[i]
		
		return sum_mean_disperions/self.n_clusters

	def banfeld_raftery_index(self):
	    """Banfeld-Raftery index -- weighted sum of the logarithms of the traces of the variance-
		covariance matrix of each cluster (rule: min)
		
		References :	[1] Clustering Indices, Bernard Desgraupes (April 2013)
						[2] https://www.stat.washington.edu/raftery/Research/PDF/banfield1993.pdf

	    Tr(W G{k})/nk -- the mean of the squared distances between the points in cluster Ck and their barycenter G{k}. 
	    If a cluster contains a single point, this trace is equal to 0 and the logarithm is undefined.
		
		[2] Appropriate for hyperspherical clusters (may be of different sizes)
	    """
		br_index = np.sum(self.clusters_size*np.log(self.WGSS_clusters/self.clusters_size))
		return br_index


    def  det_ratio_index(self):
	    """The Determinant Ratio 
	            Det_R =det(T)/det(WG).
	    """
    	return self.det_T/self.det_WG

    def ksq_detw_index(self):
    	return self.n_clusters*self.n_clusters*self.det_WG

    def log_det_ratio_index(self):
    	return self.n_samples * log(self.det_T/self.det_WG)

   	def log_ss_ratio_index(self):
   		return log(self.BGSS/self.WGSS)

   	def scott_symons_index(self):
   		"""
   		Scott-Symons Index -- weighted sum of the logarithms of the determinants of the variance-covariance matrix of each cluster
   		"""
   		scott_symons_index = 0.
   		for i in range(self.n_clusters):
   			scott_symons_index += self.clusters_size[i]*log(np.linalg.det(self.WG_clusters[i]/self.clusters_size[i]))

   		return scott_symons_index

   	def trace_wib_index(self):
   		"""
		Trace WiB index -- Maximise (BG/WG), i.e., maximise BG and minimise WG
   		"""
   		return np.trace(np.linalg.inv(self.WG).dot(self.BG))

   	def silhouette_score(self):
		"""Silhouette score (sklearn)

		Reference: [1] http://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html#sklearn.metrics.silhouette_score	
		"""
		return metrics.silhouette_score(self.data,self.labels, metric='euclidean')


	def calinski_harabaz_score(self) :
		"""Calinski-Harabasz index (sklearn)

		References:	[1] http://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html#sklearn.metrics.silhouette_score	
					[2] Clustering Indices, Bernard Desgraupes (April 2013)
		"""
		#return ((self.n_samples - self.n_clusters)/(self.n_clusters - 1)) * (self.BGSS/self.WGSS)
		return metrics.calinski_harabaz_score(self.data,self.labels)

	# internal indices -- end

# helper functions -- start
def total_scatter_matrix(data):
	"""
	Total sum of square (TSS) : sum of squared distances of points around the baycentre
	References : Clustering Indices, Bernard Desgraupes (April 2013)
	"""
	X=np.array(data.T.copy(),dtype=np.float64)

	for feature_i in range(data.shape[1]):
		X[feature_i] = X[feature_i] - np.mean(X[feature_i])

	T = np.dot(X,X.T)
	return T

# helper functions -- end

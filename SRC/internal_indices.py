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

12. C-index
13. Dunn-index

14. Davies-Bouldin index (*)
"""
from sklearn import metrics

import numpy as np
from math import sqrt

from sklearn.preprocessing import LabelEncoder
class internal_indices:

	def __init__(self,data,labels):
		#TODO: preprocess to remove noise

		#normalising labels
		le = LabelEncoder()
		le.fit(labels)

		#initialise class memebers
		self.data=np.array(data)
		self.labels=le.transform(labels)

		self.n_samples=self.data.shape[0]
		self.n_features=self.data.shape[1]

		#compute mean of data
		self.data_mean=np.mean(self.data,axis=0)

		#self.n_clusters=np.unique([x for x in self.labels if x>=0]).shape[0] (to avoid noise)
		self.n_clusters=np.unique(self.labels).shape[0]
		self.clusters_mean = np.zeros((self.n_clusters,self.n_features))
		self.clusters_size = np.zeros(self.n_clusters)

		for cluster_label in range(self.n_clusters):
			#if cluster_label >=0  (to avoid noise)
			cluster_i_pts = (self.labels==cluster_label)
			self.clusters_size[cluster_label] = np.sum(cluster_i_pts)
			self.clusters_mean[cluster_label] = np.mean(self.data[cluster_i_pts],axis=0)

		self.compute_scatter_matrices()

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

		for cluster_label in range(self.n_clusters):
			#compute within cluster matrix
			self.WG_clusters[cluster_label] = total_scatter_matrix(self.data[self.labels==cluster_label]) 
			self.WGSS_clusters[cluster_label] = np.trace(self.WG_clusters[cluster_label])

			#compute between-cluster matrix
			mean_vec = self.clusters_mean[cluster_label].reshape((self.n_features,1))
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
			sum_mean_disperions += self.WGSS_clusters[i] / self.clusters_size[i]
		
		return sum_mean_disperions/self.n_clusters

	#TODO : verify if denominator inside log is nk (total no. of pts) or nk.(nk-1)/2 (total pairs of points)
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
		return ((self.n_samples - self.n_clusters)/(self.n_clusters - 1)) * (self.BGSS/self.WGSS)
		#return metrics.calinski_harabaz_score(self.data,self.labels)

	#C-index
	def c_index(self):
		"""C-index : 	range [0,1]
						rule : min
		"""
		Sw = 0.
		Nk = 0.5 * (np.sum(self.clusters_size**2) - self.n_samples)

		s_min_max_array = get_k_largest_smallest(int(Nk))

		#parse through all pair of points
		for data1_index in range(self.n_samples-1):
			for data2_index in range(data1_index+1,self.n_samples):
				distance = euclidean_distance(self.data[data1_index],self.data[data2_index])

				if self.labels[data1_index] == self.labels[data2_index]:
					Sw += distance

				s_min_max_array.insert(distance)

		s_min = np.sum(s_min_max_array.k_smallest)
		s_max = np.sum(s_min_max_array.k_largest)

		print(Sw,s_min_max_array.k_largest,s_min_max_array.k_smallest)
		return (Sw - s_min)/(s_max - s_min)

	def dunn_index(self):
		"""Dunn index -- ratio between the minimal intracluster distance to maximal intercluster distance
		
		References :	[1] https://www.biomedcentral.com/content/supplementary/1471-2105-9-90-S2.pdf

		range : [0,infinity) | rule : max
		"""
		max_intra_cluster = min_inter_cluster = euclidean_distance(self.data[0],self.data[1])

		#parse through all pair of points
		for data1_index in range(self.n_samples-1):
			for data2_index in range(data1_index+1,self.n_samples):
				distance = euclidean_distance(self.data[data1_index],self.data[data2_index])

				#both are same index
				if (self.labels[data1_index] == self.labels[data2_index]) and (distance > max_intra_cluster):
					max_intra_cluster = distance

				elif (self.labels[data1_index] != self.labels[data2_index]) and (distance < min_inter_cluster):
					min_inter_cluster = distance

		return min_inter_cluster/max_intra_cluster

	def davies_bouldin_index(self):
		""" Davies-Bouldin index -- Small values of DB correspond to clusters that are compact, and whose
			centers are far away from each other.

			References :	[1] https://www.biomedcentral.com/content/supplementary/1471-2105-9-90-S2.pdf
							[2] https://en.wikipedia.org/wiki/Davies%E2%80%93Bouldin_index
		"""
		total_clusters_dispersion = np.zeros(self.n_clusters)

		for data_index in range(self.n_samples):
			total_clusters_dispersion[self.labels[data_index]] = euclidean_distance(self.data[data_index],self.clusters_mean[self.labels[data_index]])

		mean_clusters_dispersion = total_clusters_dispersion / self.clusters_size

		sum_Mk = 0.

		for cluster_i in range(self.n_clusters):
			max_Mk = 0.
			for cluster_j in range(self.n_clusters):
				if cluster_i != cluster_j :
					Mk = (mean_clusters_dispersion[cluster_i] + mean_clusters_dispersion[cluster_j])/euclidean_distance(self.clusters_mean[cluster_i],self.clusters_mean[cluster_j])
					if Mk > max_Mk :
						max_Mk = Mk
			sum_Mk += max_Mk

		return sum_Mk/self.n_clusters
		
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

# helper functions -- end

class get_k_largest_smallest:

	def __init__(self,k):
		self.k_largest  = np.empty(k)
		self.k_smallest = np.empty(k)

		self.largest_top = -1
		self.smallest_top = -1

		self.k = k

	def insert(self,value):
		#inserting into the k_smallest list
		if self.smallest_top < self.k-1:
			index = self.smallest_top
			while index>=0 :
				if value < self.k_smallest[index]:
					self.k_smallest[index+1] = self.k_smallest[index]
					index -= 1

				else:
					break

			self.k_smallest[index+1]=value
			self.smallest_top += 1

		else:
			if value < self.k_smallest[self.smallest_top]:
				index = self.smallest_top-1
				while index>=0 :
					if value < self.k_smallest[index]:
						self.k_smallest[index+1] = self.k_smallest[index]
						index -= 1

					else:
						break

				self.k_smallest[index+1]=value

		#inserting into k_largest
		if self.largest_top < self.k-1:
			index = self.largest_top
			while index>=0 :
				if value > self.k_largest[index]:
					self.k_largest[index+1] = self.k_largest[index]
					index -= 1
				else:
					break
			self.k_largest[index+1] = value
			self.largest_top += 1

		else:
			if value > self.k_largest[self.largest_top]:
				index = self.largest_top - 1
				while index>=0 :
					if value > self.k_largest[index]:
						self.k_largest[index+1] = self.k_largest[index]
						index -= 1
					else:
						break

				self.k_largest[index+1] = value

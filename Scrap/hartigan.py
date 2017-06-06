#check weights to take average
def hartigan_index(self,corrected = False):
	"""
	Hartigan Index : generally used to find no. cluster in a dataset (used only for K-Means Algorithm)
	Optimum number of clusters : K for which Hartigan(K) <= n (usually, n=10)

	References :	[1]	http://suendermann.com/su/pdf/ijcbs2009.pdf
	"""
	if corrected is True:
		no_clusters = self.n_clusters-1

	else:
		no_clusters = self.n_clusters+1

	#perfrom Kmeans of the data with K+1 clusters
	kmeans_clusterer=KMeans(n_clusters=no_clusters)
	labels = kmeans_clusterer.fit_predict(self.data)

	#compute cluster-centers and intra-cluster dispersion 
	
	total_dispersion = 0.

	for cluster_i in range(no_clusters):

		cluster_i_indices = (labels==cluster_i)
		cluster_i_pts = self.data[cluster_i_indices]
		cluster_i_mean = np.mean(cluster_i_pts)

		cluster_i_dispersion = np.sum((cluster_i_pts - cluster_i_mean)**2)

		if corrected is True:
			#weight to compute average : 1/no_samples_i or no_samples_i/no_samples
			average_i_dispersion = cluster_i_dispersion / np.sum(cluster_i_indices)
			total_dispersion += average_i_dispersion

		else:
			total_dispersion += cluster_i_dispersion

	if corrected is True:
		#check weight for average
		return (self.n_samples-self.n_clusters)*(total_dispersion-self.WGSS)/self.WGSS
		
	else:
		return (self.n_samples-self.n_clusters-1)*(self.WGSS-self.total_dispersion)/self.total_dispersion

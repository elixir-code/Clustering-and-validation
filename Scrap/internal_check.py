from math import log

The Calinski-Harabasz index
The Ratkowsky-Lance index
The Ray-Turi index

*
def Compute_ball_hall_index(self):
    Summation_result = 0 
    for i in n_clusters:
        WGSS_i= self.WGSS_matrices[i]
        second_summation_result = WGSS_i / self.clusters_size[i]  
        Summation_result = Summation_result + second_summation_result
    return Summation_result/ self.n_clusters


def Compute_Banfeld_Raftery_index(self):
    """Banfeld_Raftery_index
        
    This index is the weighted sum of the logarithms of the traces of the variancecovariance
    matrix of each cluster.
    The index can be written like this:

    BR= summation over k with range(0 to k):n{k}* log(trace(WG{k})/n{k})

    The quantity Tr(W G{k})/nk can be interpreted, after equation (15), as the mean of the squared distances between the points in cluster Ck and their barycenter G{k}. 
    If a cluster contains a single point, this trace is equal to 0 and the logarithm is undefined.

    """
    result = 0
    for i in self.n_clusters:
        trace_WG = np.trace(self.WG[i])
        log_term = trace_WG / self.clusters_size[i]
        log_value = log(log_term)
        product = self.clusters_size[i] * log_value
        result = result + product

    return result

def  Compute_Det_Ratio_index(self):
    """The Determinant Ratio 
        the index is defined like this:
            Det_R =det(T)/det(WG)
        T designates the total scatter matrix . This is the sum of matrices BG and WG .
    """
    WG_sum= np.sum(WG,axis=0)
    determinanant_scatter_matix = np.linalg.det(self.total_scatter_matrix)
    determinanant_WG_sum = np.linalg.det(WG_sum)

    return determinanant_scatter_matix/determinanant_WG_sum

def Compute_Ksq_DetW_index(self):

    determinanant_WG_sum = np.linalg.det(WG_sum)
    return K * K  * determinanant_WG_sum

def Compute_Log_Det_Ratio_index(self):

    log_term =self.Compute_Det_Ratio_index() 
    log_value = log(log_term)
    return self.n_samples * log_value

def Compute_Scott_Symons_index(self):

    result = 0 
    for i in n_clusters:
        WG_of_i= self.WG[i]
        WG_by_n_cluster = WG_of_i / self.n_clusters
        determinant_value = np.linalg.det(WG_by_n_cluster)
        product = self.n_clusters * log(determinant_value)
        result = result + product

def davies_bouldin_index(self):
    """Compute the Davies Bouldin index
    .
    The index is defined as the ratio of within-cluster
    and between-cluster distances.
    Parameters
    ----------
    X : array-like, shape (``n_samples``, ``n_features``)
        List of ``n_features``-dimensional data points. Each row corresponds
        to a single data point.
    labels : array-like, shape (``n_samples``,)
        Predicted labels for each sample.
    Returns
    -------
    score : float
        The resulting Davies-Bouldin index.
    References
    ----------
    .. [1] `Davies, David L.; Bouldin, Donald W. (1979).
       "A Cluster Separation Measure". IEEE Transactions on
       Pattern Analysis and Machine Intelligence. PAMI-1 (2): 224-227`_
    
    https://github.com/tomron/scikit-learn/blob/davies_bouldin_index/sklearn/metrics/cluster/unsupervised.py

    """
    X =  self.data
    labels = self.labels_pred

    X, labels = check_X_y(X, labels)
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    n_samples, _ = X.shape
    n_labels = len(le.classes_)

    intra_dists = np.zeros(n_labels)
    centroids = np.zeros((n_labels, len(X[0])), np.float32)
    for k in range(n_labels):
        cluster_k = X[labels == k]
        mean_k = np.mean(cluster_k, axis=0)
        centroids[k] = mean_k
        intra_dists[k] = np.average(pairwise_distances(cluster_k, [mean_k]))
    centroid_distances = pairwise_distances(centroids)
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.all((intra_dists[:, None] + intra_dists) == 0.0) or \
           np.all(centroid_distances == 0.0):
            return 0.0
        scores = (intra_dists[:, None] + intra_dists)/centroid_distances
        # remove inf values
        scores[scores == np.inf] = np.nan
        return np.mean(np.nanmax(scores, axis=1))

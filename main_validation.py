import numpy as np
data=np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]])
labels=np.array([0,0,1,1,-1])

import validation
center=validation.compute_cluster_center(data,labels)

validation.sum_sqr_error(data,center,labels)
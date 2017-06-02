import numpy as np

class_labels=np.array([0,0,1,1,2,2])
clust_labels=np.array([1,2,0,1,2,0])

import SRC.external_indices as extval

clust_validation=extval.external_indices(class_labels,clust_labels)
clust_validation.entropy()
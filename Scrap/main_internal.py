import numpy as np

import SRC.internal_indices as intval

#data = np.array([[1,1,1],[1,10,8],[7,8,9],[10,31,12],[30,43,15],[40,17,81]])
#labels = np.array([0,1,1,0,0,1])

A = intval.internal_indices(X,y)
A.compute_scatter_matrices()
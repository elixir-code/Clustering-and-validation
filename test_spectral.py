import numpy as np
import SRC.EDA as EDA

from sklearn.datasets import make_circles
main = EDA.EDA(force_file=False)
X, y = make_circles(n_samples=2000, factor=.4, noise=.05)
main.load_data(X,y)

EDA.visualise_2D(main.data.T[0],main.data.T[1],main.class_labels)

main.comp_distance_matrix()
main.affinity_matrix = np.exp(-0.5*main.distance_matrix**2)


from sklearn.neighbors import kneighbors_graph
connectivity = kneighbors_graph(main.data,n_neighbors=10,include_self=False)

affinity = 0.5 * (connectivity + connectivity.T)

diag = []
for row in main.affinity_matrix:
	diag.append(row.sum())

D = np.diag(diag)

D_si = D.copy()
for i in range(D_si.shape[0]):
	D_si[i][i] = 1/np.sqrt(D[i][i])

M = D_si.dot(D - main.affinity_matrix).dot(D_si)
E = np.linalg.eig(M)

main.comp_distance_matrix()
E = np.linalg.eig(main.distance_matrix)


"""
Try Spectral with :
=> k-nearest neighbors graph
=> r-neighborhood graph
=> k-nearest neighbor and r-neighborhood combined

=> Mask or do element wise multiplication of RBF affinity with connectivity *** (Important)
"""
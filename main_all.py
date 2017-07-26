# import SRC.internal_indices as intval
# import SRC.external_indices as extval
# import SRC.EDA as EDA

# from sklearn import datasets
# dataset = datasets.load_iris()
# data = dataset.data
# class_labels = dataset.target

# main = EDA.EDA()
# main.load_data(data)
# main.comp_distance_matrix()

# main.perform_kmeans(3,{"random_state":1})

# val_int=intval.internal_indices(main.data,main.kmeans_results['labels'])
# val_ext=extval.external_indices(class_labels,main.kmeans_results['labels'])

# ==========================================================================================================
import numpy as np
#from contextlib import suppress

import SRC.EDA as EDA

main = EDA.EDA(force_file=False)
main.read_data("SAMPLES/LARGE/HTRU_2.csv",label_cols=-1,normalize_labels=True)

main.read_data("SAMPLES/STANDARD/iris.data",header=None,label_cols=-1,normalize_labels=True)
EDA.visualise_3D(main.data.T[0],main.data.T[1],main.data.T[2],main.class_labels)

main.read_data("SAMPLES/agaricus-lepiota.data",header=None,label_cols=0,normalize_labels=True,na_values='?')
main.read_data("SAMPLES/yeast.data",sep='\s+',header=None,label_cols=-1,normalize_labels=True)

#60021 x 281
main.read_data("SAMPLES/LARGE/Relation Network (Directed).data",header=None,label_cols=0,normalize_labels=True)
#main.read_data("SAMPLES/STANDARD/blogData_train.csv",header=None,label_cols=None)
#main.read_data("SAMPLES/STANDARD/hepatitis.data",header=None,label_cols=0,normalize_labels=True,na_values="?")

main.read_data("SAMPLES/STANDARD/flame.txt",sep='\s+',header=None,label_cols=-1,normalize_labels=True)
#main.read_data("SAMPLES/STANDARD/Skin_NonSkin.csv",header=None,label_cols=-1,normalize_labels=True)

main.read_data("SAMPLES/Classification/data_banknote_authentication.txt",label_cols=-1,normalize_labels=True)

#print("Finished Reading Data ...")

main.comp_distance_matrix()
#print("Computed distance matrix ...")

#main.det_dbscan_params(min_samples=500)
main.dbscan_params={"min_samples":500,"eps":100}
main.perform_dbscan()
# main.perform_kmeans(no_clusters=20)

#import SRC.internal_indices as intval
# import SRC.external_indices as extval

#val_int = intval.internal_indices(main.data,main.kmeans_results['labels'])#,main.distance_matrix)
# val_int = intval.internal_indices(main.data,main.class_labels)

# print("Completed generating Internal Indices Object ...")

# val_ext = extval.external_indices(main.class_labels,main.kmeans_results['labels'])
# print("Completed generating External Indices Object\n\n")

# print("Internal Indices",end="\n\n")

# with suppress(Exception):
	print("ball-hall index : ",val_int.ball_hall_index())
	print("Banfeld-Raftery index : ",val_int.banfeld_raftery_index())
	print("Det Ratio Index : ",val_int.det_ratio_index())
	print("Ksq-detw Index  :",val_int.ksq_detw_index())
	print("Log-det Ratio Index : ",val_int.log_det_ratio_index())
	print("Log SS Ratio Index : ",val_int.log_ss_ratio_index())
	print("Scott-Symons Index : ",val_int.scott_symons_index())
	print("Trace WiB Index : ",val_int.trace_wib_index())
	print("Sillhoutte Index : ",val_int.silhouette_score())
	print("Calinski-Harabasz index : ",val_int.calinski_harabaz_score())
	print("C-index : ",val_int.c_index())
	print("Dunn Index : ",val_int.dunn_index())
	print("Davies Bouldin Index : ",val_int.davies_bouldin_index())
	print("Ray-Turi Index : ",val_int.ray_turi_index())
	print("Hartigan Index : ",val_int.hartigan_index())
	print("PBM Index : ",val_int.pbm_index())
	print("Score Function : ",val_int.score_function())

# print("\n\nExternal Indices",end='\n\n')

# with suppress(Exception):
	print("entropy:",val_ext.entropy())
	print("precision_coefficient:",val_ext.precision_coefficient())
	print("recall_coefficient:",val_ext.recall_coefficient())
	print("f_measure:",val_ext.f_measure())
	print("weighted_f_measure:",val_ext.weighted_f_measure())
	print("purity:",val_ext.purity())
	print("folkes_mallows_index:",val_ext.folkes_mallows_index())
	print("rand_index:",val_ext.rand_index())
	print("adjusted_rand_index:",val_ext. adjusted_rand_index())
	print("adjusted_mutual_information:",val_ext.adjusted_mutual_information())
	print("normalized_mutual_information:",val_ext.normalized_mutual_information())
	print("homogeneity_score:",val_ext.homogeneity_score())
	print("completness_score:",val_ext.completness_score())
	print("v_measure_score:",val_ext.v_measure_score())
	print("jaccard_co_eff:",val_ext.jaccard_co_eff())
	print("gamma_statistics:",val_ext.gamma_statistics())
	print(" kulczynski_index:",val_ext. kulczynski_index())
	print("mcnemar_index:",val_ext.mcnemar_index())
	print("phi_index:",val_ext.phi_index())
	print(" rogers_tanimoto_index:",val_ext. rogers_tanimoto_index())
	print("sokal_sneath_index1:",val_ext.sokal_sneath_index1())
	print("sokal_sneath_index2:",val_ext.sokal_sneath_index2())

# ============================================================================================================

# import numpy as np
# filter = np.array([x in [1,2] for x in main.class_labels])
# main.data = main.data[filter]
# main.class_labels = main.class_labels[filter]
# main.class_labels -= 1
# main.class_names = main.class_names[1:]

# main.standardize_data()
# EDA.visualise_2D(main.data.T[0],main.data.T[1],main.class_labels,main.class_names)

# import SRC.internal_indices as intval
# val_int = intval.internal_indices(main.data,main.class_labels)

# M = np.linalg.inv(val_int.WG).dot(val_int.BG)
# E = np.linalg.eig(M)

# =================================================================================================
# radius=10
# neigh_ind_list = [np.where(d <= radius)[0] for d in main.distance_matrix]
from sklearn.neighbors import kneighbors_graph
main.affinity_matrix = kneighbors_graph(main.data,n_neighbors=10,include_self=False)

main.comp_distance_matrix()
main.affinity_matrix = np.exp(-50*main.distance_matrix**2)

from scipy.sparse.csgraph import laplacian
lap,dd = laplacian(main.affinity_matrix,normed=True,return_diag=True)

E = np.linalg.eigh(lap)
W = E[1].T[[0,1]].T
EDA.visualise_2D(W.T[0],W.T[1],main.class_labels)

np.argsort(E[0])


==================================================================================================

import numpy as np
#from contextlib import suppress

import SRC.EDA as EDA

main = EDA.EDA(force_file=False)
#main.read_data("SAMPLES/STANDARD/flame.txt",sep='\s+',header=None,label_cols=-1,normalize_labels=True)
main.read_data("SAMPLES/STANDARD/iris.data",header=None,label_cols=-1,normalize_labels=True)

#main.read_data("SAMPLES/lda_ex.data",sep='\s+',header=None,label_cols=0,normalize_labels=True)
import SRC.internal_indices as intval
val_int = intval.internal_indices(main.data,main.class_labels)

scatter_WG = scatter_BG = 0
for k in range(val_int.n_clusters):
	choosen = list(np.where(main.class_labels == k)[0])
	data = main.data[choosen]
	#scatter_k = np.sum(np.sum(np.power((data - val_int.clusters_mean[k]),2), axis=1))
	scatter_WG_k = np.sum(np.power((data - val_int.clusters_mean[k]),2))
	scatter_WG = scatter_WG + scatter_WG_k
	print(scatter_WG_k)
	scatter_BG_k = val_int.clusters_size[k] * EDA.euclidean_distance(val_int.clusters_mean[k],val_int.data_mean,squared=True)
	scatter_BG = scatter_BG + scatter_BG_k
	print(scatter_BG_k,end='\n\n')

#calculating Within-class Scatter Matrix
WG = np.zeros((main.n_features,main.n_features),dtype='d')

from itertools import combinations

for i in range(main.n_features):
	for j in range(i,main.n_features):
		scatter = 0
		for k in range(val_int.n_clusters):
			choosen = list(np.where(main.class_labels == k)[0])
			data = main.data[choosen].T[[i,j]]
			var = (data - np.mean(data,axis=1).reshape((2,-1)))
			sum_covar = 0
			for x,y in zip(var[0], var[1]):
				sum_covar = sum_covar + x*y
			scatter = scatter + sum_covar
		WG[i][j] = WG[j][i] = scatter

# scatter_WG
# val_int.WGSS

# scatter_BG
# val_int.BGSS

======================================================================================================================

import SRC.DBCV as dbcv
dbcv.DBCV(main.data, main.dbscan_results['labels'], main.distance_matrix)

x,y,z = np.random.random((3,100000))
x = list(x)
y = list(y)
z = list(z)

x.append(0)
y.append(0)
z.append(0)

data = np.array([x,y,z]).T
labels = np.array([0]* 1001)

main = EDA.EDA(True)
main.load_data(data, labels)
main.comp_distance_matrix()

dbcv.DBCV(main.data, main.class_labels, main.distance_matrix)

0.57162133, 0.60793988, 0.6335773, 0.55707602 : Real
0.5250492881286235 : Expected


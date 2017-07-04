import numpy as np
from contextlib import suppress

# from  time import time
# from os import system

import SRC.EDA as EDA

main = EDA.EDA(False)
#main.read_data("SAMPLES/STANDARD/iris.data",header=None,label_cols=-1,normalize_labels=True)
#main.read_data("SAMPLES/LARGE/Relation Network (Directed).data",header=None,label_cols=0,normalize_labels=True)
#main.read_data("SAMPLES/STANDARD/MopsiLocationsUntil2012-Finland.txt",sep='\s+',header=None,label_cols=None,normalize_labels=False)
#main.data = main.data[:40000]
#main.data = main.data[:10000]
#main.read_data("SAMPLES/hdbscan.npy",sep='\s+',header=None,label_cols=None,normalize_labels=False)
#main.comp_distance_matrix()

#start_time=time()
#labels = main.perform_hierarchial(no_clusters=10)
#EDA.visualise_2D(main.data.T[0],main.data.T[1],labels)
#total_time = time()-start_time
main.perform_spectral_clustering(no_clusters=3,affinity='nearest_neighbors')# , params={'eigen_solver':'amg'})
main.spectral_results['labels'].dump("RESULTS/spectral_results.dump")

#print("-- time = %s seconds --"%(total_time))

import SRC.internal_indices as intval
import SRC.external_indices as extval

val_int = intval.internal_indices(main.data,main.spectral_results['labels'])#,main.distance_matrix)
val_ext = extval.external_indices(main.class_labels,main.spectral_results['labels'])


print("\n\nInternal Indices",end="\n\n")

with suppress(Exception):
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

print("\n\nExternal Indices",end='\n\n')

# with suppress(Exception):
# 	print("entropy:",val_ext.entropy())
# 	print("precision_coefficient:",val_ext.precision_coefficient())
# 	print("recall_coefficient:",val_ext.recall_coefficient())
# 	print("f_measure:",val_ext.f_measure())
# 	print("weighted_f_measure:",val_ext.weighted_f_measure())
# 	print("purity:",val_ext.purity())
# 	print("folkes_mallows_index:",val_ext.folkes_mallows_index())
# 	print("rand_index:",val_ext.rand_index())
# 	print("adjusted_rand_index:",val_ext. adjusted_rand_index())
# 	print("adjusted_mutual_information:",val_ext.adjusted_mutual_information())
# 	print("normalized_mutual_information:",val_ext.normalized_mutual_information())
# 	print("homogeneity_score:",val_ext.homogeneity_score())
# 	print("completness_score:",val_ext.completness_score())
# 	print("v_measure_score:",val_ext.v_measure_score())
# 	print("jaccard_co_eff:",val_ext.jaccard_co_eff())
# 	#print("gamma_statistics:",val_ext.gamma_statistics())
# 	print(" kulczynski_index:",val_ext. kulczynski_index())
# 	print("mcnemar_index:",val_ext.mcnemar_index())
# 	print("phi_index:",val_ext.phi_index())
# 	print("rogers_tanimoto_index:",val_ext. rogers_tanimoto_index())
# 	print("sokal_sneath_index1:",val_ext.sokal_sneath_index1())
# 	print("sokal_sneath_index2:",val_ext.sokal_sneath_index2())

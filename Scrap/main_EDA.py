import SRC.EDA as EDA
cluster_data=EDA.EDA(False)
cluster_data.read_data("SAMPLES/STANDARD/syn_2d.data",sep='\s+',header=None,label_cols=None)

cluster_data.read_data("SAMPLES/STANDARD/hdbscan.npy",sep='\s+',header=None,label_cols=None)
cluster_data.gap_statistics(15)

cluster_data.comp_distance_matrix()
cluster_data.det_dbscan_params(plot_scale=0.001)
cluster_data.perform_dbscan()

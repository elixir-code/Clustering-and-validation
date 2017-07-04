import SRC.EDA as EDA
cluster_data=EDA.EDA()
cluster_data.read_data("./SAMPLES/hdbscan.npy",0,0)
cluster_data.gap_statistics(15)

cluster_data.comp_distance_matrix()
cluster_data.det_dbscan_params(plot_scale=0.001)
cluster_data.perform_dbscan()

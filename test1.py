import SRC.EDA as EDA
from sklearn.cluster import SpectralClustering
cluster_data=EDA.EDA(force_file=False)
cluster_data.read_data("SAMPLES/STANDARD/MopsiLocationsUntil2012-Finland.txt",sep='\s+',header=None,label_cols=None,normalize_labels=False)
#cluster_data.read_data("./SAMPLES/STANDARD/hepatitis.data",header=None,label_cols=0,normalize_labels=True,na_values="?")

#EDA.visualise_2D(cluster_data.data.T[0],cluster_data.data.T[1])
print("Computing distance matrix ...")
cluster_data.comp_distance_matrix()
spectral =  SpectralClustering(n_clusters=5)
print("Starting Raw Spectral ...")
labels_spec = spectral.fit_predict(cluster_data.data)

print("Starting EDA Spectral ...")
cluster_data.perform_spectral_clustering(no_clusters=5)

EDA.visualise_2D(cluster_data.data.T[0],cluster_data.data.T[1],labels_spec)
EDA.visualise_2D(cluster_data.data.T[0],cluster_data.data.T[1],cluster_data.spectral_results["labels"])

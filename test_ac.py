import numpy as np
#from contextlib import suppress

# from  time import time
# from os import system

import SRC.EDA as EDA

main = EDA.EDA(False)
#main.read_data("SAMPLES/STANDARD/iris.data",header=None,label_cols=-1,normalize_labels=True)
#main.read_data("SAMPLES/LARGE/Relation Network (Directed).data",header=None,label_cols=0,normalize_labels=True)
main.read_data("SAMPLES/STANDARD/MopsiLocationsUntil2012-Finland.txt",sep='\s+',header=None,label_cols=None,normalize_labels=False)
#main.data = main.data[:40000]
#main.read_data("SAMPLES/hdbscan.npy",sep='\s+',header=None,label_cols=None,normalize_labels=False)
#main.comp_distance_matrix()

#start_time=time()
labels = main.perform_hierarchial(no_clusters=10)
#EDA.visualise_2D(main.data.T[0],main.data.T[1],labels)
#total_time = time()-start_time

#print("-- time = %s seconds --"%(total_time))

#labels.dump("RESULTS/maps_labels.dump")

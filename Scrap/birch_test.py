import numpy as np
#from contextlib import suppress

import SRC.EDA as EDA

main = EDA.EDA(force_file=False)

main.read_data("SAMPLES/LARGE/Relation Network (Directed).data",header=None,label_cols=0,normalize_labels=True)

from sklearn.cluster import Birch
clusterer = Birch(n_clusters=3,threshold=2)
clusterer.fit(main.data)

print("Labels : ")
print(*clusterer.labels_)

print("Cluster count : ")
print(np.bincount(clusterer.labels_))

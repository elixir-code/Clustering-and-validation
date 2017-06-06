import SRC.internal_indices as intval
import SRC.external_indices as extval
import SRC.EDA as EDA

from sklearn import datasets
dataset = datasets.load_iris()
# data = dataset.data
# class_labels = dataset.target

# main = EDA.EDA()
# main.load_data(data)
# main.comp_distance_matrix()

# main.perform_kmeans(3,{"random_state":1})

# val_int=intval.internal_indices(main.data,main.kmeans_results['labels'])
# val_ext=extval.external_indices(class_labels,main.kmeans_results['labels'])

# ==========================================================================================================

import SRC.EDA as EDA

main = EDA.EDA()
#main.read_data("SAMPLES/STANDARD/nci60.csv",label_cols=-1,na_values='NA',keep_default_na=False)
main.read_data("SAMPLES/STANDARD/student.csv",header=0,label_cols=0,normalize_labels=True)
main.comp_distance_matrix()
main.perform_kmeans(no_clusters=3)

import SRC.internal_indices as intval
import SRC.external_indices as extval

val_int = intval.internal_indices(main.data,main.kmeans_results['labels'],main.distance_matrix)

val_ext = extval.external_indices(main.class_labels,main.kmeans_results['labels'])

# ============================================================================================================


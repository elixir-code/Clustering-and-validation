import SRC.internal_indices as intval
import SRC.external_indices as extval
import SRC.EDA as EDA

from sklearn import datasets
dataset = datasets.load_iris()
data = dataset.data
class_labels = dataset.target

main = EDA.EDA()
main.load_data(data)
main.comp_distance_matrix()

main.perform_kmeans(3,{"random_state":1})

val_int=intval.internal_indices(main.data,main.kmeans_results['labels'])

val_ext=extval.external_indices(class_labels,main.kmeans_results['labels'])
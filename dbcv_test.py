import numpy as np
import SRC.EDA as EDA
import SRC.DBCV as dbcv

#required_core_dists = []
main = EDA.EDA(False)

core_dists_main = []
# for iteration in range(10):
for i in range(500):
x,y,z = np.random.random((3,100))
x = list(2*x - 1)
y = list(2*y - 1)
z = list(2*z - 1)
x.append(0)
y.append(0)
z.append(0)

data = np.array([x,y,z]).T
labels = np.array([0]* 101)
main.load_data(data, labels)
main.comp_distance_matrix()
core_dists = dbcv.DBCV(main.data, main.class_labels, main.distance_matrix)
core_dists_main.append(core_dists[-1])

#required_core_dists.append(core_dists[-1])
print("Expected : ",np.power(np.log(100000), -1/3))
print("Actual   : ",core_dists[-1])
#print(required_core_dists, np.mean(required_core_dists))

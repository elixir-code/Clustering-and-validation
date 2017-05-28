"""Exploratory Data Analysis (EDA) - Clustering Toolkit
Authors : R.Mukesh, Nitin Shravan (BuddiHealth Technologies)
"""

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.cluster import DBSCAN

class EDA:

	#reads data matrix
	def read_data(self,filename,offset_top=1,offset_left=1,sep='\t'):

		data=[]
		line_number=0
		with open(filename,encoding="utf-8") as data_file:
			for line in data_file:
				line_number+=1

				if line_number > offset_top:
					data.append([float(x) for x in line.strip().split(sep)[offset_left:]])
		self.data=np.array(data)

		self.n_samples=self.data.shape[0]
		self.n_features=self.data.shape[1]


	#computes euclidean distance matrix (for all pairs of data points)
	def comp_distance_matrix(self):
		self.distance_matrix=pairwise_distances(self.data)

	#determines dbscan parameters
	def det_dbscan_params(self,min_samples=None,plot_scale=0.02):
		
		if min_samples is None:
			if 2*self.n_features <= self.n_samples:
				min_samples=2*self.n_features
			else:
				raise Exception("please choice a value of min_samples <= no_samples")

		kdist=[]

		for src_distances in self.distance_matrix:
			kmin_distances=np.copy(src_distances[:min_samples])
			kmin_sorted=np.sort(kmin_distances)
			del kmin_distances

			#print(kmin_sorted)

			for distance in src_distances[min_samples:]:
				#print(distance)

				if distance < kmin_sorted[min_samples-1]:
					index=min_samples-2
					while index>=0 :
						if kmin_sorted[index] > distance :
							kmin_sorted[index+1]=kmin_sorted[index]
							index -= 1
						else:
							break

					kmin_sorted[index+1]=distance

				#print(kmin_sorted)

			#print(kmin_sorted,end="\n\n")

			kdist.append(kmin_sorted[min_samples-1])
			del kmin_sorted

			self.kdist=np.copy(kdist)
		
		kdist.sort(reverse=True)
		
		#plot point vs k-dist
		plt.title("Finding DBSCAN parameters (min_samples, epsilon)")
		plt.xlabel("Points ====>> ")
		plt.ylabel("K-distance (k = "+str(min_samples)+")")
		plt.grid(True)

		x_points=np.arange(0.0,self.n_samples*plot_scale,plot_scale)
		plt.plot(x_points,kdist,"k")
		plt.show()

		print("Enter estimated value of eps : ")
		eps=float(input().strip())

		self.dbscan_params={"min_samples":min_samples,"eps":eps}


	#gather results  by performing dbscan
	def perform_dbscan(self):
		dbscan_clusterer=DBSCAN(**self.dbscan_params,metric="precomputed")
		dbscan_clusterer.fit(self.distance_matrix)
		self.dbscan_results={"parameters":dbscan_clusterer.get_params(),"n_clusters":np.unique(dbscan_clusterer.labels_).max()+1,'clusters':label_cnt_dict(dbscan_clusterer.labels_)}		

		print_dict(self.dbscan_results)

def label_cnt_dict(labels):
	unique, counts = np.unique(labels, return_counts=True)
	return dict(zip(unique, counts))

def print_dict(dictionary):
	for key,value in dictionary.items():
		print(key,value,sep=" : ")

def visualise_2D(x_values,y_values,labels=None):
	"""Visualise clusters of selected 2 features"""

	sns.set_style('white')
	sns.set_context('poster')
	sns.set_color_codes()
	plot_kwds = {'alpha' : 0.5, 's' : 80, 'linewidths':0}

	palette = sns.color_palette()
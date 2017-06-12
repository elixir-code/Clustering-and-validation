"""Exploratory Data Analysis (EDA) - Clustering Toolkit
Authors : R.Mukesh, Nitin Shravan (BuddiHealth Technologies)

Dependencies: numpy, sklearn, matplotlib, hdbscan, seaborn, gapkmean
version : Python 3.0

TODO : Hierarchial Clustering Techniques and associated metrics
"""

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.cluster import DBSCAN
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans

from sklearn.cluster import SpectralClustering

from math import pow

#downloaded source -- pip didn't work 
from GAP import gap

import pandas.io.parsers as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler

from itertools import combinations
import matplotlib.patches as mpatches

class EDA:

	def load_data(self,data,labels=None):
		"""Load data externally processed in python into the EDA object
		
		keyword arguments --
		data -- Data matrix (dtype : np.array)
		labels -- Ground Truth Labels (if available)
		"""
		self.data=np.array(data)
		self.n_samples=self.data.shape[0]
		self.n_features=self.data.shape[1]

		if labels is not None:
			self.class_labels = labels

	'''
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
	'''

	def read_data(self,file_path,sep=',',header=None,label_cols=-1,normalize_labels=False,na_values=None):
		"""Read data and ground truth labels as an array from a file

		Keyword arguments -- 
		file_path -- corresponds to 'filepath_or_buffer' argument of pandas.read_csv()
		sep -- corresponds to 'sep' argument of pandas.read_csv() (default : ',')
		header -- corresponds to 'header' argument of pandas.read_csv() (default : None)
		label_cols -- an array of integer of 'ground truth' label columns in the file

		Note : By default drops all data points with attributes 'NA'
		"""
		data_frame = pd.read_csv(filepath_or_buffer=file_path,sep=sep,header=header,index_col=label_cols,na_values=na_values)
		data_frame.dropna(inplace=True)
		#May need drop na function from pandas

		self.data = data_frame.values
		self.class_labels = data_frame.index

		if normalize_labels is True:

			enc = LabelEncoder()
			label_encoder = enc.fit(self.class_labels)
			#original class names
			self.class_names = label_encoder.classes_
			self.class_labels = label_encoder.transform(self.class_labels)

		self.n_samples=self.data.shape[0]
		self.n_features=self.data.shape[1]

		del data_frame
	

	def standardize_data(self):
		"""Standardization of data
		Reference :	[1] https://7264-843222-gh.circle-artifacts.com/0/home/ubuntu/scikit-learn/doc/_build/html/stable/auto_examples/preprocessing/plot_scaling_importance.html
					[2] Standarisation v/s Normalization : http://www.dataminingblog.com/standardization-vs-normalization/
		
		Tested : Mean and variance of data
		"""
		self.std_scale = StandardScaler().fit(self.data)
		self.std_scale.transform(self.data,copy=False)

	#code to destandardise the dataset for visulisation/ metric evaluation
	def destandardize_data(self):
		self.std_scale.inverse_transform(self.data,copy=False)

	#computes euclidean distance matrix (for all pairs of data points)
	def comp_distance_matrix(self):
		self.distance_matrix=pairwise_distances(self.data)
		#self.distance_matrix = np.zeros((self.n_samples,self.n_samples),dtype=np.float32)
		#for data1_index,data2_index in combinations(range(self.n_samples),2):
		#	self.distance_matrix[data1_index][data2_index] = self.distance_matrix[data2_index][data1_index] = euclidean_distance(self.data[data1_index],self.data[data2_index])

	#TODO : arrange k-dist in increasing order and plot
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

	def wk_inertia_stat(self,k_max,k_min=1):

		Wk_array=np.empty(k_max-k_min+1,dtype=np.float64)
		inertia_array=np.empty(k_max-k_min+1,dtype=np.float64)

		#run kmeans and compute log(wk) for all n_clusters
		for no_clusters in range(k_min,k_max+1):

			kmeans_clusterer=KMeans(n_clusters=no_clusters)
			kmeans_clusterer.fit(self.data)

			Dr=np.zeros(no_clusters)
			unique,Nr=np.unique(kmeans_clusterer.labels_,return_counts=True)
			del unique

			for i in range(self.n_samples-1):
				for j in range(i+1,self.n_samples):
					if kmeans_clusterer.labels_[i]==kmeans_clusterer.labels_[j]:
						Dr[kmeans_clusterer.labels_[i]] += pow(self.distance_matrix[i][j],2)

			Wk=np.sum(Dr/(2*Nr))
			Wk_array[no_clusters-k_min]=Wk
			inertia_array[no_clusters-k_min]=kmeans_clusterer.inertia_*100

			del kmeans_clusterer,Dr,Nr,Wk

		plt.title("Wk vs n_clusters")
		plt.xlabel("n_clusters")
		plt.ylabel("Wk")
		plt.grid(True)

		plt.plot(np.arange(k_min,k_max+1),Wk_array,"k")
		plt.show()

		plt.title("INTERIA TO FIND NUMBER OF CLUSTERS")
		plt.xlabel("n_clusters")
		plt.ylabel("inertia")

		plt.plot(np.arange(k_min,k_max+1),inertia_array,"k")
		plt.show()

	#find no. of clusters - gap statistics
	def gap_statistics(self,k_max,k_min=1):
		"""Library used : gapkmeans (downloaded source : https://github.com/minddrummer/gap)"""
		#refs=None, B=10
		gaps,sk,K = gap.gap_statistic(self.data,refs=None,B=10,K=range(k_min,k_max+1),N_init = 10)
		
		plt.title("GAP STATISTICS")
		plt.xlabel("n_clusters")
		plt.ylabel("gap")

		plt.plot(K,gaps,"k",linewidth=2)
		plt.show()

	#gather results  by performing dbscan
	def perform_dbscan(self):
		dbscan_clusterer=DBSCAN(**self.dbscan_params,metric="precomputed")
		dbscan_clusterer.fit(self.distance_matrix)
		self.dbscan_results={"parameters":dbscan_clusterer.get_params(),"labels":dbscan_clusterer.labels_,"n_clusters":np.unique(dbscan_clusterer.labels_).max()+1,'clusters':label_cnt_dict(dbscan_clusterer.labels_)}		

		print_dict(self.dbscan_results)

	def perform_hdbscan(self,min_cluster_size=15):
		hdbscan_clusterer=HDBSCAN(min_cluster_size,metric="precomputed")
		hdbscan_clusterer.fit(self.distance_matrix)
		self.hdbscan_results={"parameters":hdbscan_clusterer.get_params(),"labels":hdbscan_clusterer.labels_,"probabilities":hdbscan_clusterer.probabilities_,"n_clusters":np.unique(hdbscan_clusterer.labels_).max()+1,'clusters':label_cnt_dict(hdbscan_clusterer.labels_)}

		print_dict(self.hdbscan_results)

	def perform_spectral_clustering(self,no_clusters,params={}):
		spectral_clusterer=SpectralClustering(n_clusters=no_clusters,**params)
		spectral_clusterer.fit(self.distance_matrix)
		self.spectral_results={"parameters":spectral_clusterer.get_params(),"labels":spectral_clusterer.labels_,"n_clusters":np.unique(spectral_clusterer.labels_).max()+1,"clusters":label_cnt_dict(spectral_clusterer.labels_)}

		print_dict(self.spectral_results)

		#gaussian kernel affinity matrix
		self.affinity_matrix = spectral_clusterer.affinity_matrix_

	def perform_kmeans(self,no_clusters,params={}):
		kmeans_clusterer=KMeans(n_clusters=no_clusters,**params)
		kmeans_clusterer.fit(self.data)
		self.kmeans_results={"parameters":kmeans_clusterer.get_params(),"labels":kmeans_clusterer.labels_,"n_clusters":no_clusters,'clusters':label_cnt_dict(kmeans_clusterer.labels_),"cluster_centers":kmeans_clusterer.cluster_centers_,"inertia":kmeans_clusterer.inertia_}     

		print_dict(self.kmeans_results)

def label_cnt_dict(labels):
	unique, counts = np.unique(labels, return_counts=True)
	return dict(zip(unique, counts))

def print_dict(dictionary):
	for key,value in dictionary.items():
		print(key,value,sep=" : ")

def visualise_2D(x_values,y_values,labels=None,class_names=None):
	"""Visualise clusters of selected 2 features"""

	sns.set_style('white')
	sns.set_context('poster')
	sns.set_color_codes()

	plot_kwds = {'alpha' : 0.5, 's' : 30, 'linewidths':0}

	frame = plt.gca()
	frame.axes.get_xaxis().set_visible(False)
	frame.axes.get_yaxis().set_visible(False)

	if labels is None:
		plt.scatter(x_values,y_values,c='b',**plot_kwds)

	else:
		pallete=sns.color_palette('dark',np.unique(labels).max()+1)
		colors=[pallete[x] if x>=0 else (0.0,0.0,0.0) for x in labels]
		plt.scatter(x_values,y_values,c=colors,**plot_kwds)
		legend_entries = [mpatches.Circle((0,0),1,color=x,alpha=0.5) for x in pallete]

		if class_names is None:
			legend_labels = range(len(pallete))

		else:
			legend_labels = ["class "+str(label)+" ( "+str(name)+" )" for label,name in enumerate(class_names)]

		plt.legend(legend_entries,legend_labels,loc='best')
		
	plt.show()


def euclidean_distance(vector1,vector2,squared=False):
	"""calculates euclidean distance between two vectors
	
	Keyword arguments:
	vector1 -- first data point (type: numpy array)
	vector2 -- second data point (type: numpy array)
	squared -- return square of euclidean distance (default: False)
	"""
	euclidean_distance=np.sum((vector1-vector2)**2,dtype=np.float32)

	if squared is False:
		euclidean_distance=np.sqrt(euclidean_distance,dtype=np.float32)

	return euclidean_distance

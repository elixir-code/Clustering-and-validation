import itertools

class external_indices:

	def __init__(self,class_labels,cluster_labels):

		"""Creates class labels and cluster label members and computes yy,yn,ny,nn"""	
		if len(class_labels)!=len(cluster_labels):
			raise Exception("length of class and cluster labels don't match")

		self.class_labels=class_labels
		self.cluster_labels=cluster_labels
		self.n_samples=len(class_labels)

		#compute TP (True-positive:yy), FN (False-negative:yn), FP (False-Positive:ny),TN (True-negative:nn)
		TP, FN, FP, TN = 0,0,0,0

		for i,j in itertools.combinations(range(self.n_samples),2):
			same_class = class_labels[i]==class_labels[j]
			same_cluster = cluster_labels[i]==cluster_labels[j]

			if same_class and same_cluster:
				TP += 1
			elif same_class and not same_cluster:
				FN += 1
			elif not same_class and same_cluster:
				FP += 1
			else:
				TN += 1
		self.TP,self.FN,self.FP,self.TN = TP,FN,FP,TN


	def precision_coefficient(self):
		"""Precision coefficient : fraction of pairs of points correctly grouped together to total pair of point grouped together
		
		The precision is intuitively the ability of the classifier not to label as positive a sample that is negative (sklearn Documentation),i.e., P(g1/g2)

		range : 0 (worst) to 1 (best)
		"""
		return self.TP/(self.TP+self.FP)

	def recall_coefficient(self):
		"""Recall Coefficient : fraction of pairs of points that were correctly grouped togther to that supposed to grouped together according to class labels.

		The recall is intuitively the ability of the classifier to find all the positive samples (Sklearn Documentation),i.e., P(g2/g1)

		range : 0 (worst) to 1 (best)
		"""
		return self.TP/(self.TP+self.FN)

	def f_measure(self):
		"""F-measure : harmonic mean of precision-coefficient and recall-coefficient

		range : 0 (worst) to 1 (best)
		"""
		return 2*self.TP/(2*self.TP+self.FN+self.FP)


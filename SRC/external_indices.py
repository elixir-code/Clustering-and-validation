import itertools
from math import sqrt

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

	def weighted_f_measure(self,beta=1):
		"""F-measure (alpha) : F-measure, which gives beta more weightage to recall over precision
		Reference : https://en.wikipedia.org/wiki/F1_score
					Clustering Indices, Bernard Desgraupes (April 2013)

		range : 0 (worst) to 1 (best)
		"""
		return ((1+beta*beta)*self.TP)/((1+beta*beta)*self.TP+beta*beta*self.FN+self.FP)

	def folkes_mallows_index(self):
		"""Folkes-Mallows (FM) index is the geometric mean of precision and recall
		
		Reference : http://scikit-learn.org/stable/modules/generated/sklearn.metrics.fowlkes_mallows_score.html#sklearn.metrics.fowlkes_mallows_score

		range : 0 (low similarity) to 1 (high similarity)
		"""
		return self.TP/sqrt((self.TP+self.FN)+(self.TP+self.FP))







	def Compute_adjusted_rand_index(labels_true ,labels_pred):
		"""Adjusted Rand Index (SKLEARN)
		Reference:
		http://scikit-learn.org/stable/modules/clustering.html#adjusted-rand-index 
		"""
		return metrics.adjusted_rand_score(labels_true, labels_pred) 


	def Compute_adjusted_mutual_information(labels_true,labels_pred):
		"""Adjusted Mutual information(SKLEARN)
		Reference:
		http://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_mutual_info_score.html#sklearn.metrics.adjusted_mutual_info_score
		"""
		return metrics.adjusted_mutual_info_score(labels_true, labels_pred)  


	def Compute_normalized_mutual_information(labels_true,labels_pred):
		"""normalized Mutual information(SKLEARN)
		Reference:
		http://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html#sklearn.metrics.normalized_mutual_info_score
		"""
		return metrics.normalized_mutual_info_score(labels_true, labels_pred) 


	def Compute_homogeneity_score(labels_true,labels_pred):
		"""homogeneity_score(SKLEARN)
		Reference:
		http://scikit-learn.org/stable/modules/generated/sklearn.metrics.homogeneity_score.html#sklearn.metrics.homogeneity_score
		"""
		return metrics.homogeneity_score(labels_true, labels_pred) 


	def Compute_completness_score(labels_true,labels_pred):
		"""completeness_score(SKLEARN)
		Reference:
		http://scikit-learn.org/stable/modules/generated/sklearn.metrics.completeness_score.html#sklearn.metrics.completeness_score
		"""
		return metrics.completeness_score(labels_true, labels_pred) 

	def Compute_v_measure_score(labels_true, labels_pred):
		"""v_measure_score(SKLEARN)
		Reference:
		http://scikit-learn.org/stable/modules/generated/sklearn.metrics.v_measure_score.html#sklearn.metrics.v_measure_score
		"""
		return metrics.v_measure_score(labels_true, labels_pred) 


	def Compute_fowlkes_mallows_score(labels_true, labels_pred):
		"""fowlkes_mallows_score(SKLEARN)
		Reference:
		http://scikit-learn.org/stable/modules/generated/sklearn.metrics.fowlkes_mallows_score.html#sklearn.metrics.fowlkes_mallows_score
		""" 
		return metrics.fowlkes_mallows_score(labels_true, labels_pred) 

	def Compute_purity(labels_true,labels_pred):
		"""Purity 
		Reference:
		http://www.caner.io/purity-in-python.html
		""" 
		A = np.c_[(labels_pred,labels_true)]
		n_accurate = 0.
		for j in np.unique(A[:,0]):
			z = A[A[:,0] == j, 1]
			x = np.argmax(np.bincount(z))
			n_accurate += len(z[z == x])

		return n_accurate / A.shape[0]

	def Compute_jaccard_co_eff(labels_true, labels_pred):
		"""jaccard co-eff
		chapter 10 -bible of clustering

		Case a: x i and x j belong to the same clusters of C and the same category	of P .
		Case b: x i and x j belong to the same clusters of C but different categories of P .
		Case c: x i and x j belong to different clusters of C but the same category   of P .
		j=a/(a+b+c)
		"""
	  
		return float(self.tp) / (self.tp+self.fn+self.fp)

	def Compute_gamma_statistics(labels_true, labels_pred):
		"""gamma statistics
		chapter 10 -bible of clustering

		Case a: x i and x j belong to the same clusters of C and the same category	of P .
		Case b: x i and x j belong to the same clusters of C but different categories of P .
		Case c: x i and x j belong to different clusters of C but the same category   of P .
		Case 4: x i and x j belong to different clusters of C and different category  of P .
			g=((yy+ny+yn+nn)yy-(yy+yn)(yy+ny))/sqrt((yy+yn) (yy+ny) (nn+yn) (nn+ny))
		"""
		M = self.tp +self.fn + self.fp +self.tn
		m1 = self.tp + self.fn
		m2 = self.tp + self.fp
		numerator = (M*self.tp) - (m1*m2)
		denominator = sqrt(m1 * m2 * (M - m1 ) * (M - m2))
		return numerator/denominator



	def Compute_Czekanowski-Dice_index(self):
		"""Czekanowski-Dice index
		Case 1: x i and x j belong to the same clusters of C and the same category	of P .
		Case 2: x i and x j belong to the same clusters of C but different categories of P .
		Case 3: x i and x j belong to different clusters of C but the same category   of P .
		Case 4: x i and x j belong to different clusters of C and different category  of P .
		
		The Czekanowski-Dice index (aka the Ochiai index) is defined like this:
					c=2yy/(2yy+yn+ny)
		This index is the harmonic mean of the precision and recall coefficients, that
		is to say it is identical to the F-measure
		c=2((precision * recall)/(precision + recall))
		"""
		
		numerator = 2 * self.tp
		denominator = 2 *  self.tp + self.fn + self.fp 

		return numerator/denominator
		
	def Compute_Kulczynski_index(self):
		"""Kulczynski_index
		Case 1: x i and x j belong to the same clusters of C and the same category	of P .
		Case 2: x i and x j belong to the same clusters of C but different categories of P .
		Case 3: x i and x j belong to different clusters of C but the same category   of P .
		Case 4: x i and x j belong to different clusters of C and different category  of P .
		
		The Kulczynski_index   is defined like this:
					KI=1/2((yy(yy+ny)) + (yy/(yy+yn)))
		This index is the arithmetic mean of the precision and recall coefficients:
			KI= 1\2(Precision  + RecaLL)
		"""	 
		

		term1 = self.tp/(self.tp + self.np)
		term2 = self.tp/(self.tp + self.fn)
		KI = 0.5 * (term1 + term2)
		return KI
	def Compute_McNemar_index(self):
		"""McNemar_index
		Case 1: x i and x j belong to the same clusters of C and the same category	of P .
		Case 2: x i and x j belong to the same clusters of C but different categories of P .
		Case 3: x i and x j belong to different clusters of C but the same category   of P .
		Case 4: x i and x j belong to different clusters of C and different category  of P .
		
		The McNemar_index  is defined like this:
					McN=(nn - ny)/sqrt(nn + ny)
		
		"""	 
		numerator = self.tn - self.fp
		denominator = self.tn - self.fp

		return numerator/sqrt(denominator)

	def Compute_Phi_index(self):
		"""Phi index
		Case 1: x i and x j belong to the same clusters of C and the same category	of P .
		Case 2: x i and x j belong to the same clusters of C but different categories of P .
		Case 3: x i and x j belong to different clusters of C but the same category   of P .
		Case 4: x i and x j belong to different clusters of C and different category  of P .
		
		The Phi index is a classical measure of the correlation between two dichotomic
		variables.
		It is defined like this:
			phi =(yy*nn-yn*ny)/((yy+yn)(yy+ny)(yn+nn)(ny+nn))
		
		"""	 
		numerator = (self.tp * self.tn) - (self.fn * self.fp)
		denominator = (self.tp + self.fn)*(self.tp + self.fp)*(self.fn + self.tn)*(self.fp + self.tn)
		return numerator/denominator

	def Compute_Rogers_Tanimoto_index(self):
		"""Rogers_Tanimoto_index
		Case 1: x i and x j belong to the same clusters of C and the same category	of P .
		Case 2: x i and x j belong to the same clusters of C but different categories of P .
		Case 3: x i and x j belong to different clusters of C but the same category   of P .
		Case 4: x i and x j belong to different clusters of C and different category  of P .
		
		The Rogers-Tanimoto index is defined like this:
		RT = (yy + nn)/(yy + nn + 2(yn+ny))
		"""
		

		numerator = (self.tp + self.tn)
		denominator= self.tp + self.tn + (2 *(self.fn+self.fp) )
		return numerator/denominator

	def Compute_Russel_Rao_index(self):
		"""Russel-Rao index
		Case 1: x i and x j belong to the same clusters of C and the same category	of P .
		Case 2: x i and x j belong to the same clusters of C but different categories of P .
		Case 3: x i and x j belong to different clusters of C but the same category   of P .
		Case 4: x i and x j belong to different clusters of C and different category  of P .
		The Russel-Rao index measures the proportion of concordances between the two	partitions. 
		The Russel-Rao index is defined like this:
			RR=yy/(yy+yn+ny+nn)
		"""
		
		denominator = self.tp + self.fn + self.fp + self.tn
		return self.tp/denominator

	def Compute_Sokal_Sneath_index1(self):
		"""Sokal-Sneath indice
		Case 1: x i and x j belong to the same clusters of C and the same category	of P .
		Case 2: x i and x j belong to the same clusters of C but different categories of P .
		Case 3: x i and x j belong to different clusters of C but the same category   of P .
		Case 4: x i and x j belong to different clusters of C and different category  of P .
		
		ss1= yy/(yy+2(yn + ny))
		"""
		
		denominator =self.tp /(self.tp + (2 * ( self.fn + self.fp)))
		return self.tp/denominator



	def Compute_Sokal_Sneath_index2(self):
		"""Sokal-Sneath indice
		Case 1: x i and x j belong to the same clusters of C and the same category	of P .
		Case 2: x i and x j belong to the same clusters of C but different categories of P .
		Case 3: x i and x j belong to different clusters of C but the same category   of P .
		Case 4: x i and x j belong to different clusters of C and different category  of P .
		
		ss2= (yy+nn)/(yy+nn+(1/2)(yn+ny))
		"""
	   
		numerator =self.tp + self.tn
		denominator= self.tp + self.tn + (0.5 * (self.fn + self.fp))
		return numerator/denominator

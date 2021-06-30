import numpy as np
from scipy.spatial import distance
from scipy.special import rel_entr


class distance_functions:

	def __init__(self, estimation_fold):
		self.estimation_fold = estimation_fold
		# output: list of lists of distances in each fold:
		# [[eucl qrs],[rel_ent qrs],[eucl qws],[rel_ent qws],[eucl prbs],[rel_ent prbs],[eucl pwas],[rel_ent pwas]]
		self.distances = []

	def compute_distances(self):
		# qr, qw, prbeta, pwalpha
		qrs = []
		qws = []
		prbs = []
		pwas = []

		# max_it_em number of matrices inside estimation_fold and maximization_fold for the current Fold
		for est in self.estimation_fold:
			qrs.append(est[0])			# qr
			qws.append(est[1])			# qw
			prbs.append(est[2])			# prb
			pwas.append(est[3])			# pwa

		compute = [qrs, qws, prbs, pwas]
		# Compute Euclidean Distance and Relative Entropy (em-1, em)
		for i in compute:
			self.euclidean(i)
			self.relative_entropy(i)

		return self.distances

	# SPECIAL CASE for SVC
	def extra_compute_distances_SVC(self):
		self.euclidean(self.estimation_fold)
		self.relative_entropy(self.estimation_fold)

		return self.distances

	# weights_matrix
	def euclidean(self, weights_matrices):
		euclidean_list = []
		#num_rows = len(weights_matrices[0])
		#previous_matrix = np.zeros((num_rows, 2))
		previous_matrix = weights_matrices[0]

		for em_matrix in weights_matrices[1:]:
			dist = []
			for prev, actual in zip(previous_matrix, em_matrix):
				dist.append(distance.euclidean(prev, actual))
			mean_dist = np.mean(dist)
			euclidean_list.append(mean_dist)
			previous_matrix = em_matrix

		self.distances.append(euclidean_list)

	def relative_entropy(self, weights_matrices):
		rel_entropy_list = []
		#num_rows = len(weights_matrices[0])
		previous_matrix = weights_matrices[0]

		for em_matrix in weights_matrices[1:]:
			dist = []
			for prev, actual in zip(previous_matrix, em_matrix):
				temp = 0
				for el in range(2):
					temp += rel_entr(actual[el], prev[el])
				dist.append(temp)
			mean_dist = np.mean(dist)
			rel_entropy_list.append(mean_dist)
			previous_matrix = em_matrix

		self.distances.append(rel_entropy_list)

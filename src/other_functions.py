import pickle


class other_functions:

	def __init__(self, estimation_fold, num_fold, math_model, k_cv, max_it_em):
		self.estimation_fold = estimation_fold
		self.num_fold = num_fold
		self.math_model = math_model
		self.k_cv = k_cv
		self.max_it_em = max_it_em


	def save_weigths_probs(self):
		# qr, qw, prbeta, pwalpha
		qrs = []
		qws = []
		prbs = []
		pwas = []

		# max_it_em number of matrices inside estimation_foldfor the current Fold
		for est in self.estimation_fold:
			qrs.append(est[0])			# qr
			qws.append(est[1])			# qw
			prbs.append(est[3])			# prb
			pwas.append(est[4])			# pwa

		to_save = [qrs, qws, prbs, pwas]
		type_to_save = ['weigths', 'weigths', 'probs', 'probs']
		text_to_save = ['qrs', 'qws', 'prbs', 'pwas']

		for ts, ty, te in zip(to_save, type_to_save, text_to_save):
			with open("results/ehu_experiment_" + ty + "_" + te + "_" + self.math_model + "_f" + str(self.num_fold) +
					  "_k" + str(self.k_cv) + "_ems" + str(self.max_it_em) + ".pickle", 'wb') \
					as file_pckl:
				pickle.dump(ts, file_pckl, pickle.HIGHEST_PROTOCOL)

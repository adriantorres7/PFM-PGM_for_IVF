import numpy as np
import copy
from itertools import combinations
from src.datasets import eu_data_ehu

class eu_model_ehu:
    def __init__(self, data: eu_data_ehu, classifier_embryos,gen_q_w=lambda d: np.ones(d) * 0.5):

        self.eu_data_ehu = data
        self.classifier_embryos = classifier_embryos
        self.gen_q_w = gen_q_w



    def initialize(self):

        self.q_w = np.zeros((self.eu_data_ehu.num_embryos, 2))
        self.q_w[:,0] = self.gen_q_w(self.eu_data_ehu.num_embryos)
        self.q_w[:,1] = 1. - self.q_w[:,0]

        self.p_w_by_alpha = np.ones((self.eu_data_ehu.num_embryos, 2))
        fixed_q_w = []
        for i_c in np.arange(self.eu_data_ehu.num_cycles):
            if self.eu_data_ehu.num_emb_transf_per_cycle[i_c] == self.eu_data_ehu.num_emb_implanted_per_cycle[i_c]:
                for i_e in self.eu_data_ehu.cycle_has_trans_embryos[i_c]:
                    fixed_q_w.append(i_e)
                    self.q_w[i_e, :] = np.array([0., 1.])
            if self.eu_data_ehu.num_emb_implanted_per_cycle[i_c]==0:
                for i_e in self.eu_data_ehu.cycle_has_trans_embryos[i_c]:
                    fixed_q_w.append(i_e)
                    self.q_w[i_e, :] = np.array([1., 0.])
        self.unfixed_q_w = np.array([i_e for i_e in np.arange(self.eu_data_ehu.num_embryos)
                                     if i_e not in fixed_q_w])
        print('Fixed: ',len(fixed_q_w))
        self.duplicated_embryos = np.repeat(self.eu_data_ehu.embryos, 2, axis=0)
        self.duplicated_embryos_labels = np.tile([0, 1], self.eu_data_ehu.num_embryos)

        return self.q_w, self.p_w_by_alpha


    # Estimates the probability distribution for the model
    # Compute new weights taking into account all the system and the hidden variables
    def estimations(self):
        self.p_w_by_alpha = self.fit_class_embryos.predict_proba(self.eu_data_ehu.embryos)
        
        self.q_w[self.unfixed_q_w,:]=self.p_w_by_alpha[self.unfixed_q_w,:]
        
        return self.q_w,self.p_w_by_alpha
    # Embryos' model learning with computed weigths (q_w)
    def fit(self):
        self.fit_class_embryos = self.classifier_embryos.fit(self.duplicated_embryos,
                                                             self.duplicated_embryos_labels,
                                                             sample_weight=self.q_w.flatten())
        return self.fit_class_embryos
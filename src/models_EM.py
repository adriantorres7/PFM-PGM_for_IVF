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
        fixed_cycles = []
        for i_c in np.arange(self.eu_data_ehu.num_cycles):
            if self.eu_data_ehu.num_emb_transf_per_cycle[i_c] == self.eu_data_ehu.num_emb_implanted_per_cycle[i_c]:
                fixed_cycles.append(i_c)
                for i_e in self.eu_data_ehu.cycle_has_trans_embryos[i_c]:
                    fixed_q_w.append(i_e)
                    self.q_w[i_e, :] = np.array([0., 1.])
            elif self.eu_data_ehu.num_emb_implanted_per_cycle[i_c]==0:
                fixed_cycles.append(i_c)
                for i_e in self.eu_data_ehu.cycle_has_trans_embryos[i_c]:
                    fixed_q_w.append(i_e)
                    self.q_w[i_e, :] = np.array([1., 0.])
        print('Fixed!: ',len(fixed_q_w))
        self.unfixed_q_w = np.array([i_e for i_e in np.arange(self.eu_data_ehu.num_embryos)
                                     if i_e not in fixed_q_w])
        self.unfixed_cycles = np.array([i_c for i_c in np.arange(self.eu_data_ehu.num_cycles)
                                        if i_c not in fixed_cycles])

        self.duplicated_embryos = np.repeat(self.eu_data_ehu.embryos, 2, axis=0)
        self.duplicated_embryos_labels = np.tile([0, 1], self.eu_data_ehu.num_embryos)

        return self.q_w, self.p_w_by_alpha


    # Estimates the probability distribution for the model
    # Compute new weights taking into account all the system and the hidden variables
    def estimations(self):
        self.p_w_by_alpha = self.fit_class_embryos.predict_proba(self.eu_data_ehu.embryos)
        
        self.q_w[self.unfixed_q_w,:]=self.p_w_by_alpha[self.unfixed_q_w,:]
        self.compute_qwes()

        return self.q_w, self.p_w_by_alpha

    # UPDATE weights according to EQUATION 4.4
    def compute_qwes(self):
        self.q_w = copy.deepcopy(self.q_w)

        for i_c in self.unfixed_cycles:
            lp=self.eu_data_ehu.num_emb_implanted_per_cycle[i_c]
            b=self.eu_data_ehu.num_emb_transf_per_cycle[i_c]
            comb=list(combinations(self.eu_data_ehu.cycle_has_trans_embryos[i_c], lp))
            set_embryos=set(self.eu_data_ehu.cycle_has_trans_embryos[i_c])
            p_comb=np.ones(len(comb))
            for i_co,co in enumerate(comb):
                for i_e in co:
                    p_comb[i_co]*=self.p_w_by_alpha[i_e,1]
                for i_e in set_embryos-set(co):
                    p_comb[i_co]*=self.p_w_by_alpha[i_e,0]
            den=sum(p_comb)
            for i_e in self.eu_data_ehu.cycle_has_trans_embryos[i_c]:
                aux=0
                for i_co,co in enumerate(comb):
                    if i_e in co:
                        aux+=p_comb[i_co]
                self.q_w[i_e,1]=aux/den
                self.q_w[i_e,0]=1-aux/den



    # Embryos' model learning with computed weigths (q_w)
    def fit(self):
        self.fit_class_embryos = self.classifier_embryos.fit(self.duplicated_embryos,
                                                             self.duplicated_embryos_labels,
                                                             sample_weight=self.q_w.flatten())
        return self.fit_class_embryos
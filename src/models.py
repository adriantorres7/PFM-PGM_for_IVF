import numpy as np
import copy
from itertools import combinations
from src.datasets import eu_data_ehu


class eu_model_ehu:
    def __init__(self, data: eu_data_ehu, classifier_embryos, classifier_cycles,
                 gen_q_r=lambda d: np.ones(d) * 0.5, gen_q_w=lambda d: np.ones(d) * 0.5):

        self.eu_data_ehu = data
        self.classifier_embryos = classifier_embryos
        self.classifier_cycles = classifier_cycles
        self.fit_thetas = np.ones((2, 2))
        self.fit_thetas[0, 1] = 0
        self.fit_thetas[1, 0] = 0
        self.gen_q_r = gen_q_r
        self.gen_q_w = gen_q_w



    def initialize(self):

        self.q_r = np.zeros((self.eu_data_ehu.num_cycles, 2))
        self.q_r[:,0] = self.gen_q_r(self.eu_data_ehu.num_cycles)
        self.q_r[:,1] = 1. - self.q_r[:,0]
        self.p_r_by_beta = np.ones((self.eu_data_ehu.num_cycles, 2))
        fixed_q_r = []
        for i_c, num_implt in enumerate(self.eu_data_ehu.num_emb_implanted_per_cycle):
            if num_implt > 0:
                fixed_q_r.append(i_c)
                self.q_r[i_c, :] = np.array([0., 1.])
        self.unfixed_q_r = np.array([i for i in np.arange(self.eu_data_ehu.num_cycles)
                                     if i not in fixed_q_r])

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
        self.unfixed_q_w = np.array([i_e for i_e in np.arange(self.eu_data_ehu.num_embryos)
                                     if i_e not in fixed_q_w])

        self.i_vects = []
        for i_c, num_imp in enumerate(self.eu_data_ehu.num_emb_implanted_per_cycle):
            self.i_vects.append(list(combinations(self.eu_data_ehu.cycle_has_trans_embryos[i_c], num_imp)))

        self.q_i = []
        fixed_q_i = []
        for i_c in np.arange(self.eu_data_ehu.num_cycles):
            n_vects = len(self.i_vects[i_c])
            self.q_i.append(np.ones(n_vects)/float(n_vects))
            if n_vects == 1:
                fixed_q_i.append(i_c)
        self.unfixed_q_i = np.array([i_c for i_c in np.arange(self.eu_data_ehu.num_cycles)
                                     if i_c not in fixed_q_i])

        self.duplicated_cycles = np.repeat(self.eu_data_ehu.cycles, 2, axis=0)
        self.duplicated_cycles_labels = np.tile([0, 1], self.eu_data_ehu.num_cycles)
        self.duplicated_embryos = np.repeat(self.eu_data_ehu.embryos, 2, axis=0)
        self.duplicated_embryos_labels = np.tile([0, 1], self.eu_data_ehu.num_embryos)

        return self.q_r, self.q_w, self.p_r_by_beta, self.p_w_by_alpha


    # Estimates the probability distribution for each of the models (cycles and embryos)
    # Compute new weights taking into account all the system and the hidden variables
    def estimations(self):
        self.p_r_by_beta = self.fit_class_cycles.predict_proba(self.eu_data_ehu.cycles)
        self.p_w_by_alpha = self.fit_class_embryos.predict_proba(self.eu_data_ehu.embryos)

        self.compute_qrs()
        self.compute_qwes()
        self.compute_qics()

        return self.q_r, self.q_w, self.p_r_by_beta, self.p_w_by_alpha

    '''
    compute_qrs: EQUATION 3.16
    A few simplifications:
     - q_r needs to be updated (unfixed_q_r) only for failed cycles: there exists a single i vector 
       where everything is negative (i_e = 0 for all e)
     - In sum_i prod_e sum_w: sum_w (theta_w*alpha_w) = 1 always if the embryos were not transfered! This
       means that the vectors i can be reduced to transfered embryos
     - As a consecuence, when r=0, the whole sum_i prod_e sum_w(theta*alpha)= 1, and then q(r=0) ~ p(r=0 | beta) 
    '''
    def compute_qrs(self):
        self.q_r = copy.deepcopy(self.q_r)

        for c in self.unfixed_q_r:
            # Formula loop
            # qr per a aquest cicle
            qr = self.p_r_by_beta[c, :].copy()
            # Only consider r=1. Only consider a single i vector (all values to 0). Only consider transferred embryos.
            # qr[1] = 1.
            for e in self.eu_data_ehu.cycle_has_trans_embryos[c]:
                p_i_e = np.array([self.fit_thetas[0, 0],
                                  self.fit_thetas[1, 0]])
                qr[1] *= np.sum(p_i_e * self.p_w_by_alpha[e,:])
            # print('qr: ',qr)
            self.q_r[c, :] = qr / np.sum(qr)

        #print('qrs:\n {}'.format(self.q_r))

    '''
    compute_qwes: EQUATION 3.17
    A few simplifications:
     - In sum_i prod_e sum_w: sum_w (theta_w*alpha_w) = 1 always if the embryos were not transfered! This
       means that the vectors i can be reduced to transfered embryos
    '''
    def compute_qwes(self):
        self.q_w = copy.deepcopy(self.q_w)

        for e in self.unfixed_q_w:
            c = self.eu_data_ehu.embryo_belong_to_cycle[e]

            for v_w in [0,1]:
                p_r = np.zeros(2)

                for v_r in [0, 1]:
                    for i_iv, i_vect in enumerate(self.i_vects[c]):  # per a cada vector ic
                        aux_b = self.fit_thetas[v_r * v_w * int(e in self.eu_data_ehu.cycle_has_trans_embryos[c]),
                                                int(e in i_vect)]

                        for ep in self.eu_data_ehu.cycle_has_trans_embryos[c]:
                            if e != ep:
                                p_i_ep = np.array([self.fit_thetas[0, int(ep in i_vect)],
                                                   self.fit_thetas[v_r, int(ep in i_vect)]])
                                aux_b *= np.sum(self.p_w_by_alpha[ep, :] * p_i_ep)
                        p_r[v_r] += aux_b
                    p_r[v_r] *= self.p_w_by_alpha[e, v_w]
                self.q_w[e, v_w] = np.sum(self.p_r_by_beta[c, :] * p_r)
            self.q_w[e, :] /= np.sum(self.q_w[e,:])

    '''
    compute_qwes: EQUATION 3.18
    A few simplifications:
     - In sum_i prod_e sum_w: sum_w (theta_w*alpha_w) = 1 always if the embryos was not transfered! This
       means that the vectors i can be reduced to transfered embryos
    '''
    def compute_qics(self):
        self.q_i = copy.deepcopy(self.q_i)

        for c in self.unfixed_q_i:
            for i_iv, i_vect in enumerate(self.i_vects[c]):  # per a cada vector ic
                p_r = np.ones(2)
                for v_r in [0, 1]:
                    for e in self.eu_data_ehu.cycle_has_trans_embryos[c]:
                        p_i_e = np.array([self.fit_thetas[0, int(e in i_vect)],
                                           self.fit_thetas[v_r, int(e in i_vect)]])
                        p_r[v_r] *= np.sum(self.p_w_by_alpha[e, :] * p_i_e)

                self.q_i[c][i_iv] = np.sum(self.p_r_by_beta[c, :] * p_r)
            self.q_i[c][:] /= np.sum(self.q_i[c][:])


    # Cycles' model learning with computed weigths (q_r)
    # Embryos' model learning with computed weigths (q_w)
    # Compute theta1
    def fit(self):
        self.fit_class_cycles = self.classifier_cycles.fit(self.duplicated_cycles,
                                                           self.duplicated_cycles_labels,
                                                           sample_weight=self.q_r.flatten())
        self.fit_class_embryos = self.classifier_embryos.fit(self.duplicated_embryos,
                                                             self.duplicated_embryos_labels,
                                                             sample_weight=self.q_w.flatten())
        self.mle_theta1()

        return self.fit_class_embryos, self.fit_class_cycles, self.fit_thetas

    '''
    Maximization step for each cycle: EQUATION 3.21
    '''
    def mle_theta1(self):
        num = 0
        den = 0
        self.fit_thetas = copy.deepcopy(self.fit_thetas)

        for c in self.unfixed_q_i:
            for i_iv, i_vect in enumerate(self.i_vects[c]):  # per a cada vector ic
                for e in self.eu_data_ehu.cycle_has_trans_embryos[c]:  # using cycle_has_trans_embryos here as theta1 is forced
                    aux = self.q_i[c][i_iv] * self.q_r[c, 1] * self.q_w[e, 1]
                    den += aux
                    if e in i_vect:
                        num += aux

        #theta1
        self.fit_thetas[1, 1] = num / den
        #1-theta
        self.fit_thetas[1, 0] = 1 - self.fit_thetas[1, 1]


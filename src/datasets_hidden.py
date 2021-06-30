import pandas as pd
import numpy as np
from sklearn import preprocessing

class eu_data_ehu:
    def __init__(self, file_cycles="data/eh_cycles.csv", file_embryos='data/eh_embryos.csv', em=1):
        # READ DATA
        self.data_cycles = pd.read_csv(file_cycles)
        self.data_embryos = pd.read_csv(file_embryos)
        
        #ONE-HOT ENCODING
        self.data_cycles = pd.get_dummies(self.data_cycles, columns=['Indicac', 'EmbPrev', 'AboPrev', 'caSemen',
                                                                     'Protocol', 'Estimul'], drop_first=False)

        self.embryo_quality = self.data_embryos["CALIDAD+2"].values
        self.data_embryos = pd.get_dummies(self.data_embryos,
                                           columns=['Tecnica', 'Vac', 'REL', 'Epv', 'CP', 'Z', 'simet+2', 'ZP+2',
                                                    'vac+2', 'multiNuc+2'],
                                           drop_first=False)        
        
        #CREATION OF INTERNAL VARIABLES
        self.num_cycles = self.data_cycles.shape[0]
        self.num_embryos = self.data_embryos.shape[0]
        self.embryo_was_transfered = (self.data_embryos["Transfer"].values == "Si").astype(int)
        self.embryo_was_implanted = -1*np.ones(self.num_embryos, dtype=int)
        self.embryo_belong_to_cycle = self.data_embryos["CodigoCiclo"].values
        self.cycle_has_embryos = [None]*self.num_cycles
        self.cycle_has_trans_embryos = [None]*self.num_cycles
        self.num_emb_transf_per_cycle = np.zeros(self.num_cycles, dtype=int)
        for i in np.arange(self.num_cycles):
            self.cycle_has_embryos[i] = (np.where(self.embryo_belong_to_cycle == i)[0]).astype(int)
            aux = np.where(self.data_embryos["Transfer"].iloc[self.cycle_has_embryos[i]].values == "Si")[0]
            self.cycle_has_trans_embryos[i] = self.cycle_has_embryos[i][aux]
            self.num_emb_transf_per_cycle[i] = len(self.cycle_has_trans_embryos[i])
        self.num_emb_implanted_per_cycle = (self.num_emb_transf_per_cycle * self.data_cycles['TasaExito'].values).astype(int)

        # embryo_was_implanted: 0 if no embryo implanted, 1 if all implanted; -1 otherwise
        for c in np.arange(self.num_cycles):
            if self.num_emb_implanted_per_cycle[c] == 0:
                self.embryo_was_implanted[self.cycle_has_trans_embryos[c]] = 0
            elif self.num_emb_implanted_per_cycle[c] == self.num_emb_transf_per_cycle[c]:
                self.embryo_was_implanted[self.cycle_has_trans_embryos[c]] = 1
            else:
                self.embryo_was_implanted[self.cycle_has_trans_embryos[c]] = -1

        unique, counts = np.unique(self.embryo_was_implanted, return_counts=True)
        print('--- unique self.embryo_was_implanted datasets.py: {}'.format(dict(zip(unique, counts))))
        
        # REMOVING FEATURES AND STANDARDIZING
        aux = self.data_cycles.copy()
        for col in ['Codigo','AMH','nEmbTrans','TasaExito','Indicac_desconocido','Indicac_otros','EmbPrev_No',
                    'AboPrev_No']:
            del aux[col]
        self.cycles = aux.values
        self.cycles = preprocessing.scale(self.cycles)

        aux = self.data_embryos.copy()
        for col in ['CodigoCiclo', 'CodigoOvoc','TasaExito','Vac_No','REL_No','Epv_Normal','CP_Normal',
                    'simet+2_No','ZP+2_Normal', 'vac+2_No', 'multiNuc+2_No', 'Transfer', 'Vitrificado','CALIDAD+2']:
            del aux[col]
        self.embryos = aux.values
        self.embryos = preprocessing.scale(self.embryos)

    # METHOD TO REMOVE CYCLES CORRESPONDING TO GIVEN SET OF INDICES
    def remove_cycles(self, inds_to_remove):
        inds_to_remove[::-1].sort()
        inds_embryos_to_remove = [i_e for i_c in inds_to_remove for i_e in self.cycle_has_embryos[i_c]]
        inds_embryos_to_remove[::-1].sort()

        self.num_cycles -= len(inds_to_remove)
        self.cycles = np.delete(self.cycles, inds_to_remove, axis=0)
        self.data_cycles.drop(self.data_cycles.index[inds_to_remove],inplace=True)
        self.num_emb_transf_per_cycle = np.delete(self.num_emb_transf_per_cycle, inds_to_remove)
        self.num_emb_implanted_per_cycle = np.delete(self.num_emb_implanted_per_cycle, inds_to_remove)
        for i_c in inds_to_remove:
            del self.cycle_has_embryos[i_c]
            del self.cycle_has_trans_embryos[i_c]

        self.num_embryos -= len(inds_embryos_to_remove)
        self.embryos = np.delete(self.embryos, inds_embryos_to_remove, axis=0)
        self.data_embryos.drop(self.data_embryos.index[inds_embryos_to_remove],inplace=True)
        self.embryo_was_transfered = np.delete(self.embryo_was_transfered, inds_embryos_to_remove)
        self.embryo_was_implanted = np.delete(self.embryo_was_implanted, inds_embryos_to_remove)
        self.embryo_quality = np.delete(self.embryo_quality,inds_embryos_to_remove)
        self.embryo_belong_to_cycle = np.delete(self.embryo_belong_to_cycle, inds_embryos_to_remove)

        unique, counts = np.unique(self.embryo_was_implanted, return_counts=True)
        print('--- unique self.embryo_was_implanted dins remove_cycles.py: {}'.format(dict(zip(unique, counts))))

        # update indices
        for i in np.arange(len(self.embryo_belong_to_cycle)):
            self.embryo_belong_to_cycle[i] -= len(np.where(inds_to_remove < self.embryo_belong_to_cycle[i])[0])
        for c in self.cycle_has_embryos:
            for i in np.arange(len(c)):
                c[i] -= len(np.where(inds_embryos_to_remove < c[i])[0])
        for c in self.cycle_has_trans_embryos:
            for i in np.arange(len(c)):
                c[i] -= len(np.where(inds_embryos_to_remove < c[i])[0])

    # METHOD TO CREATE A COPY OF THE DATASET
    def copy(self):
        return self.__copy__()

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)

        result.data_cycles = self.data_cycles.copy()
        result.data_embryos = self.data_embryos.copy()
        result.num_cycles = self.num_cycles
        result.num_embryos = self.num_embryos
        result.embryo_was_transfered = np.copy(self.embryo_was_transfered)
        result.embryo_was_implanted = np.copy(self.embryo_was_implanted)
        result.embryo_quality = np.copy(self.embryo_quality)
        result.embryo_belong_to_cycle = np.copy(self.embryo_belong_to_cycle)
        result.num_emb_transf_per_cycle = np.copy(self.num_emb_transf_per_cycle)
        result.num_emb_implanted_per_cycle = np.copy(self.num_emb_implanted_per_cycle)
        result.cycles = np.copy(self.cycles)
        result.embryos = np.copy(self.embryos)

        result.cycle_has_embryos = [None] * self.num_cycles
        result.cycle_has_trans_embryos = [None] * self.num_cycles
        for i in np.arange(self.num_cycles):
            result.cycle_has_embryos[i] = self.cycle_has_embryos[i].copy()
            result.cycle_has_trans_embryos[i] = self.cycle_has_trans_embryos[i].copy()

        return result

from src.datasets import eu_data_ehu
from src.EM import EM
from src.models import eu_model_ehu
from src.distance_functions import distance_functions
from src.other_functions import other_functions
import time, datetime
import pickle
import numpy as np
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.metrics import roc_auc_score


##### Functions to compute the METRICS to evaluate the model
def lp_loss(bag_sizes, real_lps, pred_labels):      # level proportion loss
    i = 0
    loss = 0
    for i_b, b in enumerate(bag_sizes):
        pred_lps = np.sum(pred_labels[i:(i+b)])
        loss += np.abs(pred_lps-real_lps[i_b])
        i += b
    loss /= float(np.sum(bag_sizes))
    return loss

def ps_recall(pu_labels, predicted_labels):         # pseudo-recall (cycles)
    num=0.
    den=0.
    for i_l, pu_lab in enumerate(pu_labels):
        if pu_lab == 1:
            den += 1.
            if pu_lab == predicted_labels[i_l]:
                num += 1.
    return num/den

def prop_predicted_pos(predicted_labels):           # prop. of embryos predicted positives
    return np.sum(predicted_labels)/float(len(predicted_labels))

def logloss(bag_sizes, real_lps, pred_proba): # Alternative formulation of the logloss (not covered in the memory)
    i=0
    loss=0
    for i_b, b in enumerate(bag_sizes):
        prop=real_lps[i_b]/b
        mean=np.mean(pred_proba[i:(i+b)])
        log_e=prop*np.log(mean)+(1-prop)*np.log(1-mean)
        loss+=b*log_e
    loss /= float(np.sum(bag_sizes))
    return -loss
def loglike(bag_sizes, real_lps, pred_proba): #Negative log-likelihood
    i=0
    like=0
    for i_b, b in enumerate(bag_sizes):
        lp=real_lps[i_b]
        probs=pred_proba[i:(i+b)]
        comb=list(combinations(range(b), lp))
        total=0
        for c in comb:
            aux=1
            for i_c in c:
                aux*=probs[i_c]
            for i_c in set(range(b))-set(c):
                aux*=(1-probs[i_c])
            total+=aux
        like+=np.log(total)
    like /= float(np.sum(bag_sizes))
    return -like



math_model_list = ['LR','RF200', 'GBOOST','ETREES']


partial_times = []

qr_euclidean_distances_exps = []
qr_rel_entropy_distances_exps = []
qw_euclidean_distances_exps = []
qw_rel_entropy_distances_exps = []
prb_euclidean_distances_exps = []
prb_rel_entropy_distances_exps = []
pwa_euclidean_distances_exps = []
pwa_rel_entropy_distances_exps = []

#################################

# START experiments
start_time = time.time()
timestamp = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
print('dia i hora: {}'.format(timestamp))

# Iterate over all models.
for idx_m, m in enumerate(math_model_list):
    print('\n-- EXPERIMENT {} --'.format(m))
    data = eu_data_ehu()
    data_clean = eu_data_ehu()
    math_model = math_model_list[idx_m]

    k_cv = 5 # Number of Folds
    max_it_em =100 # Maximum number of EM iterations
    num_execs = 10 # Number of different EM initializations

    np.random.seed(7)


    # The partition is made at cycle level
    l_insts = np.arange(data.num_cycles)
    kf = KFold(n_splits=k_cv)               # Cross-validation with k_cv folds

    l_metrics = []
    distances_folds = []
    pred_probs = {}
    pred_probs['embryo']=[]
    pred_probs['cycle']=[]
    pred_probs['full']=[]
    real_info={}
    real_info['implanted']=[]
    real_info['quality']=[]
    real_info['transfered']=[]
    l_auc_roc = []

    print('Math model: {}'.format(math_model))
    print('Cross-Validation Folds: {}'.format(k_cv))
    print('EM max iterations: {}'.format(max_it_em))
    print('EM full executions: {}'.format(num_execs))

    nfold = 0
    for train, test in kf.split(l_insts):
        print(' Fold {}'.format(nfold))

        tr_data = data.copy()
        tr_data_clean = data_clean.copy()
        tr_data.remove_cycles(test)
        tr_data_clean.remove_cycles(test)
        ts_data = data.copy()
        ts_data_clean = data_clean.copy()
        ts_data.remove_cycles(train)
        ts_data_clean.remove_cycles(train)


        # Default case
        cl_embryos = LogisticRegression(random_state=0)
        cl_cycles = LogisticRegression(random_state=0)
        model_final = eu_model_ehu(tr_data, cl_embryos, cl_cycles)
        em = EM(model_final, max_its=max_it_em)

        loglike_iteration_final = 100
        best_iteration = 0
        best_estimation_record = []

        # Iterate over the different initializations of the EM
        for i in range(num_execs):
            print('   execution {}'.format(i))
            
            # MODEL SELECTION. We consider the same type of model for both embryo and cycle. Default= LR
            if (math_model == 'LR'):
                cl_embryos = LogisticRegression(random_state=0)
                cl_cycles = LogisticRegression(random_state=0)
            elif (math_model == 'RF200'):
                cl_embryos = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=0)
                cl_cycles = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=0)
            elif (math_model == 'GBOOST'):
                cl_embryos = GradientBoostingClassifier(random_state=0)
                cl_cycles = GradientBoostingClassifier(random_state=0)
            elif (math_model == 'ETREES'):
                cl_embryos = ExtraTreesClassifier(n_estimators=100, random_state=0)
                cl_cycles = ExtraTreesClassifier(n_estimators=100, random_state=0)
            else:
                print('!!! ERROR: NO Mathematical model defined !!!')

            # CREATE AN INSTANCE OF THE MODEL
            model = eu_model_ehu(tr_data, cl_embryos, cl_cycles,
                                 gen_q_r=lambda d: np.random.random(d), gen_q_w=lambda d: np.random.random(d))
            
            # CREATE AND RUN AN INSTANCE OF THE EM ALGORITHM
            em = EM(model, max_its=max_it_em)
            em.run()

            # PREDICTIONS for the TRAIN set
            cycles_pred_labs = model.fit_class_cycles.predict(tr_data.cycles)       # Cycles' label prediction
            cycles_pred_proba = model.fit_class_cycles.predict_proba(tr_data.cycles)[:,1]
            embryos_pred_labs = model.fit_class_embryos.predict(tr_data.embryos)    # Embryos' label prediction
            embryos_pred_proba = model.fit_class_embryos.predict_proba(tr_data.embryos)[:,1]

            # Transforming predictions to adapt to metrics        
            bag_sizes = np.array([len(c) for c in tr_data.cycle_has_trans_embryos])
            cycles_lps = tr_data.num_emb_implanted_per_cycle
            transf_embryos_pred_proba = np.array([embryos_pred_proba[i_e]*cycles_pred_proba[i_c]*model.fit_thetas[1,1]
                                             for i_c,c in enumerate(tr_data.cycle_has_trans_embryos) for i_e in c])
            transf_embryos_pred_labs = np.array([embryos_pred_labs[i_e]
            	for c in tr_data.cycle_has_trans_embryos for i_e in c])

            # COMPUTE THE NEGATIVE LOG-LIKELIHOOD FOR THE CURRENT EM EXECUTION
            loglike_iteration_actual = loglike(bag_sizes, cycles_lps, transf_embryos_pred_proba)

            # If the current loglike is lower than the best one yet, we update it
            if loglike_iteration_actual < loglike_iteration_final:
                loglike_iteration_final = loglike_iteration_actual
                best_iteration = i
                model_final = model
                best_estimation_record = em.estimation_record

        # Model with the lowest negative log-likelihood
        model = model_final

        # PREDICTIONS for the TEST set        
        cycles_pred_labs = model.fit_class_cycles.predict(ts_data.cycles)       # Cycles' label prediction
        cycles_pred_proba = model.fit_class_cycles.predict_proba(ts_data.cycles)[:,1]
        embryos_pred_labs = model.fit_class_embryos.predict(ts_data.embryos)    # Embryos' label prediction
        embryos_pred_proba = model.fit_class_embryos.predict_proba(ts_data.embryos)[:,1]

        # Predictions of the corresponding cycle for each embryo
        total_cycles_pred_proba = np.array([cycles_pred_proba[i_c]
                                             for i_c,c in enumerate(ts_data.cycle_has_embryos) for i_e in c])        
        
        # Embryo info to make plots separating by different factors
        pred_probs['embryo'].extend(embryos_pred_proba)
        pred_probs['cycle'].extend(total_cycles_pred_proba)
        pred_probs['full'].extend(embryos_pred_proba*total_cycles_pred_proba*model.fit_thetas[1,1])
        real_info['implanted'].extend(ts_data.embryo_was_implanted)
        real_info['quality'].extend(ts_data.embryo_quality)
        real_info['transfered'].extend(ts_data.embryo_was_transfered)
        # to compute ps_recall
        cycles_pu_labels = (ts_data.num_emb_implanted_per_cycle > 0).astype(float)
        cycles_pu_labels[np.where(cycles_pu_labels == 0)] = np.nan

        # Transforming predictions to adapt to metrics        
        bag_sizes = np.array([len(c) for c in ts_data.cycle_has_trans_embryos])
        cycles_lps = ts_data.num_emb_implanted_per_cycle
        transf_embryos_pred_proba = np.array([embryos_pred_proba[i_e]*cycles_pred_proba[i_c]*model.fit_thetas[1,1]
                                             for i_c,c in enumerate(ts_data.cycle_has_trans_embryos) for i_e in c])
        transf_embryos_pred_labs = np.array([embryos_pred_labs[i_e]
            for c in ts_data.cycle_has_trans_embryos for i_e in c])

        # COMPUTE METRICS (except AUC)
        metrics = [ps_recall(cycles_pu_labels, cycles_pred_labs),
         prop_predicted_pos(cycles_pred_labs),
         lp_loss(bag_sizes, cycles_lps, transf_embryos_pred_labs),
         prop_predicted_pos(embryos_pred_labs),
         model.fit_thetas[1,1],
         logloss(bag_sizes, cycles_lps, transf_embryos_pred_proba),
         loglike(bag_sizes, cycles_lps, transf_embryos_pred_proba)]
        print('Metrics: ',metrics)
        l_metrics.append(metrics)

        #SAVE distances for the best EM execution
        df = distance_functions(best_estimation_record)
        distances_folds.append(df.compute_distances())

        # COMPUTE AUC score. This is done separately because we only consider cases when the
        # embryo outcome is unequivocally known. That is, those with labels 1 or 0 originally.
        xtest_clean=np.zeros([ts_data_clean.embryos.shape[0],
                 ts_data_clean.embryos.shape[1]+ts_data_clean.cycles.shape[1]])
        for e in np.arange(ts_data_clean.num_embryos):
            xtest_clean[e]=np.concatenate([ts_data_clean.embryos[e],
            	ts_data_clean.cycles[ts_data_clean.embryo_belong_to_cycle[e]]])

        ytest_clean = ts_data_clean.embryo_was_implanted
        unique, counts = np.unique(ytest_clean, return_counts=True)
        print('--- unique ytest_clean before inds_to_remove: {}'.format(dict(zip(unique, counts))))
        inds_to_remove = np.where(ts_data_clean.embryo_was_implanted == -1)
        print('inds_to_remove: {}'.format(inds_to_remove))
        xtest_clean = np.delete(xtest_clean, inds_to_remove, axis=0)
        ytest_clean = np.delete(ytest_clean, inds_to_remove, axis=0)
        unique, counts = np.unique(ytest_clean, return_counts=True)
        print('--- unique ytest_clean after inds_to_remove: {}'.format(dict(zip(unique, counts))))

        #The final prediction is the product of both classifiers and theta1
        ypred_clean_embryos = model.fit_class_embryos.predict_proba(xtest_clean[:,:ts_data_clean.embryos.shape[1]])
        ypred_clean_cycles = model.fit_class_cycles.predict_proba(xtest_clean[:,ts_data_clean.embryos.shape[1]:])
        ypred=ypred_clean_embryos*ypred_clean_cycles*model.fit_thetas[1,1]

        score=roc_auc_score(ytest_clean, ypred[:, 1])
        print('  * roc_auc_score: {}'.format(score))
        l_auc_roc.append(score)

        nfold += 1 #fold counter
        # END FOLD

    qr_euclidean_distances = []
    qr_rel_entropy_distances = []
    qw_euclidean_distances = []
    qw_rel_entropy_distances = []
    prb_euclidean_distances = []
    prb_rel_entropy_distances = []
    pwa_euclidean_distances = []
    pwa_rel_entropy_distances = []
    # Per a cada fold agafo per separat les diferents llistes i les agrupo
    # p.ex: qr_euclidean_distances: llista de llistes de distàncies euclídees de cada fold
    for idx_f, f in enumerate(distances_folds):

        qr_euclidean_distances.append(f[0])
        qr_rel_entropy_distances.append(f[1])
        qw_euclidean_distances.append(f[2])
        qw_rel_entropy_distances.append(f[3])
        prb_euclidean_distances.append(f[4])
        prb_rel_entropy_distances.append(f[5])
        pwa_euclidean_distances.append(f[6])
        pwa_rel_entropy_distances.append(f[7])

    qr_euclidean_distances_exps.append(qr_euclidean_distances)
    qr_rel_entropy_distances_exps.append(qr_rel_entropy_distances)
    qw_euclidean_distances_exps.append(qw_euclidean_distances)
    qw_rel_entropy_distances_exps.append(qw_rel_entropy_distances)
    prb_euclidean_distances_exps.append(prb_euclidean_distances)
    prb_rel_entropy_distances_exps.append(prb_rel_entropy_distances)
    pwa_euclidean_distances_exps.append(pwa_euclidean_distances)
    pwa_rel_entropy_distances_exps.append(pwa_rel_entropy_distances)

    partial_time = (time.time() - start_time)
    partial_times.append(partial_time)
    print('({}s)'.format(partial_time))

    # Save DISTANCES (for all folds) for THIS experiment into pickle files
    with open("results/ehu_experiment_distances_qr_euclidean_"
          + math_model + "_k" + str(k_cv) + "_ems" + str(max_it_em) + ".pickle", 'wb') as qr_euclidean_pckl:
        pickle.dump(qr_euclidean_distances, qr_euclidean_pckl, pickle.HIGHEST_PROTOCOL)
    with open("results/ehu_experiment_distances_qr_rel_entropy_"
          + math_model + "_k" + str(k_cv) + "_ems" + str(max_it_em) + ".pickle", 'wb') as qr_rel_entropy_pckl:
        pickle.dump(qr_rel_entropy_distances, qr_rel_entropy_pckl, pickle.HIGHEST_PROTOCOL)
    with open("results/ehu_experiment_distances_qw_euclidean_"
          + math_model + "_k" + str(k_cv) + "_ems" + str(max_it_em) + ".pickle", 'wb') as qw_euclidean_pckl:
        pickle.dump(qw_euclidean_distances, qw_euclidean_pckl, pickle.HIGHEST_PROTOCOL)
    with open("results/ehu_experiment_distances_qw_rel_entropy_"
          + math_model + "_k" + str(k_cv) + "_ems" + str(max_it_em) + ".pickle", 'wb') as qw_rel_entropy_pckl:
        pickle.dump(qw_rel_entropy_distances, qw_rel_entropy_pckl, pickle.HIGHEST_PROTOCOL)
    with open("results/ehu_experiment_distances_prb_euclidean_"
          + math_model + "_k" + str(k_cv) + "_ems" + str(max_it_em) + ".pickle", 'wb') as prb_euclidean_pckl:
        pickle.dump(prb_euclidean_distances, prb_euclidean_pckl, pickle.HIGHEST_PROTOCOL)
    with open("results/ehu_experiment_distances_prb_rel_entropy_"
          + math_model + "_k" + str(k_cv) + "_ems" + str(max_it_em) + ".pickle", 'wb') as prb_rel_entropy_pckl:
        pickle.dump(prb_rel_entropy_distances, prb_rel_entropy_pckl, pickle.HIGHEST_PROTOCOL)
    with open("results/ehu_experiment_distances_pwa_euclidean_"
          + math_model + "_k" + str(k_cv) + "_ems" + str(max_it_em) + ".pickle", 'wb') as pwa_euclidean_pckl:
        pickle.dump(pwa_euclidean_distances, pwa_euclidean_pckl, pickle.HIGHEST_PROTOCOL)
    with open("results/ehu_experiment_distances_pwa_rel_entropy_"
          + math_model + "_k" + str(k_cv) + "_ems" + str(max_it_em) + ".pickle", 'wb') as pwa_rel_entropy_pckl:
        pickle.dump(pwa_rel_entropy_distances, pwa_rel_entropy_pckl, pickle.HIGHEST_PROTOCOL)

    # SAVE METRICS FOR CURRENT MODEL
    with open("results/ehu_experiment_metrics_"
              + math_model + "_k" + str(k_cv) + "_ems" + str(max_it_em)+".pickle", 'wb') as metrics_pckl:#f_pckl:
        pickle.dump(l_metrics, metrics_pckl, pickle.HIGHEST_PROTOCOL)
    with open("results/probabilities_"+ math_model + ".pickle", 'wb') as prob_pckl:#f_pckl:
        pickle.dump(pred_probs, prob_pckl, pickle.HIGHEST_PROTOCOL)
    with open("results/real_info_"+ math_model +".pickle", 'wb') as impl_pckl:#f_pckl:
        pickle.dump(real_info, impl_pckl, pickle.HIGHEST_PROTOCOL)
    with open("results/aucroc_"+ math_model+ ".pickle", 'wb') as metrics_pckl:#f_pckl:
        pickle.dump(l_auc_roc, metrics_pckl, pickle.HIGHEST_PROTOCOL)
    print('\nMetrics\' RESULTS for experiment {}:'.format(math_model))
    for f in range(len(l_metrics)):
        print('Fold {}:'.format(f))
        print('   [ps_recall (cycles), prop_predicted_pos (cycles), lp_loss (embryos), prop_predicted_pos (embryos), theta1, logloss (embryos)]:')
        print('   {}'.format(l_metrics[f]))

# END of experiments
print("\n--- Experiments processed in %s seconds ---" % (time.time() - start_time))
for t in partial_times:
    print('     {}s'.format(t))
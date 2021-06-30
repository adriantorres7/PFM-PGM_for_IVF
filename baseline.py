from src.datasets_hidden import eu_data_ehu
from src.distance_functions import distance_functions
from src.other_functions import other_functions
import time, datetime
import pickle
import numpy as np
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import KFold
from sklearn.svm import SVC
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


# START experiments
start_time = time.time()
timestamp = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
print('dia i hora: {}'.format(timestamp))

# Iterate over all models.
for idx_m, m in enumerate(math_model_list):
    print('\n-- EXPERIMENT {} --'.format(m))

    data = eu_data_ehu()
    data_clean = eu_data_ehu()


    unique, counts = np.unique(data.embryo_was_implanted, return_counts=True)
    print('--- unique data.embryo_was_implanted after eu_data_ehu(): {}'.format(dict(zip(unique, counts))))

    # NO EM STRATEGY. FIX -1 LABELS TO 0.
    data.embryo_was_implanted[np.where(data.embryo_was_implanted == -1)] = 0
    unique, counts = np.unique(data.embryo_was_implanted, return_counts=True)
    print('--- unique data.embryo_was_implanted after data.embryo_was_implanted[np.where(data.embryo_was_implanted == -1)] = 0:\n{}'.format(dict(zip(unique, counts))))

    math_model = math_model_list[idx_m]

    # NUMBER OF FOLDS
    k_cv = 5



    np.random.seed(7)

    # The partition is made at cycle level
    l_insts = np.arange(data.num_cycles)
    kf = KFold(n_splits=k_cv)               # Cross-validation with k_cv folds

    l_metrics = []
    l_auc_roc = []
    distances_folds = []
    pred_probs = []
    real_info={}
    real_info['implanted']=[]
    real_info['quality']=[]
    real_info['transfered']=[]

    print('Total number of Embryos: {}'.format(data.num_embryos))
    print('Math model: {}'.format(math_model))
    print('Cross-Validation Folds: {}'.format(k_cv))
    print('Model learning without EM strategy')

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


        # MODEL SELECTION. Default= LR
        cl_embryos = LogisticRegression(random_state=0)

        if (math_model == 'LR'):
            cl_embryos = LogisticRegression(random_state=0)
        elif (math_model == 'RF200'):
            cl_embryos = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=0)
        elif (math_model == 'GBOOST'):
            cl_embryos = GradientBoostingClassifier(random_state=0)
        elif (math_model == 'ETREES'):
            cl_embryos = ExtraTreesClassifier(n_estimators=100, random_state=0)
        else:
            print('!!! ERROR: NO Mathematical model defined !!!')

        
        # TRAIN/TEST split for current fold. The labels are whether the embryo implanted. Unknown outcomes are fixed to 0.
        xtrain = tr_data.embryos
        ytrain = tr_data.embryo_was_implanted
        unique, counts = np.unique(ytrain, return_counts=True)
        print('--- unique ytrain before prediction: {}'.format(dict(zip(unique, counts))))
        xtest = ts_data.embryos
        ytest = ts_data.embryo_was_implanted
        unique, counts = np.unique(ytest, return_counts=True)
        print('--- unique ytest before prediction: {}'.format(dict(zip(unique, counts))))

        # TRAINING the classifier
        fit_class_embryos = cl_embryos.fit(xtrain, ytrain)
        
        # PREDICTIONS for the TEST set
        embryos_pred_labs = fit_class_embryos.predict(xtest)
        embryos_pred_proba = fit_class_embryos.predict_proba(ts_data.embryos)[:,1]

        # Embryo info to make plots separating by different factors
        pred_probs.extend(embryos_pred_proba)
        real_info['implanted'].extend(ts_data_clean.embryo_was_implanted)
        real_info['quality'].extend(ts_data_clean.embryo_quality)
        real_info['transfered'].extend(ts_data_clean.embryo_was_transfered)

        # Transforming predictions to adapt to metrics
        bag_sizes = np.array([len(c) for c in ts_data.cycle_has_trans_embryos])
        cycles_lps = ts_data.num_emb_implanted_per_cycle
        transf_embryos_pred_proba = np.array([embryos_pred_proba[i_e]
                                             for i_c,c in enumerate(ts_data.cycle_has_trans_embryos) for i_e in c])
        
        transf_embryos_pred_labs = np.array([embryos_pred_labs[i_e]
            for c in ts_data.cycle_has_trans_embryos for i_e in c])

        # COMPUTE METRICS (except AUC)
        metrics = [lp_loss(bag_sizes, cycles_lps, transf_embryos_pred_labs),
         prop_predicted_pos(embryos_pred_labs),
         logloss(bag_sizes, cycles_lps, transf_embryos_pred_proba),
         loglike(bag_sizes, cycles_lps, transf_embryos_pred_proba)]

        l_metrics.append(metrics)

        # COMPUTE AUC score. This is done separately because we only consider cases when the
        # embryo outcome is unequivocally known. That is, those with labels 1 or 0 originally.
        xtest_clean = ts_data_clean.embryos
        ytest_clean = ts_data_clean.embryo_was_implanted
        unique, counts = np.unique(ytest_clean, return_counts=True)
        print('--- unique ytest_clean before inds_to_remove: {}'.format(dict(zip(unique, counts))))
        inds_to_remove = np.where(ts_data_clean.embryo_was_implanted == -1)
        print('inds_to_remove: {}'.format(inds_to_remove))
        xtest_clean = np.delete(xtest_clean, inds_to_remove, axis=0)
        ytest_clean = np.delete(ytest_clean, inds_to_remove, axis=0)
        unique, counts = np.unique(ytest_clean, return_counts=True)
        print('--- unique ytest_clean after inds_to_remove: {}'.format(dict(zip(unique, counts))))

        ypred_clean = fit_class_embryos.predict_proba(xtest_clean)

        score=roc_auc_score(ytest_clean, ypred_clean[:,1])
        print('  * roc_auc_score: {}'.format(score))
        l_auc_roc.append(score)

        nfold += 1  #fold counter
        # END FOLD

    # SAVE METRICS FOR CURRENT MODEL
    with open("results/baseline/aucroc_"+ math_model +"_0"+"_hidden"+ ".pickle", 'wb') as metrics_pckl:#f_pckl:
        pickle.dump(l_auc_roc, metrics_pckl, pickle.HIGHEST_PROTOCOL)
    with open("results/baseline/metrics_"+ math_model +"_0"+"_hidden"+ ".pickle", 'wb') as metrics_pckl:#f_pckl:
        pickle.dump(l_metrics, metrics_pckl, pickle.HIGHEST_PROTOCOL)
    with open("results/baseline/probabilities_"+ math_model +"_0"+"_hidden"+ ".pickle", 'wb') as prob_pckl:#f_pckl:
        pickle.dump(pred_probs, prob_pckl, pickle.HIGHEST_PROTOCOL)
    with open("results/baseline/real_info_"+ math_model +"_0"+"_hidden"+ ".pickle", 'wb') as impl_pckl:#f_pckl:
        pickle.dump(real_info, impl_pckl, pickle.HIGHEST_PROTOCOL)


# END of experiments
print("\n--- Experiments processed in %s seconds ---" % (time.time() - start_time))
for t in partial_times:
    print('     {}s'.format(t))
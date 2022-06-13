import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, precision_recall_fscore_support, \
    confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from scipy.stats import ttest_1samp
from sklearn.decomposition import PCA


def save_print_metrics(df, rates, unseen_data_metrics, model_name):
    f = open(OUTPUT_DIR + model_name + '.txt', 'w')

    tp = rates['tp']
    tn = rates['tn']
    fp = rates['fp']
    fn = rates['fn']

    # ZeroR always predicts the the majority class, thus creating a 0/0 expression. Just to avoid it:
    if (tp + fp) == 0:
        prec = None
    else:
        prec = tp / (tp + fp)

    print("------------ "+ model_name + " L1O ------------")
    print("------------ "+ model_name + " L1O ------------", file=f)
    print(df)
    print(df.to_string(), file=f)
    print("------------ "+ model_name + " MEAN ------------")
    print("------------ "+ model_name + " MEAN ------------", file=f)
    print(f"Accuracy: {np.mean(df['Accuracy'])},\n"
          f"Sensitivity: {tp / (tp + fn)},\n"
          f"Specificity: {tn / (tn + fp)},\n"
          f"Precision: {prec}")
    print(f"Accuracy: {np.mean(df['Accuracy'])},\n"
          f"Sensitivity: {tp / (tp + fn)},\n"
          f"Specificity: {tn / (tn + fp)},\n"
          f"Precision: {prec}", file=f)
    print("------------ " + model_name + " PERFORMANCE w/ UNSEEN DATA ------------")
    print("------------ " + model_name + " PERFORMANCE w/ UNSEEN DATA ------------", file=f)
    print(f"Accuracy: {unseen_data_metrics['acc']},\n"
          f"Sensitivity: {unseen_data_metrics['sens']},\n"
          f"Specificity: {unseen_data_metrics['spec']},\n"
          f"Precision: {unseen_data_metrics['prec']}")
    print(f"Accuracy: {unseen_data_metrics['acc']},\n"
          f"Sensitivity: {unseen_data_metrics['sens']},\n"
          f"Specificity: {unseen_data_metrics['spec']},\n"
          f"Precision: {unseen_data_metrics['prec']}", file=f)


OUTPUT_DIR = "./output_ML/MD/"
FILENAME = 'MDindiv_0p05'
tn = fp = tp = fn = prediction = 0

path_train = "C:/Users/imano/Desktop/MU/PBL/PBL-NEUROMOD/Voxel_metrics_tosplit/MD_indiv_train.csv"
path_unseen = "C:/Users/imano/Desktop/MU/PBL/PBL-NEUROMOD/Voxel_metrics_tosplit/MD_indiv_test.csv"
df_train = pd.read_csv(path_train, encoding='latin1', delimiter=',')
df_unseen = pd.read_csv(path_unseen, encoding='latin1', delimiter=',')

# Just to use a subgroup
my_file = open("C:/Users/imano/Desktop/MU/PBL/PBL-NEUROMOD/Voxel_metrics_tosplit/MD_indiv_regs_FDR_0p05.txt", "r")
content = my_file.read()
subgroup = content.split("\n")
new_list = [s.replace('"', '') for s in subgroup]
new_list.append("Label")
new_list.remove('')
my_file.close()
df_train = df_train[new_list]
df_unseen = df_unseen[new_list]

y_train_all = np.array(df_train['Label'])
df_train.drop(['Label'], inplace=True, axis='columns')

y_validate = np.array(df_unseen['Label'])
df_unseen.drop(['Label'], inplace=True, axis='columns')

# PCA application
pca_train = PCA(n_components=min(df_train.shape[0], df_unseen.shape[0]))
df_train = pca_train.fit_transform(df_train)
eigenvalues = pca_train.explained_variance_
var_explainability = pca_train.explained_variance_ratio_
cum_sum_eigenvalues = np.cumsum(var_explainability)
idx = next(i for i, v in enumerate(cum_sum_eigenvalues) if v > 0.9)
print(f"Nº of PCs: {idx}")

pca_val = PCA(n_components=min(df_train.shape[0], df_unseen.shape[0]))
df_unseen = pca_val.fit_transform(df_unseen)

x_train_all = np.array(df_train)
x_validate = np.array(df_unseen)


"""Dummy model"""
df = pd.DataFrame()
dummy = DummyClassifier(random_state=42)
for i in range(x_train_all.shape[0] - 1):
    x_train = x_train_all
    y_train = y_train_all

    x_test = x_train[i, :]
    x_test = np.expand_dims(x_test, axis=0)
    x_train = np.delete(x_train, i, axis=0)

    y_test = y_train[i]
    y_test = np.array([y_test])
    y_train = np.delete(y_train, i, axis=0)

    dummy.fit(x_train, y_train)
    dummy_pred = dummy.predict(x_test)

    if dummy_pred == 0 and y_test == 0:
        tn += 1
    elif dummy_pred == 1 and y_test == 1:
        tp += 1
    elif dummy_pred == 0 and y_test == 1:
        fn += 1
    elif dummy_pred == 1 and y_test == 0:
        fp += 1

    df.loc[i, 'Accuracy'] = accuracy_score(y_test, dummy_pred)

true_false_rates = {'tp': tp, 'tn': tn, 'fn': fn, 'fp': fp}

prediction = dummy.predict(x_validate)
acc_val = accuracy_score(y_validate, prediction)
prec_val, sens_val, _, _ = precision_recall_fscore_support(y_validate, prediction, average='micro')
tn_val, fp_val, fn_val, tp_val = confusion_matrix(y_validate, prediction).ravel()
unseen_metrics = {'acc': acc_val, 'prec': prec_val, 'sens': sens_val, 'spec': tn_val / (tn_val+fp_val)}

dummy_pred_df = df
save_print_metrics(dummy_pred_df, true_false_rates, unseen_metrics, 'ZeroR_L1O_' + FILENAME)


"""SVM model"""
tn = fp = tp = fn = prediction = 0
df = pd.DataFrame()
svc = svm.SVC(random_state=42, verbose=1)
for i in range(x_train_all.shape[0] - 1):
    x_train = x_train_all
    y_train = y_train_all

    x_test = x_train[i, :]
    x_test = np.expand_dims(x_test, axis=0)
    x_train = np.delete(x_train, i, axis=0)

    y_test = y_train[i]
    y_test = np.array([y_test])
    y_train = np.delete(y_train, i, axis=0)

    svc.fit(x_train, y_train)
    svc_pred = svc.predict(x_test)

    if svc_pred == 0 and y_test == 0:
        tn += 1
    elif svc_pred == 1 and y_test == 1:
        tp += 1
    elif svc_pred == 0 and y_test == 1:
        fn += 1
    elif svc_pred == 1 and y_test == 0:
        fp += 1

    df.loc[i, 'Accuracy'] = accuracy_score(y_test, svc_pred)

true_false_rates = {'tp': tp, 'tn': tn, 'fn': fn, 'fp': fp}

prediction = svc.predict(x_validate)
acc_val = accuracy_score(y_validate, prediction)
prec_val = precision_score(y_validate, prediction)
_, sens_val, _, _ = precision_recall_fscore_support(y_validate, prediction, average='micro')
tn_val, fp_val, fn_val, tp_val = confusion_matrix(y_validate, prediction).ravel()
unseen_metrics = {'acc': acc_val, 'prec': prec_val, 'sens': sens_val, 'spec': tn_val / (tn_val+fp_val)}

svm_pred_df = df
save_print_metrics(svm_pred_df, true_false_rates, unseen_metrics, 'SVM_L1O_' + FILENAME)

"""RF model"""
tn = fp = tp = fn = prediction = 0
df = pd.DataFrame()
rf = RandomForestClassifier(n_estimators=100, random_state=42, bootstrap=False, verbose=1)
for i in range(x_train_all.shape[0] - 1):
    x_train = x_train_all
    y_train = y_train_all

    x_test = x_train[i, :]
    x_test = np.expand_dims(x_test, axis=0)
    x_train = np.delete(x_train, i, axis=0)

    y_test = y_train[i]
    y_test = np.array([y_test])
    y_train = np.delete(y_train, i, axis=0)

    rf.fit(x_train, y_train)
    rf_pred = rf.predict(x_test)

    df.loc[i, 'Accuracy'] = accuracy_score(y_test, rf_pred)

    if rf_pred == 0 and y_test == 0:
        tn += 1
    elif rf_pred == 1 and y_test == 1:
        tp += 1
    elif rf_pred == 0 and y_test == 1:
        fn += 1
    elif rf_pred == 1 and y_test == 0:
        fp += 1

true_false_rates = {'tp': tp, 'tn': tn, 'fn': fn, 'fp': fp}

prediction = rf.predict(x_validate)
acc_val = accuracy_score(y_validate, prediction)
prec_val = precision_score(y_validate, prediction)
_, sens_val, _, _ = precision_recall_fscore_support(y_validate, prediction, average='micro')
tn_val, fp_val, fn_val, tp_val = confusion_matrix(y_validate, prediction).ravel()
unseen_metrics = {'acc': acc_val, 'prec': prec_val, 'sens': sens_val, 'spec': tn_val / (tn_val+fp_val)}

rf_pred_df = df
save_print_metrics(rf_pred_df, true_false_rates, unseen_metrics, 'RF_L1O_' + FILENAME)


"""Naive Bayes model"""
tn = fp = tp = fn = prediction = 0
df = pd.DataFrame()
nb = GaussianNB()
for i in range(x_train_all.shape[0] - 1):
    x_train = x_train_all
    y_train = y_train_all

    x_test = x_train[i, :]
    x_test = np.expand_dims(x_test, axis=0)
    x_train = np.delete(x_train, i, axis=0)

    y_test = y_train[i]
    y_test = np.array([y_test])
    y_train = np.delete(y_train, i, axis=0)

    nb.fit(x_train, y_train)
    nb_pred = nb.predict(x_test)

    df.loc[i, 'Accuracy'] = accuracy_score(y_test, nb_pred)

    if nb_pred == 0 and y_test == 0:
        tn += 1
    elif nb_pred == 1 and y_test == 1:
        tp += 1
    elif nb_pred == 0 and y_test == 1:
        fn += 1
    elif nb_pred == 1 and y_test == 0:
        fp += 1

true_false_rates = {'tp': tp, 'tn': tn, 'fn': fn, 'fp': fp}

prediction = nb.predict(x_validate)
acc_val = accuracy_score(y_validate, prediction)
prec_val = precision_score(y_validate, prediction)
_, sens_val, _, _ = precision_recall_fscore_support(y_validate, prediction, average='micro')
tn_val, fp_val, fn_val, tp_val = confusion_matrix(y_validate, prediction).ravel()
unseen_metrics = {'acc': acc_val, 'prec': prec_val, 'sens': sens_val, 'spec': tn_val / (tn_val+fp_val)}

nb_pred_df = df
save_print_metrics(nb_pred_df, true_false_rates, unseen_metrics, 'NB_L1O_' + FILENAME)

"""LDA model"""
tn = fp = tp = fn = prediction = 0
df = pd.DataFrame()
lda = LinearDiscriminantAnalysis()
for i in range(x_train_all.shape[0] - 1):
    x_train = x_train_all
    y_train = y_train_all

    x_test = x_train[i, :]
    x_test = np.expand_dims(x_test, axis=0)
    x_train = np.delete(x_train, i, axis=0)

    y_test = y_train[i]
    y_test = np.array([y_test])
    y_train = np.delete(y_train, i, axis=0)

    lda.fit(x_train, y_train)
    lda_pred = lda.predict(x_test)

    df.loc[i, 'Accuracy'] = accuracy_score(y_test, lda_pred)

    if lda_pred == 0 and y_test == 0:
        tn += 1
    elif lda_pred == 1 and y_test == 1:
        tp += 1
    elif lda_pred == 0 and y_test == 1:
        fn += 1
    elif lda_pred == 1 and y_test == 0:
        fp += 1

true_false_rates = {'tp': tp, 'tn': tn, 'fn': fn, 'fp': fp}

prediction = lda.predict(x_validate)
acc_val = accuracy_score(y_validate, prediction)
prec_val = precision_score(y_validate, prediction)
_, sens_val, _, _ = precision_recall_fscore_support(y_validate, prediction, average='micro')
tn_val, fp_val, fn_val, tp_val = confusion_matrix(y_validate, prediction).ravel()
unseen_metrics = {'acc': acc_val, 'prec': prec_val, 'sens': sens_val, 'spec': tn_val / (tn_val+fp_val)}

lda_pred_df = df
save_print_metrics(lda_pred_df, true_false_rates, unseen_metrics, 'LDA_L1O_' + FILENAME)


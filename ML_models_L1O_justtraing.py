import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from scipy.stats import ttest_1samp


def save_print_metrics(df, rates, model_name):
    f = open(OUTPUT_DIR + model_name + '.txt', 'w')

    tp = rates['tp']
    fp = rates['fp']
    tn = rates['tn']
    fn = rates['fn']

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


OUTPUT_DIR = "./output_ML/MD/JustTraining/"
FILENAME = 'MDindiv_0p05'
tn = fp = tp = fn = 0

path = "C:/Users/imano/Desktop/MU/PBL/PBL-NEUROMOD/Voxel_metrics/MD_indiv.csv"
df = pd.read_csv(path, encoding='latin1', delimiter=',')

# Just to use a subgroup
my_file = open("C:/Users/imano/Desktop/MU/PBL/PBL-NEUROMOD/Voxel_metrics/MD_indiv_regs_FDR_0p05.txt", "r")
content = my_file.read()
subgroup = content.split("\n")
new_list = [s.replace('"', '') for s in subgroup]
new_list.append("Label")
new_list.remove('')
my_file.close()
df = df[new_list]

y_train_all = np.array(df['Label'])
df.drop(['Label'], inplace=True, axis='columns')

# PCA application
pca_train = PCA(n_components=30)
df = pca_train.fit_transform(df)
eigenvalues = pca_train.explained_variance_
var_explainability = pca_train.explained_variance_ratio_
cum_sum_eigenvalues = np.cumsum(var_explainability)
idx = next(i for i, v in enumerate(cum_sum_eigenvalues) if v > 0.9)
print(f"Nº of PCs: {idx}")

x_train_all = np.array(df)

"""Dummy model"""
df = pd.DataFrame()
for i in range(x_train_all.shape[0] - 1):
    x_train = x_train_all
    y_train = y_train_all

    x_test = x_train[i, :]
    x_test = np.expand_dims(x_test, axis=0)
    x_train = np.delete(x_train, i, axis=0)

    y_test = y_train[i]
    y_test = np.array([y_test])
    y_train = np.delete(y_train, i, axis=0)

    dummy = DummyClassifier(random_state=42)
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

    df.loc[i, 'Accuracy'] = dummy.score(x_test, y_test)


true_false_rates = {'tp': tp, 'tn': tn, 'fn': fn, 'fp': fp}
save_print_metrics(df, true_false_rates, 'ZeroR_L1O_' + FILENAME)

popmean = np.mean(df['Accuracy'])

"""SVM model"""
df = pd.DataFrame()
tn = fp = tp = fn = 0
pred_set = []
for i in range(x_train_all.shape[0] - 1):
    x_train = x_train_all
    y_train = y_train_all

    x_test = x_train[i, :]
    x_test = np.expand_dims(x_test, axis=0)
    x_train = np.delete(x_train, i, axis=0)

    y_test = y_train[i]
    y_test = np.array([y_test])
    y_train = np.delete(y_train, i, axis=0)

    svc = svm.SVC(random_state=42, verbose=1)
    svc.fit(x_train, y_train)
    svc_pred = svc.predict(x_test)
    pred_set.append(svc_pred)

    if svc_pred == 0 and y_test == 0:
        tn += 1
    elif svc_pred == 1 and y_test == 1:
        tp += 1
    elif svc_pred == 0 and y_test == 1:
        fn += 1
    elif svc_pred == 1 and y_test == 0:
        fp += 1

    df.loc[i, 'Accuracy'] = svc.score(x_test, y_test)

true_false_rates = {'tp': tp, 'tn': tn, 'fn': fn, 'fp': fp}
save_print_metrics(df, true_false_rates, 'SVM_L1O_' + FILENAME)

_, pval_svm = ttest_1samp(pred_set, popmean=popmean, alternative='two-sided')

"""RF model"""
tn = fp = tp = fn = 0
df = pd.DataFrame()
pred_set = []
for i in range(x_train_all.shape[0] - 1):
    x_train = x_train_all
    y_train = y_train_all

    x_test = x_train[i, :]
    x_test = np.expand_dims(x_test, axis=0)
    x_train = np.delete(x_train, i, axis=0)

    y_test = y_train[i]
    y_test = np.array([y_test])
    y_train = np.delete(y_train, i, axis=0)

    rf = RandomForestClassifier(n_estimators=100, random_state=42, bootstrap=False, verbose=1)
    rf.fit(x_train, y_train)
    rf_pred = rf.predict(x_test)
    pred_set.append(rf_pred)

    df.loc[i, 'Accuracy'] = rf.score(x_test, y_test)

    if rf_pred == 0 and y_test == 0:
        tn += 1
    elif rf_pred == 1 and y_test == 1:
        tp += 1
    elif rf_pred == 0 and y_test == 1:
        fn += 1
    elif rf_pred == 1 and y_test == 0:
        fp += 1

true_false_rates = {'tp': tp, 'tn': tn, 'fn': fn, 'fp': fp}
save_print_metrics(df, true_false_rates, 'RF_L1O_' + FILENAME)

_, pval_rf= ttest_1samp(pred_set, popmean=popmean, alternative='two-sided')

"""Naive Bayes model"""
tn = fp = tp = fn = 0
df = pd.DataFrame()
pred_set = []
for i in range(x_train_all.shape[0] - 1):
    x_train = x_train_all
    y_train = y_train_all

    x_test = x_train[i, :]
    x_test = np.expand_dims(x_test, axis=0)
    x_train = np.delete(x_train, i, axis=0)

    y_test = y_train[i]
    y_test = np.array([y_test])
    y_train = np.delete(y_train, i, axis=0)

    nb = GaussianNB()
    nb.fit(x_train, y_train)
    nb_pred = nb.predict(x_test)
    pred_set.append(nb_pred)

    df.loc[i, 'Accuracy'] = nb.score(x_test, y_test)

    if nb_pred == 0 and y_test == 0:
        tn += 1
    elif nb_pred == 1 and y_test == 1:
        tp += 1
    elif nb_pred == 0 and y_test == 1:
        fn += 1
    elif nb_pred == 1 and y_test == 0:
        fp += 1

true_false_rates = {'tp': tp, 'tn': tn, 'fn': fn, 'fp': fp}
save_print_metrics(df, true_false_rates, 'NB_L1O_' + FILENAME)

_, pval_nb = ttest_1samp(pred_set, popmean=popmean, alternative='two-sided')

"""LDA model"""
tn = fp = tp = fn = 0
df = pd.DataFrame()
pred_set = []
for i in range(x_train_all.shape[0] - 1):
    x_train = x_train_all
    y_train = y_train_all

    x_test = x_train[i, :]
    x_test = np.expand_dims(x_test, axis=0)
    x_train = np.delete(x_train, i, axis=0)

    y_test = y_train[i]
    y_test = np.array([y_test])
    y_train = np.delete(y_train, i, axis=0)

    lda = LinearDiscriminantAnalysis()
    lda.fit(x_train, y_train)
    lda_pred = lda.predict(x_test)
    pred_set.append(lda_pred)

    df.loc[i, 'Accuracy'] = lda.score(x_test, y_test)

    if lda_pred == 0 and y_test == 0:
        tn += 1
    elif lda_pred == 1 and y_test == 1:
        tp += 1
    elif lda_pred == 0 and y_test == 1:
        fn += 1
    elif lda_pred == 1 and y_test == 0:
        fp += 1

true_false_rates = {'tp': tp, 'tn': tn, 'fn': fn, 'fp': fp}
save_print_metrics(df, true_false_rates, 'LDA_L1O_' + FILENAME)

_, pval_lda = ttest_1samp(pred_set, popmean=popmean, alternative='two-sided')

print(f"p-value NB: {pval_nb}")
print(f"p-value LDA: {pval_lda}")
print(f"p-value RF: {pval_rf}")
print(f"p-value SVM: {pval_svm}")
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def save_print_metrics(rates, model_name):
    f = open(OUTPUT_DIR + model_name + '.txt', 'w')

    tp = rates['tp']
    fp = rates['fp']
    tn = rates['tn']
    fn = rates['fn']

    print("------------ "+ model_name + " L1O ------------")
    print("------------ "+ model_name + " L1O ------------", file=f)
    print(df)
    print(df.to_string(), file=f)
    print("------------ "+ model_name + " MEAN ------------")
    print("------------ "+ model_name + " MEAN ------------", file=f)
    print(f"Accuracy: {np.mean(df['Accuracy'])},\n"
          f"Sensitivity: {tp / (tp + fn)},\n"
          f"Specificity: {tn / (tn + fp)},\n"
          f"Precision: {tp / (tp + fp)}")
    print(f"Accuracy: {np.mean(df['Accuracy'])},\n"
          f"Sensitivity: {tp / (tp + fn)},\n"
          f"Specificity: {tn / (tn + fp)},\n"
          f"Precision: {tp / (tp + fp)}", file=f)


OUTPUT_DIR = "./output_ML/MD/"
FILENAME = 'MDdual'
tn = fp = tp = fn = 0

path = "C:/Users/imano/Desktop/MU/PBL/PBL-NEUROMOD/Voxel_metrics/MD_dual.csv"
df = pd.read_csv(path, encoding='latin1', delimiter=',')

y_train_all = np.array(df['Label'])

df.drop(['Label'], inplace=True, axis='columns')
x_train_all = np.array(df)

"""SVM model"""
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

    svc = svm.SVC(random_state=42, verbose=1)
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

    df.loc[i, 'Accuracy'] = svc.score(x_test, y_test)

true_false_rates = {'tp': tp, 'tn': tn, 'fn': fn, 'fp': fp}
save_print_metrics(true_false_rates, 'SVM_L1O_' + FILENAME)

"""RF model"""
tn = fp = tp = fn = 0
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

    rf = RandomForestClassifier(n_estimators=100, random_state=42, bootstrap=False, verbose=1)
    rf.fit(x_train, y_train)
    rf_pred = rf.predict(x_test)

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
save_print_metrics(true_false_rates, 'RF_L1O_' + FILENAME)

"""Naive Bayes model"""
tn = fp = tp = fn = 0
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

    nb = GaussianNB()
    nb.fit(x_train, y_train)
    nb_pred = nb.predict(x_test)

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
save_print_metrics(true_false_rates, 'NB_L1O_' + FILENAME)

"""LDA model"""
tn = fp = tp = fn = 0
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

    lda = LinearDiscriminantAnalysis()
    lda.fit(x_train, y_train)
    lda_pred = lda.predict(x_test)

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
save_print_metrics(true_false_rates, 'LDA_L1O_' + FILENAME)
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB

OUTPUT_DIR = "./output_ML/MD/"
tn = fp = tp = fn = 0

path = "C:/Users/imano/Desktop/MU/PBL/PBL-NEUROMOD/Voxel_metrics/MD_dual.csv"
df = pd.read_csv(path, encoding='latin1', delimiter=',')
# df = df[["A8m",
# "A8dl",
# "A9l",
# "A6dl",
# "A6m",
# "A9m",
# "A10m",
# "A9/46d",
# "IFJ",
# "A46",
# "A9/46v",
# "A8vl",
# "A6vl",
# "A10l",
# "A44d",
# "IFS",
# "A45c",
# "A45r",
# "A44op",
# "A44v",
# "A14m",
# "A12/47o",
# "A11l",
# "A11m",
# "A13",
# "A12/47l",
# "A4hf",
# "A6cdl",
# "A4ul",
# "A4tl",
# "A6cvl",
# "A1/2/3ll",
# "A4ll",
# "A38m",
# "A41/42",
# "TE1.0/TE1.2",
# "A22c",
# "A38l",
# "A22r",
# "A21c",
# "A21r",
# "A37dl",
# "aSTS",
# "A20iv",
# "A37elv",
# "A20r",
# "A20il",
# "A37vl",
# "A20cl",
# "A20cv",
# "A20rv",
# "A37mv",
# "A37lv",
# "A35/36r",
# "A35/36c",
# "TL",
# "A28/34",
# "TI",
# "TH",
# "rpSTS",
# "cpSTS",
# "A7r",
# "A7c",
# "A5l",
# "A7pc",
# "A7ip",
# "A39c",
# "A39rd",
# "A40rd",
# "A40c",
# "A39rv",
# "A40rv",
# "A7m",
# "A5m",
# "dmPOS",
# "A31",
# "A1/2/3ulhf",
# "A1/2/3tonIa",
# "A2",
# "G",
# "vIa",
# "dIa",
# "vId/vIg",
# "dIg",
# "dId",
# "A23d",
# "A24rv",
# "A32p",
# "A23v",
# "A24cd",
# "A23c",
# "A32sg",
# "cLinG",
# "rCunG",
# "rLinG",
# "vmPOS",
# "mOccG",
# "V5/MT+",
# "iOccG",
# "msOccG",
# "lsOccG",
# "mAmyg",
# "lAmyg",
# "rHipp",
# "cHipp",
# "vCa",
# "GP",
# "NAC",
# "vmPu",
# "dCa",
# "dlPu",
# "mPFtha",
# "mPMtha",
# "Stha",
# "rTtha",
# "PPtha",
# "Otha",
# "cTtha",
# "lPFtha",
# "Label"
# ]]

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

f = open(OUTPUT_DIR + 'SVM_L1O_MDdual.txt', 'w')
print("------------ SVM L1O ------------")
print("------------ SVM L1O ------------", file=f)
print(df)
print(df.to_string(), file=f)
print("------------ SVM MEAN ------------")
print("------------ SVM MEAN ------------", file=f)
print(f"Accuracy: {np.mean(df['Accuracy'])},\n"
      f"Sensitivity: {tp / (tp + fn)},\n"
      f"Specificity: {tn / (tn + fp)},\n"
      f"Precision: {tp / (tp + fp)}")
print(f"Accuracy: {np.mean(df['Accuracy'])},\n"
      f"Sensitivity: {tp / (tp + fn)},\n"
      f"Specificity: {tn / (tn + fp)},\n"
      f"Precision: {tp / (tp + fp)}", file=f)

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

    if svc_pred == 0 and y_test == 0:
        tn += 1
    elif svc_pred == 1 and y_test == 1:
        tp += 1
    elif svc_pred == 0 and y_test == 1:
        fn += 1
    elif svc_pred == 1 and y_test == 0:
        fp += 1

if (tp + fp) == 0:
    prec = 0
else:
    prec = tp / (tp + fp)

f = open(OUTPUT_DIR + 'RF_L1O_MDdual.txt', 'w')
print("------------ RF L1O ------------")
print("------------ RF L1O ------------", file=f)
print(df)
print(df.to_string(), file=f)
print("------------ RF MEAN ------------")
print("------------ RF MEAN ------------", file=f)
print(f"Accuracy: {np.mean(df['Accuracy'])},\n"
      f"Sensitivity: {tp / (tp + fn)},\n"
      f"Specificity: {tn / (tn + fp)},\n"
      f"Precision: {prec}")
print(f"Accuracy: {np.mean(df['Accuracy'])},\n"
      f"Sensitivity: {tp / (tp + fn)},\n"
      f"Specificity: {tn / (tn + fp)},\n"
      f"Precision: {prec}", file=f)

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

    if svc_pred == 0 and y_test == 0:
        tn += 1
    elif svc_pred == 1 and y_test == 1:
        tp += 1
    elif svc_pred == 0 and y_test == 1:
        fn += 1
    elif svc_pred == 1 and y_test == 0:
        fp += 1

if (tp + fp) == 0:
    prec = 0
else:
    prec = tp / (tp + fp)

f = open(OUTPUT_DIR + 'NB_L1O_MDdual.txt', 'w')
print("------------ NB L1O ------------")
print("------------ NB L1O ------------", file=f)
print(df)
print(df.to_string(), file=f)
print("------------ NB MEAN ------------")
print("------------ NB MEAN ------------", file=f)
print(f"Accuracy: {np.mean(df['Accuracy'])},\n"
      f"Sensitivity: {tp / (tp + fn)},\n"
      f"Specificity: {tn / (tn + fp)},\n"
      f"Precision: {prec}")
print(f"Accuracy: {np.mean(df['Accuracy'])},\n"
      f"Sensitivity: {tp / (tp + fn)},\n"
      f"Specificity: {tn / (tn + fp)},\n"
      f"Precision: {prec}", file=f)


import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB

output_dir = "./output_ML/"

path_to_train = "C:/Users/imano/Desktop/MU/PBL/PBL-NEUROMOD/train_weightCM_table.csv"
df_train = pd.read_csv(path_to_train, encoding='latin1', delimiter=',')
x_train = np.array(df_train[['EfficiencyW', 'MeanBetwennessW', 'MeanClustCoffW', 'MeanStrength', 'kDensity', 'TransitivityW']])
y_train = np.array(df_train['Label'])

path_to_test = "C:/Users/imano/Desktop/MU/PBL/PBL-NEUROMOD/test_weightCM_table.csv"
df_test = pd.read_csv(path_to_test, encoding='latin1', delimiter=',')
x_test = np.array(df_test[['EfficiencyW', 'MeanBetweennessW', 'MeanClustCoffW', 'MeanStrength', 'kDensity', 'TransitivityW']])
y_test = np.array(df_test['Label'])


"""SVM model"""
svc = svm.SVC(random_state=42, verbose=1)
svc.fit(x_train, y_train)
svc_pred = svc.predict(x_test)

cm = confusion_matrix(y_test, svc_pred)
plt.figure()
sns.heatmap(cm, annot=True, cmap='Blues')
plt.savefig(output_dir + 'Conf_map_svm')
plt.show()

print("------------ SVM EVALUATION ------------")
print(f"Mean Accuracy: {svc.score(x_test, y_test)}")

"""RF model"""
rf = RandomForestClassifier(n_estimators=100, random_state=42, bootstrap=True, verbose=1)
rf.fit(x_train, y_train)
rf_pred = rf.predict(x_test)

cm = confusion_matrix(y_test, rf_pred)
plt.figure()
sns.heatmap(cm, annot=True, cmap='Blues')
plt.savefig(output_dir + 'Conf_map_rf')
plt.show()

print("------------ RF EVALUATION ------------")
print(f"Mean Accuracy: {rf.score(x_test, y_test)}")

"""Naive Bayes model"""
nb = GaussianNB()
nb.fit(x_train, y_train)
nb_pred = nb.predict(x_test)

cm = confusion_matrix(y_test, nb_pred)
plt.figure()
sns.heatmap(cm, annot=True, cmap='Blues')
plt.savefig(output_dir + 'Conf_map_nb')
plt.show()

print("------------ NB EVALUATION ------------")
print(f"Mean Accuracy: {nb.score(x_test, y_test)}")
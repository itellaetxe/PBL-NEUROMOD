import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from imblearn.combine import SMOTETomek
import seaborn as sns

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.backend import clear_session
import tensorflow as tf
from datetime import datetime

from PBL_DWI_lib.load_all_ConnMats import load_all_ConnMats


def cnn_model(parameters, xtr, ytr, xval, yval):
    """
    param parameters:
    return: model, history
    """

    # Check if parameters are correctly introduced
    if parameters['num_of_layers'] != (len(parameters['num_of_neurons']) or len(parameters['dropout_vals'])):
        raise Exception("Parameters are not consistent.")

    # xtr, xval, ytr, yval = train_test_split(x_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

    # If SMOTETomek is wanted to be applied
    if parameters['imbalanced']:
        arr = np.reshape(xtr, (xtr.shape[0], 247 * 247))
        smtom = SMOTETomek(sampling_strategy='auto', random_state=42)
        xtr_res, ytr = smtom.fit_resample(arr, ytr)
        xtr = np.reshape(xtr_res, newshape=(xtr_res.shape[0], 247, 247, 1))

    neurons = parameters['num_of_neurons']
    # dropouts = parameters['dropout_vals']

    # Build model architecture
    model = Sequential()
    model.add(Conv2D(neurons[0], kernel_size=(3, 3), input_shape=(247, 247, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(neurons[1], kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # model.add(Conv2D(neurons[2], kernel_size=(3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # #
    # model.add(Conv2D(neurons[3], kernel_size=(3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(Conv2D(neurons[4], kernel_size=(3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    # model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(64))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(optimizer=Adam(learning_rate=parameters['lr']),
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'AUC', Precision(), Recall()])

    mcp = ModelCheckpoint(output_dir + '.mdl.wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')

    # Run model training
    print("------------------- TRAINING STARTED -------------------")
    history = model.fit(x=xtr,
                        y=ytr,
                        batch_size=parameters['batch_size'],
                        verbose=1,
                        epochs=parameters['epoch'],
                        validation_data=(xval, yval),
                        callbacks=[mcp])
    print("------------------- TRAINING FINISHED -------------------")

    return history, model

# Seed = 42
np.random.seed(42)
tf.random.set_seed(42)
method = 'raw'
comments = 'Crossvalidated'

base_path = "D:/PROCESSED_ADNI_CONTROL_GROUP/results/"
input_struct_NC = load_all_ConnMats(base_path, method)

base_path = "D:/PROCESSED_ADNI_AD_GROUP/PROCESSED_AD_GROUP/"
input_struct_AD = load_all_ConnMats(base_path, method)

x_train = np.concatenate((input_struct_NC, input_struct_AD), axis=0)
y_train = np.zeros((input_struct_NC.shape[0] + input_struct_AD.shape[0]))
y_train[input_struct_NC.shape[0]:input_struct_NC.shape[0] + input_struct_AD.shape[0]] = 1

base_path = "D:/TEST/NC/"
x_test_NC = load_all_ConnMats(base_path, method)

base_path = "D:/TEST/AD/"
x_test_AD = load_all_ConnMats(base_path, method)

x_test = np.concatenate((x_test_NC, x_test_AD), axis=0)
# y_test = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
y_test = np.zeros((x_test_NC.shape[0] + x_test_AD.shape[0]))
y_test[x_test_NC.shape[0]:x_test_NC.shape[0] + x_test_AD.shape[0]] = 1

parameters = {'num_of_neurons': [16, 8],  # [layer0, layer1, layer2, ...]
              # 'dropout_vals': [0.5, 0.5, 0.5],
              'batch_size': 8,  # 32, 64, 128, ...
              'num_of_layers': 2,  # total = num_of_layers + 1
              'lr': 0.001,
              # 'sgd_momentum': 0.4371162594318422, # Just if SGD optimizer is selected
              'epoch': 30,
              'imbalanced': False}

output_dir = './output_CNN/'
n_folds = 5
model_history = []
df = pd.DataFrame()

for i in range(n_folds):
    print("Training on Fold: ",i+1)
    x_train, val_x, y_train, val_y = train_test_split(x_train, y_train, test_size=np.round(95/5)/100, random_state=np.random.randint(1, 1000, 1)[0])
    history, model = cnn_model(parameters, x_train, y_train, val_x, val_y)
    model_history.append(history)

    print("=======" * 12, end="\n\n\n")
    loss, accuracy, auc, pre, rec = model.evaluate(x_test, y_test, verbose=0)
    df.loc[i, 'Loss'] = loss
    df.loc[i, 'AUC'] = auc
    df.loc[i, 'accuracy'] = accuracy
    df.loc[i, 'Precision'] = pre
    df.loc[i, 'Recall'] = rec

color = cm.rainbow(np.linspace(0, 1, n_folds))

plt.figure()
plt.title('Test Loss')
for i, c in zip(range(n_folds), color):
    plt.plot(model_history[i].history['val_loss'], label=f'Test Fold {i+1}', linestyle='--', color=c)
plt.legend()
# plt.show()

plt.figure()
plt.title('Train Loss')
for i, c in zip(range(n_folds), color):
    plt.plot(model_history[i].history['loss'], label=f'Training Fold {i+1}', color=c)
plt.legend()
# plt.show()

plt.figure()
plt.title('Train AUC')
for i, c in zip(range(n_folds), color):
    plt.plot(model_history[i].history['auc'], label=f'Training Fold {i+1}', color=c)
plt.legend()
# plt.show()

plt.figure()
plt.title('Test AUC')
for i, c in zip(range(n_folds), color):
    plt.plot(model_history[i].history['val_auc'], label=f'Test Fold {i+1}', linestyle='--', color=c)
plt.legend()
plt.show()

df.loc[n_folds, 'Loss'] = np.mean(df['Loss'])
df.loc[n_folds, 'AUC'] = np.mean(df['AUC'])
df.loc[n_folds, 'accuracy'] = np.mean(df['accuracy'])
df.loc[n_folds, 'Precision'] = np.mean(df['Precision'])
df.loc[n_folds, 'Recall'] = np.mean(df['Recall'])
print('#### VALIDATION RESULTS (final row is mean) ####')
print(df)

# model.save(output_dir + 'output_CNN_model.h5')

# y_pred = model.predict(x_test, verbose=0)
# cm = confusion_matrix(y_test, np.round(y_pred))
# plt.figure()
# categories = ['NC', 'AD']
# sns.heatmap(cm, annot=True, cmap='Blues')
# plt.savefig(output_dir + 'Conf_map')
# plt.show()
#
# csv_tmp = pd.DataFrame({'Datetime': [datetime.now().strftime("%m/%d/%Y, %H:%M:%S")],
#                         'num_of_neurons': [str(parameters['num_of_neurons'])],
#                         'batch_size': [parameters['batch_size']],
#                         'lr': [parameters['lr']],
#                         'epoch': [parameters['epoch']],
#                         'imbalanced': [parameters['imbalanced']],
#                         'Loss': [loss],
#                         'Acc': [accuracy],
#                         'AUC': [auc],
#                         'Pre': [pre],
#                         'Rec': [rec],
#                         'method': method,
#                         'comments': comments})
#

clear_session()


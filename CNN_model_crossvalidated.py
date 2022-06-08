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

    mcp = ModelCheckpoint(OUTPUT_DIR + '.mdl.wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')

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
METHOD = 'weighted'
comments = 'Crossvalidated-L1O with weighted data'

base_path = "D:/PROCESSED_ADNI_CONTROL_GROUP/results/"
input_struct_NC, subs_NC = load_all_ConnMats(base_path, METHOD)

base_path = "D:/PROCESSED_ADNI_AD_GROUP/PROCESSED_AD_GROUP/"
input_struct_AD, subs_AD = load_all_ConnMats(base_path, METHOD)

subs_train = subs_NC + subs_AD

x_train_all = np.concatenate((input_struct_NC, input_struct_AD), axis=0)
y_train_all = np.zeros((input_struct_NC.shape[0] + input_struct_AD.shape[0]))
y_train_all[input_struct_NC.shape[0]:input_struct_NC.shape[0] + input_struct_AD.shape[0]] = 1

# base_path = "D:/TEST/NC/"
# x_test_NC, subs_NC = load_all_ConnMats(base_path, METHOD)
#
# base_path = "D:/TEST/AD/"
# x_test_AD, subs_AD = load_all_ConnMats(base_path, METHOD)
#
# subs_test = subs_NC + subs_AD
#
# x_test = np.concatenate((x_test_NC, x_test_AD), axis=0)
# y_test = np.zeros((x_test_NC.shape[0] + x_test_AD.shape[0]))
# y_test[x_test_NC.shape[0]:x_test_NC.shape[0] + x_test_AD.shape[0]] = 1

parameters = {'num_of_neurons': [16, 8],  # [layer0, layer1, layer2, ...]
              # 'dropout_vals': [0.5, 0.5, 0.5],
              'batch_size': 8,  # 32, 64, 128, ...
              'num_of_layers': 2,  # total = num_of_layers + 1
              'lr': 0.001,
              # 'sgd_momentum': 0.4371162594318422, # Just if SGD optimizer is selected
              'epoch': 30,
              'imbalanced': False}

OUTPUT_DIR = './output_CNN/'
TOTAL_SUBS = x_train_all.shape[0]
N_FOLDS = x_train_all.shape[0]
model_history = []
df = pd.DataFrame()

for i in range(TOTAL_SUBS):
    print("Training on Fold: ", i+1)
    x_train = x_train_all
    y_train = y_train_all

    x_test = x_train[i, :, :, :]
    x_test = np.expand_dims(x_test, axis=0)
    x_train = np.delete(x_train, i, axis=0)
    y_test = y_train[i]
    y_test = np.array([y_test])
    y_train = np.delete(y_train, i, axis=0)

    x_train, val_x, y_train, val_y = train_test_split(x_train, y_train, test_size=0.3, random_state=42)

    history, model = cnn_model(parameters, x_train, y_train, val_x, val_y)
    model_history.append(history)

    print("=======" * 12, end="\n\n\n")
    loss, accuracy, auc, pre, rec = model.evaluate(x_test, y_test, verbose=0)
    df.loc[i, 'Loss'] = loss
    df.loc[i, 'AUC'] = auc
    df.loc[i, 'accuracy'] = accuracy
    df.loc[i, 'Precision'] = pre
    df.loc[i, 'Recall'] = rec

print(df)

color = cm.rainbow(np.linspace(0, 1, N_FOLDS))

plt.figure()
plt.title('Test Loss')
for i, c in zip(range(N_FOLDS), color):
    plt.plot(model_history[i].history['val_loss'], label=f'Test Fold {i+1}', linestyle='--', color=c)
plt.xlabel('Epoch')
plt.savefig(OUTPUT_DIR + 'Test_loss_L1O')

plt.figure()
plt.title('Train Loss')
for i, c in zip(range(N_FOLDS), color):
    plt.plot(model_history[i].history['loss'], label=f'Training Fold {i+1}', color=c)
plt.xlabel('Epoch')
plt.savefig(OUTPUT_DIR + 'Train_loss_L1O')

plt.figure()
plt.title('Train AUC')
for i, c in zip(range(N_FOLDS), color):
    plt.plot(model_history[i].history['auc'], label=f'Training Fold {i+1}', color=c)
plt.xlabel('Epoch')
plt.savefig(OUTPUT_DIR + 'Train_AUC_L1O')

plt.figure()
plt.title('Test AUC')
for i, c in zip(range(N_FOLDS), color):
    plt.plot(model_history[i].history['val_auc'], label=f'Test Fold {i+1}', linestyle='--', color=c)
plt.xlabel('Epoch')
plt.savefig(OUTPUT_DIR + 'Test_AUC_L1O')
plt.show()

print("***\nMEAN VALUES\n***", end="\n\n\n")
print(f"Loss: {np.mean(df['Loss'])}")
print(f"AUC: {np.mean(df['AUC'])}")
print(f"Accuracy: {np.mean(df['accuracy'])}")
print(f"Precision: {np.mean(df['Precision'])}")
print(f"Recall: {np.mean(df['Recall'])}")

csv_tmp = pd.DataFrame({'Datetime': [datetime.now().strftime("%m/%d/%Y, %H:%M:%S")],
                        'num_of_neurons': [str(parameters['num_of_neurons'])],
                        'batch_size': [parameters['batch_size']],
                        'lr': [parameters['lr']],
                        'epoch': [parameters['epoch']],
                        'imbalanced': [parameters['imbalanced']],
                        'Loss': [np.mean(df['Loss'])],
                        'Acc': [np.mean(df['accuracy'])],
                        'AUC': [np.mean(df['AUC'])],
                        'Pre': [np.mean(df['Precision'])],
                        'Rec': [np.mean(df['Recall'])],
                        'method': METHOD,
                        'comments': comments})

# Add the training metrics and parameters to the csv in order to keep track of them
csv_path = "C:/Users/imano/Desktop/MU/PBL/PBL-NEUROMOD/output_CNN/trials.csv"
csv = pd.read_csv(csv_path)
csv = pd.concat([csv, csv_tmp])
csv.to_csv(csv_path, index=False)

clear_session()


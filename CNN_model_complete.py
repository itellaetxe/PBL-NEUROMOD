import os
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek
from scipy.io import loadmat

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import ModelCheckpoint


def load_all_ConnMats(base_path):

    dirs = os.listdir(base_path)
    r = re.compile("sub-")
    filtered = [folder for folder in dirs if r.match(folder)]

    subs = [base_path + sub for sub in filtered]
    subs = [sub + "/ConnMat" for sub in subs]

    mat_paths = []
    for sub in subs:
        tmp_dir = os.listdir(sub)
        tmp_path = [file for file in tmp_dir if ".mat" in file][0]
        tmp_path = sub + "/" + tmp_path
        mat_paths.append(tmp_path)

    input_struct = np.zeros((len(mat_paths), 247, 247, 1))
    for cnt, mat in enumerate(mat_paths):
        tmp_mat = loadmat(mat)
        tmp_mat = tmp_mat['myConnMat']
        input_struct[cnt, :, :, 0] = tmp_mat

    return input_struct


def cnn_model(parameters, x_train, y_train):
    """
    param parameters:
    return: model, history
    """

    # Check if parameters are correctly introduced
    if parameters['num_of_layers'] != (len(parameters['num_of_neurons']) or len(parameters['dropout_vals'])):
        raise Exception("Parameters are not consistent.")

    xtr, xval, ytr, yval = train_test_split(x_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

    # If SMOTETomek is wanted to be applied
    if parameters['imbalanced']:
        arr = np.reshape(xtr, (xtr.shape[0], 247*247))
        smtom = SMOTETomek(sampling_strategy='auto', random_state=42)
        xtr_res, ytr = smtom.fit_resample(arr, ytr)
        xtr = np.reshape(xtr_res, newshape=(xtr_res.shape[0], 247, 247, 1))

    # input_variables = xtr.shape[1]

    neurons = parameters['num_of_neurons']
    dropouts = parameters['dropout_vals']

    # Build model architecture
    model = Sequential()
    model.add(Conv2D(neurons[0], kernel_size=(3, 3), input_shape=(247, 247, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(neurons[1], kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(neurons[2], kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(neurons[3], kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(optimizer=Adam(learning_rate=parameters['lr']),
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'AUC', Precision(), Recall()])

    mcp = ModelCheckpoint('.mdl.wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')

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


base_path = "D:/PROCESSED_ADNI_CONTROL_GROUP/results/"
input_struct_NC = load_all_ConnMats(base_path)

base_path = "D:/PROCESSED_ADNI_AD_GROUP/PROCESSED_AD_GROUP/"
input_struct_AD = load_all_ConnMats(base_path)

x_train = np.concatenate((input_struct_AD, input_struct_NC), axis=0)
y_train = np.zeros((input_struct_NC.shape[0]+input_struct_AD.shape[0]))
y_train[input_struct_NC.shape[0]:input_struct_NC.shape[0]+input_struct_AD.shape[0]] = 1

base_path = "D:/TEST/"
x_test = load_all_ConnMats(base_path)

y_test = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

parameters = {'num_of_neurons': [128, 64, 32, 16],  # [layer0, layer1, layer2, ...]
              'dropout_vals': [0.5, 0.5, 0.5],
              'batch_size': 64,  # 32, 64, 128, ...
              'num_of_layers': 4,  # total = num_of_layers + 1
              'lr': 0.001,
              # 'sgd_momentum': 0.4371162594318422, # Just if SGD optimizer is selected
              'epoch': 30,
              'imbalanced': True}

history, model = cnn_model(parameters, x_train, y_train)
output_dir = './output_CNN/'
# model.save(output_dir + 'output_CNN_model.h5')

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='best')
plt.show()
plt.savefig(output_dir + 'loss')

plt.figure()
plt.plot(history.history['auc'])
plt.plot(history.history['val_auc'])
plt.title('AUC')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='best')
plt.show()

loss, accuracy, auc, pre, rec = model.evaluate(x_test, y_test, verbose=0)

f = open(output_dir + 'testing_results.txt', 'w')
print(f"Loss: {loss}\n"
      f"Accuracy: {accuracy}\n"
      f"AUC: {auc}\n"
      f"Precision: {pre}\n"
      f"Recall: {rec}\n", file=f)

print("\n------------------- EVALUATION RESULTS -------------------")
print(f"Loss: {loss}\n"
      f"Accuracy: {accuracy}\n"
      f"AUC: {auc}\n"
      f"Precision: {pre}\n"
      f"Recall: {rec}\n")
print("----------------------------------------------------------")
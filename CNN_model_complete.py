import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek
from scipy.io import loadmat
import numpy as np


def cnn_model(parameters, x_train, y_train, x_test, y_test):
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
        smtom = SMOTETomek(random_state=42)
        xtr, ytr = smtom.fit_resample(xtr, ytr)

    # input_variables = xtr.shape[1]

    neurons = parameters['num_of_neurons']
    dropouts = parameters['dropout_vals']

    # Build model architecture
    model = Sequential()
    model.add(Conv2D(neurons[0], kernel_size=(3, 3), input_shape=(247, 247, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(neurons[1], kernel_size=(3, 3), activation='relu'))
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


# x_train = pd.read_csv('C:\\', encoding='latin1')
# y_train = pd.read_csv('C:\\', encoding='latin1')
# x_test = pd.read_csv('C:\\', encoding='latin1')
# y_test = pd.read_csv('C:\\', encoding='latin1')

mat = loadmat("C:/Users/imano/Desktop/MU/PBL/PBL-NEUROMOD/ConnMat_sub-ADNI011S4827_ses-M60.mat")
mat = mat['myConnMat']
x_train = np.zeros((6, 247, 247, 1))
x_train[0, :, :, 0] = mat
x_train[1, :, :, 0] = mat
x_train[2, :, :, 0] = mat
x_train[3, :, :, 0] = mat
x_train[4, :, :, 0] = mat
x_train[5, :, :, 0] = mat
y_train = np.array([0, 0, 0, 0, 0, 0])

x_test = np.zeros((2, 247, 247, 1))
x_test[0, :, :, 0] = mat
x_test[1, :, :, 0] = mat
y_test = np.array([0, 0])

parameters = {'num_of_neurons': [32, 32],  # [layer0, layer1, layer2, ...]
              'dropout_vals': [0.5, 0.5, 0.5],
              'batch_size': 64,  # 32, 64, 128, ...
              'num_of_layers': 2,  # total = num_of_layers + 1
              'lr': 0.0001,
              # 'sgd_momentum': 0.4371162594318422, # Just if SGD optimizer is selected
              'epoch': 5000,
              'imbalanced': False}

history, model = cnn_model(parameters, x_train, y_train, x_test, y_test)
output_dir = './output_CNN/'
model.save(output_dir + 'output_CNN_model.h5')

scores = pd.DataFrame()
loss, accuracy, auc, pre, rec = model.evaluate(x_test, y_test, verbose=1)


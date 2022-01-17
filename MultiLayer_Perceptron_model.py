import numpy as np
import pandas as pd
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek

def mlp_model(parameters):
    """
    param parameters:
    return: model, history
    """

    # Check if parameters are correctly introduced
    if parameters['num_of_layers'] != (len(parameters['num_of_neurons']) or len(parameters['dropout_vals'])):
        raise Exception("Parameters are not consistent.")

    # Load data
    X_train = pd.read(...)
    Y_train = pd.read(...)
    X_test = pd.read(...)
    Y_test = pd.read(...)
    xtr, xval, ytr, yval = train_test_split(X_train, Y_train, test_size=0.2, random_state=42, stratify=Y_train)

    # If SMOTETomek is wanted to be applied
    if parameters['imbalanced'] == True:
        smtom = SMOTETomek(random_state=42)
        xtr, ytr = smtom.fit_resample(xtr, ytr)

    neurons = parameters['num_of_neurons']
    dropouts = parameters['dropout_vals']

    # Build model architecture
    model = Model.Sequential()
    model.add(Dense(neurons[0], input_shape=(inputVariables,), activation='relu'))

    for layer in range(0, len(parameters['num_of_layers']), 1):
        model.add(Dropout(dropouts[layer]))
        model.add(Dense(neurons[layer], activation='relu'))

    model.add(Dropout(0.8))
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

parameters = {'num_of_neurons': [8, 8, 8], # [layer0, layer1, layer2, ...]
              'dropout_vals': [0.5, 0.5, 0.5],
              'batch_size': 32, # 32, 64, 128, ...
              'num_of_layers': 4, # total = num_of_layers + 1
              'lr': 0.0001,
              # 'sgd_momentum': 0.4371162594318422, # Just if SGD optimizer is selected
              'epoch': 5000,
              'imbalanced': False}

mlp_model(parameters)
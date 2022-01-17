import numpy as np
import pandas as pd
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

def MLP(model_parameters):
    layer0 = model_parameters['Dense']

    model = Model.Sequential()
    model.add(Dense(layer0, input_shape=(inputVariables,), activation='relu'))

model_parameters = {'Dense': 5,
             'Dense_1': 2,
             'Dense_2': 0,
             'Dense_3': 3,
             'Dense_4': 3,
             'Dense_5': 6,
             'Dropout': 0.418858415012314,
             'Dropout_1': 0.6755306079803495,
             'Dropout_2': 0.3551235307624525,
             'Dropout_3': 0.031860071810303925,
             'Dropout_4': 0.0899677933324493,
             'Dropout_5': 0.2566022208036773,
             'batch_size': 1,
             'choiceval': 1,
             'layerChoice': 4,
             'lr': 0.0001,
             'sgd_momentum': 0.4371162594318422,
             'epoch': 5000}

MLP(model_parameters)
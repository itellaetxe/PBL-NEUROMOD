import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        arr = np.reshape(xtr, (xtr.shape[0], 247 * 247))
        smtom = SMOTETomek(sampling_strategy='auto', random_state=42)
        xtr_res, ytr = smtom.fit_resample(arr, ytr)
        xtr = np.reshape(xtr_res, newshape=(xtr_res.shape[0], 247, 247, 1))

    neurons = parameters['num_of_neurons']

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

    # Adding batch normalization layer
    # model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(64))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(optimizer=Adam(learning_rate=parameters['lr']),
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'AUC', Precision(), Recall()])

    # Only the best model throughout the training will be stored
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

OUTPUT_DIR = './output_CNN/'
METHOD = 'raw'

# Seed = 42
np.random.seed(42)
tf.random.set_seed(42)

# Loading the data

comments = 'Different testing batch selected'

base_path = "D:/PROCESSED_ADNI_CONTROL_GROUP/results/"
input_struct_NC, subs_NC = load_all_ConnMats(base_path, METHOD)

base_path = "D:/PROCESSED_ADNI_AD_GROUP/PROCESSED_AD_GROUP/"
input_struct_AD, subs_AD = load_all_ConnMats(base_path, METHOD)

subs_train = subs_NC + subs_AD

x_train = np.concatenate((input_struct_NC, input_struct_AD), axis=0)
y_train = np.zeros((input_struct_NC.shape[0] + input_struct_AD.shape[0]))
y_train[input_struct_NC.shape[0]:input_struct_NC.shape[0] + input_struct_AD.shape[0]] = 1

base_path = "D:/TEST/NC/"
x_test_NC, subs_NC = load_all_ConnMats(base_path, METHOD)

base_path = "D:/TEST/AD/"
x_test_AD, subs_AD = load_all_ConnMats(base_path, METHOD)

subs_test = subs_NC + subs_AD

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
              'epoch': 50,
              'imbalanced': False}

if len(set(subs_test).intersection(subs_train)) == 0:  # Check that testing data is not also being used for training
    history, model = cnn_model(parameters, x_train, y_train)
else:
    raise Exception("Training data is repeated in the testing set.")
# model.save(output_dir + 'output_CNN_model.h5')

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='best')
plt.savefig(OUTPUT_DIR + 'loss')
plt.show()

# plt.figure()
plt.plot(history.history['auc'])
plt.plot(history.history['val_auc'])
plt.title('AUC')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='best')
plt.savefig(OUTPUT_DIR + 'AUC')
plt.show()

loss, accuracy, auc, pre, rec = model.evaluate(x_test, y_test, verbose=0)
y_pred = model.predict(x_test, verbose=0)
cm = confusion_matrix(y_test, np.round(y_pred))
plt.figure()
categories = ['NC', 'AD']
sns.heatmap(cm, annot=True, cmap='Blues')
plt.savefig(OUTPUT_DIR + 'Conf_map')
plt.show()

# f = open(output_dir + 'testing_results.txt', 'w')
# print(f"Loss: {loss}\n"
#       f"Accuracy: {accuracy}\n"
#       f"AUC: {auc}\n"
#       f"Precision: {pre}\n"
#       f"Recall: {rec}\n", file=f)

print("\n------------------- EVALUATION RESULTS -------------------")
print(f"Loss: {loss}\n"
      f"Accuracy: {accuracy}\n"
      f"AUC: {auc}\n"
      f"Precision: {pre}\n"
      f"Recall: {rec}\n")
print("----------------------------------------------------------")

csv_tmp = pd.DataFrame({'Datetime': [datetime.now().strftime("%m/%d/%Y, %H:%M:%S")],
                        'num_of_neurons': [str(parameters['num_of_neurons'])],
                        'batch_size': [parameters['batch_size']],
                        'lr': [parameters['lr']],
                        'epoch': [parameters['epoch']],
                        'imbalanced': [parameters['imbalanced']],
                        'Loss': [loss],
                        'Acc': [accuracy],
                        'AUC': [auc],
                        'Pre': [pre],
                        'Rec': [rec],
                        'method': METHOD,
                        'comments': comments})

# trials=pd.DataFrame(columns=['Datetime', 'num_of_neurons', 'batch_size', 'lr', 'epoch',
# 'imbalanced', 'Loss', 'Acc', 'AUC', 'Pre', 'Rec'])
# trials.to_csv("trials.csv", index=False)

# Add the training metrics and parameters to the csv in order to keep track of them
csv_path = "C:/Users/imano/Desktop/MU/PBL/PBL-NEUROMOD/output_CNN/trials.csv"
csv = pd.read_csv(csv_path)
csv = pd.concat([csv, csv_tmp])
csv.to_csv(csv_path, index=False)

# 06/01/2022, 09:20:23 --> BatchNormalization layer added
# 06/01/2022, 10:20:14 --> Random seed changed to 0
# 06/01/2022, 10:23:25 --> Random seed changed to 7

clear_session()


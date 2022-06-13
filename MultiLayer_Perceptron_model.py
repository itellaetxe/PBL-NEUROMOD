import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek
import matplotlib.pyplot as plt
from datetime import datetime


def mlp_model(parameters, x_train, y_train):
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

    input_variables = xtr.shape[1]

    neurons = parameters['num_of_neurons']
    dropouts = parameters['dropout_vals']

    # Build model architecture
    model = Sequential()
    model.add(Dense(neurons[0], input_shape=(input_variables,), activation='relu'))

    for layer in range(0, parameters['num_of_layers'], 1):
        model.add(Dropout(dropouts[layer]))
        model.add(Dense(neurons[layer], activation='relu'))

    model.add(Dropout(0.8))
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

# df_train = pd.read_csv("C:/Users/imano/Desktop/MU/PBL/PBL-NEUROMOD/train_weightCM_table.csv", encoding='latin1', delimiter=',')
# df_train['Label'].replace({'AD': 1, 'CN': 0}, inplace=True)
# df_test = pd.read_csv("C:/Users/imano/Desktop/MU/PBL/PBL-NEUROMOD/test_weightCM_table.csv", encoding='latin1', delimiter=',')
# df_test['Label'].replace({'AD': 1, 'CN': 0}, inplace=True)
#
# x_train = df_train[['EfficiencyW', 'MeanBetwennessW', 'MeanClustCoffW', 'MeanStrength', 'kDensity', 'TransitivityW']]
# y_train = df_train['Label']
# x_test = df_test[['EfficiencyW', 'MeanBetweennessW', 'MeanClustCoffW', 'MeanStrength', 'kDensity', 'TransitivityW']]
# y_test = df_test['Label']

path = "C:/Users/imano/Desktop/MU/PBL/PBL-NEUROMOD/Voxel_metrics/FA_dual.csv"
df = pd.read_csv(path, encoding='latin1', delimiter=',')

# Just to use a subgroup
my_file = open("C:/Users/imano/Desktop/MU/PBL/PBL-NEUROMOD/Voxel_metrics/FA_dual_regs_FDR_0p05.txt", "r")
content = my_file.read()
subgroup = content.split("\n")
new_list = [s.replace('"', '') for s in subgroup]
new_list.append("Label")
new_list.remove('')
my_file.close()
df = df[new_list]

y_train = df['Label']
x_train = df.drop(['Label'], inplace=False, axis='columns')

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

parameters = {'num_of_neurons': [16, 8],  # [layer0, layer1, layer2, ...]
              'dropout_vals': [0.5, 0.5, 0.5],
              'batch_size': 8,  # 32, 64, 128, ...
              'num_of_layers': 2,  # total = num_of_layers + 1
              'lr': 0.001,
              # 'sgd_momentum': 0.4371162594318422, # Just if SGD optimizer is selected
              'epoch': 200,
              'imbalanced': True}

output_dir = './output_MLP/'
history, model = mlp_model(parameters, x_train, y_train)
# model.save(output_dir + 'output_MLP_model.h5')

loss, accuracy, auc, pre, rec = model.evaluate(x_test, y_test, verbose=0)
print("\n------------------- EVALUATION RESULTS -------------------")
print(f"Loss: {loss}\n"
      f"Accuracy: {accuracy}\n"
      f"AUC: {auc}\n"
      f"Precision: {pre}\n"
      f"Recall: {rec}\n")
print("----------------------------------------------------------")

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='best')
plt.savefig(output_dir + 'loss')
plt.show()

plt.figure()
plt.plot(history.history['auc'])
plt.plot(history.history['val_auc'])
plt.title('AUC')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='best')
plt.savefig(output_dir + 'auc')
plt.show()

csv_tmp = pd.DataFrame({'Datetime': [datetime.now().strftime("%m/%d/%Y, %H:%M:%S")],
                        'num_of_neurons': [str(parameters['num_of_neurons'])],
                        'dropout_vals': [str(parameters['dropout_vals'])],
                        'batch_size': [parameters['batch_size']],
                        'lr': [parameters['lr']],
                        'epoch': [parameters['epoch']],
                        'imbalanced': [parameters['imbalanced']],
                        'Loss': [loss],
                        'Acc': [accuracy],
                        'AUC': [auc],
                        'Pre': [pre],
                        'Rec': [rec]})

# Add the training metrics and parameters to the csv in order to keep track of them
csv_path = "C:/Users/imano/Desktop/MU/PBL/PBL-NEUROMOD/output_MLP/trials.csv"
csv = pd.read_csv(csv_path)
csv = pd.concat([csv, csv_tmp])
csv.to_csv(csv_path, index=False)


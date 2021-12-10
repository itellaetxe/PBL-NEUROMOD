# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 12:50:24 2021

@author: IMANOL
"""

import pandas as pd
import numpy as np

dataset = pd.read_csv("./idaSearch_12_09_2021.csv")
dataset.columns

for sample, idx in zip(dataset['Research Group'], range(0, len(dataset['Research Group']))):
    if sample == 'EMCI' or sample == 'LMCI':
        dataset['Research Group'][idx] = 'MCI'

MCIcount=0
SMCcount=0
CNcount=0
ADcount=0
for sample in dataset['Research Group']:
    if sample == 'MCI':
        MCIcount += 1
    if sample == 'SMC':
        SMCcount += 1
    if sample == 'CN':
        CNcount += 1
    if sample == 'AD':
        ADcount += 1

descriptionTypes = np.unique(dataset['Description'])
print(descriptionTypes)

patientNum = np.unique(dataset['Subject ID'])
print(len(patientNum))




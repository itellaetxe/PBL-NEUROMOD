# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 12:50:24 2021

@author: IMANOL
"""

import pandas as pd

datasetV1 = pd.read_csv("./Trial_1_11_04_2021.csv")
datasetV1.columns

rowsToDelete = []
for sample in range(datasetV1.shape[0]):
    mod = datasetV1['Modality']
    if mod[sample] != 'DTI':
        rowsToDelete.append(sample)

datasetDTI = datasetV1.drop(rowsToDelete, axis=0, inplace=False)
datasetDTI.drop('Modality', axis=1, inplace=True)
datasetDTI.reset_index(drop=True)
a = 1
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 09:12:02 2021

@author: IMANOL
"""

import pandas as pd
import numpy as np

dataset = pd.read_csv("./idaSearch_12_09_2021.csv")

# Set both EMCI and LMCI as MCI
for sample, idx in zip(dataset['Research Group'], range(0, len(dataset['Research Group']))):
    if sample == 'EMCI' or sample == 'LMCI':
        dataset['Research Group'][idx] = 'MCI'

# Delete all structural (MPRAGE) data from dataset        
descr = list(dataset['Description'])
locs = dataset.loc[dataset['Description'].str.contains('MPRAGE')]
rowsToDelete = locs.index

dataset.drop(rowsToDelete, axis='index', inplace=True)

#%% FILTERING DTI SAMPLES BASED ON DESCRIPTION TAGS

# Save 'Axial DTI' tags in new dataframe
descr = list(dataset['Description'])
locs = dataset.loc[dataset['Description'] == 'Axial DTI']
outputDTI = pd.DataFrame(locs)
rowsToDelete = locs.index
dataset.drop(rowsToDelete, axis='index', inplace=True)
dataset.reset_index(inplace=True)
outputDTI.reset_index(inplace=True)

# Save 'Axial MB DTI' tags in new dataframe 
descr = list(dataset['Description'])
locs = dataset.loc[dataset['Description'] == 'Axial MB DTI']
outputDTI2 = pd.DataFrame(locs)
rowsToDelete = locs.index
dataset.drop(rowsToDelete, axis='index', inplace=True)
dataset.reset_index(inplace=True)
outputDTI2.reset_index(inplace=True)

# Concatenate both 'Axial DTI' and 'Axial MB DTI'dataframes into one
out = pd.concat([outputDTI, outputDTI2])

# Delete Subject IDs if they are already present in the new dataframe to avoid
# redundancy and derived data
for i in range(0, len(outputDTI['Subject ID'])):
    repeated_IDs = dataset.loc[dataset['Subject ID'].str.contains(outputDTI['Subject ID'][i])]
    dataset.drop(repeated_IDs.index, axis='index', inplace=True)
    
out.drop(['index','level_0'], axis='columns', inplace=True)
out.reset_index(inplace=True)
out.drop(['index'], axis='columns', inplace=True)
    
#%% EVALUATE THE RESULTS

print(out['Research Group'].value_counts())
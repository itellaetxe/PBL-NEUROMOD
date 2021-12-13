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

#%% T1 REDUNDANCY REDUCTION
dataset = pd.read_csv("./idaSearch_12_09_2021.csv")
# How many image description types?
descriptionTypes = np.unique(dataset['Description'])
print(descriptionTypes)

# How many patients do we have?
patientNum = np.unique(dataset['Subject ID'])
print(len(patientNum))

# Check if patients have MPRAGE or IR-FSPGR images
contains_MPRAGE = dataset.loc[dataset['Description'].str.contains('MPRAGE')]
contains_IR_FSPGR = dataset.loc[dataset['Description'].str.contains('IR-FSPGR')]

id_BOTH = contains_MPRAGE.loc[dataset['Description'].str.contains('IR-FSPGR')]
# There's no patient with both MPRAGE and IR-FSPGR images. Visual inspection of IR-FSPGR images show
# good resolution and look good for registration purposes (That's why we want T1 stuff, anyways)

# Check if MPRAGE cases have more than one image of this modality
MPRAGE_unique_idxs = []
aux_list = []
for sample, idx in zip(contains_MPRAGE['Subject ID'], range(0, len(contains_MPRAGE['Subject ID']))):
    if sample not in aux_list:
        aux_list.append(sample)
        MPRAGE_unique_idxs.append(contains_MPRAGE.index[idx])

# There were 653 MPRAGE images in the dataset, we got 524 non-repeated occurrences
# meaning that we discarded (653-524) MPRAGE images that would be redundant.

# Check if IR-FSPGR cases have more than one image of this modality
FSPGR_unique_idxs = []
aux_list = []
for sample, idx in zip(contains_IR_FSPGR['Subject ID'], range(0, len(contains_IR_FSPGR['Subject ID']))):
    if sample not in aux_list:
        aux_list.append(sample)
        FSPGR_unique_idxs.append(contains_IR_FSPGR.index[idx])
# There were 124 IR-FSPGR images in the dataset, we got 117 non-repeated occurrences
# meaning that we discarded (124-117) images
unique_ids = []
for idx in MPRAGE_unique_idxs:








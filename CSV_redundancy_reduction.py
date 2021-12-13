# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 09:12:02 2021

@author: IMANOL

This script is thought to perform an early filtering of the data to be
downloaded. A .csv file from ADNI is downloaded and the samples that are interesting
for us are filtered.

Once the list of interest is generated, the files that are not adecuate are deleted
from the computer by adding the correct path.
"""

import shutil
import pandas as pd
import numpy as np

dataset = pd.read_csv("./idaSearch_12_09_2021.csv")

#%% FILTERING DTI-RELATED DATA

# Set both EMCI and LMCI as MCI
for sample, idx in zip(dataset['Research Group'], range(0, len(dataset['Research Group']))):
    if sample == 'EMCI' or sample == 'LMCI':
        dataset['Research Group'][idx] = 'MCI'

# Delete all structural (MPRAGE) data from dataset        
descr = list(dataset['Description'])
locs = dataset.loc[dataset['Description'].str.contains('MPRAGE')]
rows_to_delete = locs.index
dataset.drop(rows_to_delete, axis='index', inplace=True)

"""FILTERING DTI SAMPLES BASED ON DESCRIPTION TAGS"""

# Save 'Axial DTI' tags in new dataframe
descr = list(dataset['Description'])
locs = dataset.loc[dataset['Description'] == 'Axial DTI']
output_DTI = pd.DataFrame(locs)
rows_to_delete = locs.index
dataset.drop(rows_to_delete, axis='index', inplace=True)
dataset.reset_index(inplace=True)
output_DTI.reset_index(inplace=True)

# Save 'Axial MB DTI' tags in new dataframe 
descr = list(dataset['Description'])
locs = dataset.loc[dataset['Description'] == 'Axial MB DTI']
output_DTI2 = pd.DataFrame(locs)
rows_to_delete = locs.index
dataset.drop(rows_to_delete, axis='index', inplace=True)
dataset.reset_index(inplace=True)
output_DTI2.reset_index(inplace=True)

# Concatenate both 'Axial DTI' and 'Axial MB DTI'dataframes into one
out = pd.concat([output_DTI, output_DTI2])

# Delete Subject IDs if they are already present in the new dataframe to avoid
# redundancy and derived data
for i in range(0, len(output_DTI['Subject ID'])):
    repeated_IDs = dataset.loc[dataset['Subject ID'].str.contains(output_DTI['Subject ID'][i])]
    dataset.drop(repeated_IDs.index, axis='index', inplace=True)
    
out.drop(['index','level_0'], axis='columns', inplace=True)
out.reset_index(inplace=True)
out.drop(['index'], axis='columns', inplace=True)

#%% T1 REDUNDANCY REDUCTION

dataset = pd.read_csv("./idaSearch_12_09_2021.csv")

# How many image description types?
descriptionTypes = np.unique(dataset['Description'])
#print(descriptionTypes)

# How many patients do we have?
patientNum = np.unique(dataset['Subject ID'])
#print(len(patientNum))

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

#%% COMBINING BOTH DTI AND T1 RELATED FILTERINGS IN ONE
out2 = []
id_list = list(out['Subject ID'])

"""Indexes are converted into ID numbers, and stored in out2"""

for i in MPRAGE_unique_idxs:
    out2.append(dataset['Subject ID'][i])
    
for i in FSPGR_unique_idxs:
    out2.append(dataset['Subject ID'][i])

"""Duplicated values from each list are erased. IDs that are not present in
   both lists are alse deleted."""
   
l1 = []
[l1.append(x) for x in id_list if x not in l1]
l2 = []
[l2.append(x) for x in out2 if x not in l2]
        
for ID in l1:
    if ID not in l2:
        l1.remove(ID) # l1 is the list of interest, which is equal to out

a = out.duplicated(subset=['Subject ID'], keep='last')
rows_to_delete = list(a.index[a==True])
out.drop(rows_to_delete, inplace=True, axis='rows')

""" Evaluation of the output"""

print(out['Research Group'].value_counts())

#%% DELETING UNWANTED SUBJECT FILES FROM STORAGE PATH

path = "C:\\Users\\IMANOL\\Desktop\\MU\\" # must modify
list_all = list(dataset['Subject ID'])

# If the subject ID has not been selected, its directory will be deleted
for subject in list_all:
    if subject not in l1: 
        shutil.rmtree(path+str(subject), ignore_errors=True)
        print("Directory "+path+str(subject)+" was deleted.")



# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 12:50:24 2021

@author: IMANOL
"""

import pandas as pd
import numpy as np

dataset = pd.read_csv("./idaSearch_12_09_2021.csv")
dataset.columns

#
for sample, idx in zip(dataset['Research Group'], range(0, len(dataset['Research Group']))):
    if sample == 'EMCI' or sample == 'LMCI':
        dataset['Research Group'][idx] = 'MCI'

# Cognitive status counting
MCIcount = 0
SMCcount = 0
CNcount = 0
ADcount = 0
for sample in dataset['Research Group']:
    if sample == 'MCI':
        MCIcount += 1
    if sample == 'SMC':
        SMCcount += 1
    if sample == 'CN':
        CNcount += 1
    if sample == 'AD':
        ADcount += 1

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
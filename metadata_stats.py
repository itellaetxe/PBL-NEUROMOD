import pandas as pd
import numpy as np
import scipy.stats as stats

# Read data ADNIMERGE simplified metadata
rmetadata = pd.read_excel("patientmetadata.xlsx")
metadata = rmetadata[rmetadata["COLPROT"].str.contains("ADNI3")]  # Filter by ADNI3 project.
metadata = rmetadata
# Separate AD and CN groups
AD = metadata[metadata["DX_bl"] == "AD"]
CN = metadata[metadata["DX_bl"] == "CN"]

# Take NaN values of AGE, ADAS and MMSE from the AGE-,ADAS- & MMSE- *_bl column values
zutabek = AD.columns[5:]
for i in range(0, 3):
    idxs = AD[zutabek[i]].isna()
    AD.loc[idxs, zutabek[i]] = AD.loc[idxs, zutabek[i+3]]  # Caveat warning from pandas... ignore...
    idxs = CN[zutabek[i]].isna()
    CN.loc[idxs, zutabek[i]] = CN.loc[idxs, zutabek[i+3]]

# Impute the NaN values left with the mean of the column
for i in range(0, 3):
    AD[zutabek[i]].fillna(AD[zutabek[i]].mean(), inplace=True)
    CN[zutabek[i]].fillna(CN[zutabek[i]].mean(), inplace=True)

# Get our AD and CN subject lists
AD_md = pd.read_csv("AD_GROUP_ADNI3.csv")  # "md" stands for metadata
AD_md = AD_md[AD_md["Modality"] == "DTI"]
AD_md = AD_md[AD_md["Group"] == "AD"]
AD_ids = AD_md["Subject"]
AD_sex = AD_md["Sex"]

CN_md = pd.read_csv("CN_CONTROL_GROUP_ADNI3.csv")  # "md" stands for metadata
CN_md = CN_md[CN_md["Modality"] == "DTI"]  # Exclude duplicate entries
CN_md = CN_md[CN_md["Group"] == "CN"]  # 6 MCI cases leaked to the CN group. idk how or why tbh...
CN_ids = CN_md["Subject"]
CN_sex = CN_md["Sex"]

# Filter our subjects from the big tables
AD_filtered = AD.loc[AD["PTID"].isin(AD_ids)]
AD_filtered.reset_index(drop=True, inplace=True)
CN_filtered = CN.loc[CN["PTID"].isin(CN_ids)]
CN_filtered.reset_index(drop=True, inplace=True)

"""
When selecting the patients, for each of them, only their last visit was taken for the dataset.
Because of this, we will get rid of the repeated PTID entries, leaving only the last entry, corresponding to
the last visit of the given patient.
"""
unique_ad_ids = pd.unique(AD_filtered["PTID"])
unique_cn_ids = pd.unique(CN_filtered["PTID"])


# Chi-Square test for SEX in CN and AD groups
# Contingency table for SEX
df = pd.DataFrame(columns=["Sex", "Class"])
df["Sex"] = pd.concat([CN_sex, AD_sex], axis=0)
df["Class"] = pd.concat([CN_md["Group"], AD_md["Group"]], axis=0)
sex_crosstab = pd.crosstab(df["Class"], df["Sex"], margins=True, margins_name="Total")
_, p_sex, _, _ = stats.chi2_contingency(sex_crosstab)  # Chi-Square test for SEX

# T-test of differences for AGE
AD_age = AD_md["Age"]
CN_age = CN_md["Age"]
_, p_age_var = stats.bartlett(AD_age, CN_age)  # Test if equal variance
if p_age_var > 0.05:
    var_bool = True  # Equal variance, null hypothesis cannot be rejected
else:
    var_bool = False  # Unequal variance, null hypothesis rejected

_, p_age = stats.ttest_ind(AD_age, CN_age, equal_var=var_bool)

# T-test of differences for MMSE


# T-test of differences for ADAS11


# T-test of differences for ADAS13




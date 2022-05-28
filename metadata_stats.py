import pandas as pd
import numpy as np
import scipy.stats as stats

# Read data ADNIMERGE simplified metadata
rmetadata = pd.read_excel("patientmetadata.xlsx")
metadata = rmetadata[rmetadata["COLPROT"].str.contains("ADNI3")]
# Separate AD and CN groups
AD = metadata[metadata["DX_bl"] == "AD"]
CN = metadata[metadata["DX_bl"] == "CN"]

# Take NaN values of AGE, ADAS and MMSE from the *_bl column values
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
AD_md = AD_md[AD_md["Group"] == "AD"]
AD_ids = AD_md["Subject"]
AD_sex = AD_md["Sex"]

CN_md = pd.read_csv("CN_CONTROL_GROUP_ADNI3.csv")  # "md" stands for metadata
CN_md = CN_md[CN_md["Group"] == "CN"]  # 6 MCI cases leaked to the CN group. idk how or why tbh...
CN_ids = CN_md["Subject"]
CN_sex = CN_md["Sex"]

# Filter our subjects from the big tables
AD_filtered = AD.loc[AD["PTID"].isin(AD_ids)]
CN_filtered = CN.loc[CN["PTID"].isin(CN_ids)]

# Chi-Square test for sex in CN and AD groups
# Contingency table
df = pd.DataFrame(columns=["Sex", "Class"])
df["Sex"] = pd.concat([CN_sex, AD_sex], axis=0)
df["Class"] = pd.concat([CN_md["Group"], AD_md["Group"]], axis=0)
data_crosstab = pd.crosstab(df["Class"], df["Sex"], margins=True, margins_name="Total")
# significance level
alpha = 0.05

# Chi-Square test for SEX
stat, p, dof, expected = stats.chi2_contingency(data_crosstab)


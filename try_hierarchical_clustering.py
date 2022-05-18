import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import scipy
from scipy.io import loadmat
import scipy.cluster.hierarchy as sch
import scipy.sparse as sps

from atlas_dict import number2parcel


def calculate_sparsity(df):
    rows, cols = df.corr().shape
    total_elems = rows*cols

    NNZ = len(sps.find(df)[0])

    spars = (total_elems - NNZ)/total_elems
    return spars


df = loadmat("D:/PROCESSED_ADNI_AD_GROUP/PROCESSED_AD_GROUP/sub-ADNI013S6768_ses-M00/ConnMat/ConnMat_sub-ADNI013S6768_ses-M00.mat")
df = pd.DataFrame(np.array(df['myConnMat']))
# df = df.rename(columns=number2parcel.get_names_dict, inplace=False)

spars = calculate_sparsity(df)
print(f"Sparsity of this connectivity matrix is {spars}")
# The emptier the matrix, the more the spars value will aproximate to 0.

fig = plt.figure()
ax1 = plt.subplot(221)
ax1.matshow(np.log(df.corr() +1), cmap='RdYlGn')
ax1.set_title("Original ConnMat")
fig.colorbar(ax1.pcolormesh(np.random.random((1, 1)) * 100, cmap='RdYlGn',
                            vmin=-1, vmax=1),ax=ax1, ticks=[-1, 0, 1], aspect=40, shrink=.8)

X=df.corr()
X[pd.isna(X)] = 0
X = X.values

d = sch.distance.pdist(X)   # vector of ('55' choose 2) pairwise distances
L = sch.linkage(d, method='complete')
ax2 = plt.subplot(212)
dn = sch.dendrogram(L)
ax2.set_title("Dendrogram")
ind = sch.fcluster(L, t=0.005*d.max(), criterion='inconsistent', depth=20)
columns = [df.columns.tolist()[i] for i in list((np.argsort(ind)))]
df = df.reindex(columns, axis=1)

ax3 = plt.subplot(222)
corr = df.corr()

# Plot the correlation matrix
# fig, ax = plt.subplots(figsize=(size, size))
ax3.matshow(corr, cmap='RdYlGn')
ax3.set_title("Clustered ConnMat")
fig.colorbar(ax3.pcolormesh(np.random.random((1, 1)) * 100, cmap='RdYlGn',
                            vmin=-1, vmax=1) ,ax=ax3, ticks=[-1, 0, 1], aspect=40, shrink=.8)

plt.show()


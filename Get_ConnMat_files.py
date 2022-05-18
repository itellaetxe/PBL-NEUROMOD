import os
import re
from scipy.io import loadmat
import pandas as pd
import numpy as np
from atlas_dict import number2parcel
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch



def calculate_sparsity(df):
    rows, cols = df.shape
    total_elems = rows*cols

    # NNZ = len(sps.find(df.corr())[0])

    NNZ = (df == 0).sum().sum()
    spars = (total_elems - NNZ)/total_elems
    return spars


base_path = "D:/PROCESSED_ADNI_AD_GROUP/PROCESSED_AD_GROUP/"
dirs = os.listdir(base_path)
r = re.compile("sub-")
filtered = [folder for folder in dirs if r.match(folder)]

subs = [base_path + sub for sub in filtered]
subs = [sub + "/ConnMat" for sub in subs]

csv_paths = []
for sub in subs:
    tmp_dir = os.listdir(sub)
    csv_path = [file for file in tmp_dir if ".mat" in file][0]
    csv_paths.append(sub + "/" + csv_path)

print(csv_paths)

dfs = np.zeros((247, 247, 50))
for i, file in enumerate(csv_paths):
    mat_file = loadmat(file)
    df_tmp = np.array(mat_file['myConnMat'])
    dfs[:,:,i] = df_tmp

# for i in range(0, dfs.shape[2], 1):
for i in range(0, 10, 1):
    df2 = pd.DataFrame(dfs[:,:,i])
    # df2 = tmp_mat.rename(number2parcel.get_names_dict, axis='columns')
    # print(calculate_sparsity(df2))

    X = df2.corr()
    X[pd.isna(X)] = 0
    X = X.values

    d = sch.distance.pdist(X)  # vector of ('55' choose 2) pairwise distances
    L = sch.linkage(d, method='complete')
    # ax2 = plt.subplot(212)
    # dn = sch.dendrogram(L)
    # ax2.set_title("Dendrogram")
    ind = sch.fcluster(L, t=0.005 * d.max(), criterion='inconsistent', depth=20)
    columns = [df2.columns.tolist()[i] for i in list((np.argsort(ind)))]
    df3 = df2.reindex(columns, axis=1)

    # Plot the correlation matrix
    # fig, ax = plt.subplots(figsize=(size, size))
    plt.matshow(df3.corr(), cmap='RdYlGn', fignum=0)
    plt.title("Clustered ConnMat")
    # fig.colorbar(ax3.pcolormesh(np.random.random((1, 1)) * 100, cmap='RdYlGn',
    #                             vmin=-1, vmax=1), ax=ax3, ticks=[-1, 0, 1], aspect=40, shrink=.8)

    plt.show()

# df_now = df3.corr()
# df_now2 = df_now.rename(number2parcel.get_names_dict, axis='columns', inplace=False)
# left = [var for var in df_now2.columns if '_l' in var]
# right = [var for var in df_now2.columns if '_r' in var]
# plt.matshow(df_now2[left])
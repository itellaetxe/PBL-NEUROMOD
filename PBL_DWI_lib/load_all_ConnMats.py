import os
import re
import numpy as np
from scipy.io import loadmat


def load_all_ConnMats(base_path, method):
    dirs = os.listdir(base_path)
    r = re.compile("sub-")
    filtered = [folder for folder in dirs if r.match(folder)]

    subs = [base_path + sub for sub in filtered]
    subs = [sub + "/ConnMat" for sub in subs]

    mat_paths = []
    for sub in subs:
        tmp_dir = os.listdir(sub)
        tmp_path = [file for file in tmp_dir if ".mat" in file][0]
        tmp_path = sub + "/" + tmp_path
        mat_paths.append(tmp_path)

    input_struct = np.zeros((len(mat_paths), 247, 247, 1))
    for cnt, mat in enumerate(mat_paths):
        tmp_mat = loadmat(mat)
        tmp_mat = tmp_mat['myConnMat']

        if method == 'raw':
            input_struct[cnt, :, :, 0] = tmp_mat
        elif method == 'weighted':
            input_struct[cnt, :, :, 0] = tmp_mat/np.max(abs(tmp_mat))
        elif method == 'upper_raw':
            input_struct[cnt, :, :, 0] = np.triu(tmp_mat)

    return input_struct, filtered
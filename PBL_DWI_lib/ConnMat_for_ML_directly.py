import numpy as np
from sklearn.decomposition import PCA
from PBL_DWI_lib.load_all_ConnMats import *
from PBL_DWI_lib.number2parcel import *
import matplotlib.pyplot as plt


class Cm4ml:

    def __init__(self, numpy_4d_mat, num_of_parcels):
        self.numpy_4d_mat = numpy_4d_mat
        self.num_of_parcels = num_of_parcels

    def rearange_ConnMats(self):
        df_new = np.zeros((self.numpy_4d_mat.shape[0], self.numpy_4d_mat.shape[1] * self.numpy_4d_mat.shape[2]))

        for i in range(0, self.numpy_4d_mat.shape[0], 1):
            tmp_vect = np.reshape(self.numpy_4d_mat[i, :, :, 0],
                                  newshape=(1, self.numpy_4d_mat.shape[1] * self.numpy_4d_mat.shape[2]))
            df_new[i, :] = tmp_vect

        col_sums = np.sum(df_new, axis=0)
        ind = np.argpartition(col_sums, kth=-self.num_of_parcels)[-self.num_of_parcels:]
        print(ind)

        return df_new[:, ind]

    def get_most_connected_parcels(self, num_of_components):
        sum_mat = np.squeeze(np.sum(self.numpy_4d_mat, axis=0), axis=2)
        sum_list = np.sum(sum_mat, axis=0)

        idx = (-sum_list).argsort()[:num_of_components]
        [print(f"{get_names_dict.get(i)}") for i in idx]

        return idx

    def pca_from_rearanged(self, num_of_components, scree):
        df_new = np.zeros((self.numpy_4d_mat.shape[0], self.numpy_4d_mat.shape[1] * self.numpy_4d_mat.shape[2]))

        for i in range(0, self.numpy_4d_mat.shape[0], 1):
            tmp_vect = np.reshape(self.numpy_4d_mat[i, :, :, 0],
                                  newshape=(1, self.numpy_4d_mat.shape[1] * self.numpy_4d_mat.shape[2]))
            df_new[i, :] = tmp_vect

        col_sums = np.sum(df_new, axis=0)
        ind = np.argpartition(col_sums, kth=-self.num_of_parcels)[-self.num_of_parcels:]
        df_new = df_new[:, ind]

        n_samples = df_new.shape[0]

        pca = PCA(n_components=num_of_components)
        eigenvectors = pca.fit_transform(df_new)

        df_centered = df_new - np.mean(df_new, axis=0)
        cov_matrix = np.dot(df_centered.T, df_centered) / n_samples
        eigenvalues = pca.explained_variance_
        var_explainability = pca.explained_variance_ratio_
        cum_sum_eigenvalues = np.cumsum(var_explainability)

        # for eigenvalue, eigenvector in zip(eigenvalues, pca.components_):
            # print(np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector)))
            # print(eigenvalue)

        if scree is True:
            thr = 0.9
            idx = next(i for i, v in enumerate(cum_sum_eigenvalues) if v > thr)

            plt.bar(list(range(1, num_of_components + 1, 1)), eigenvalues, label="Eigenvalues")
            plt.step(list(range(1, num_of_components + 1, 1)), cum_sum_eigenvalues, color="red",
                     label="Cumulative explained variance")
            plt.plot(idx + 1, thr, marker="*", markersize=10, markeredgecolor="green", markerfacecolor="green")
            plt.xticks(np.arange(1, num_of_components + 1, step=1))
            plt.xlabel("Principal component")
            plt.ylabel("Eigenvalue")
            plt.xlim((0, num_of_components + 1))
            plt.title("Scree plot")
            plt.legend()
            plt.show()

        return eigenvectors, eigenvalues, var_explainability


method = 'weighted'
base_path = "D:/PROCESSED_ADNI_CONTROL_GROUP/results/"
input_struct_NC = load_all_ConnMats(base_path, method)

base_path = "D:/PROCESSED_ADNI_AD_GROUP/PROCESSED_AD_GROUP/"
input_struct_AD = load_all_ConnMats(base_path, method)

x_train = np.concatenate((input_struct_NC, input_struct_AD), axis=0)
y_train = np.zeros((input_struct_NC.shape[0] + input_struct_AD.shape[0]))
y_train[input_struct_NC.shape[0]:input_struct_NC.shape[0] + input_struct_AD.shape[0]] = 1

trial = Cm4ml(x_train, 50)
# trial2 = trial.rearange_ConnMats()
# trial3 = trial.pca_from_rearanged(num_of_components=25, scree=True)
trial4 = trial.get_most_connected_parcels(50)
print("")


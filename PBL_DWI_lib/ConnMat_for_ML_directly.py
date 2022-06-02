import numpy as np
from sklearn.decomposition import PCA


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

        return df_new[:, ind]


    def pca_from_rearanged(self):
        n_samples = df.shape[0]

        pca = PCA(n_components=number_of_comps)
        eigenvectors = pca.fit_transform(df)

        df_centered = df - np.mean(df, axis=0)
        cov_matrix = np.dot(df_centered.T, df_centered) / n_samples
        eigenvalues = pca.explained_variance_
        var_explainability = pca.explained_variance_ratio_
        cum_sum_eigenvalues = np.cumsum(var_explainability)

        for eigenvalue, eigenvector in zip(eigenvalues, pca.components_):
            # print(np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector)))
            print(eigenvalue)

        return eigenvectors, eigenvalues, var_explainability



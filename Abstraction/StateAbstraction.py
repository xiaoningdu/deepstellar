import numpy as np
from sklearn.decomposition import PCA
import time
import joblib
import _pickle as pickle
import os
from Abstraction.Coder import Coder


class StateAbstraction:
    def __init__(self, state_profile_folder, comp_num, bits, par_k, n_step):
        self.state_profile_folder = state_profile_folder
        self.comp_num = comp_num
        self.profile_file_list = get_all_file(state_profile_folder)
        self.cache_dir = os.path.join(state_profile_folder, "cache")
        self.pca_trans_dir = os.path.join(state_profile_folder, "pca_trans")
        self.pca_model_f = os.path.join(self.cache_dir, 'pca_model_cmp_%s.joblib' % self.comp_num)
        self.diag_matrix_f = os.path.join(self.cache_dir, 'diag_matrix.npy')
        self.min_array_f = os.path.join(self.cache_dir, 'min_array.npy')

        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
            os.makedirs(self.pca_trans_dir, exist_ok=True)
            self.pca_fit()
            self.pca_trans()
            self.get_quantization_matrix()

        self.pca_model = joblib.load(self.pca_model_f)
        self.diag_matrix = np.load(self.diag_matrix_f)
        self.min_array = np.load(self.min_array_f)

        self.bits = bits
        self.par_k = par_k
        self.dimension = len(par_k)
        self.n_step = n_step
        self.min_array = self.min_array[range(self.dimension)]  # tailor to the dimension
        self.diag_matrix = self.diag_matrix[:, range(self.dimension)][range(self.dimension), :]  # tailor to the dimension
        self.diag_matrix = self.diag_matrix.dot(np.diag(par_k))  # prepare par_k/range
        self.coder = Coder(bits, self.dimension)  # init a Coder for the encoding and decoding

    def pca_fit(self):
        """
        Read data from the data_repo and calculate the first comp_num principal components.
        For choose to sample the data before fitting PCA model
        """

        # read from data repo
        all_sample_data = []
        for f in self.profile_file_list:
            sample_chunk = np.load(os.path.join(self.state_profile_folder, f))
            all_sample_data.extend(sample_chunk)

        # fitting PCA model and save the model to the 'cache' folder under the data_repo
        start = time.time()
        pca = PCA(n_components=self.comp_num, copy=False)
        pca.fit(np.array([e for l in all_sample_data for e in l]))
        joblib.dump(pca, self.pca_model_f)
        print('pca fitting used %s ...' % (time.time() - start))

    def pca_trans(self):
        """
        Transform all the data with the PCA model and save the transformed data to pca_trans folder inside the repo folder
        """
        pca = joblib.load(self.pca_model_f)
        for f in self.profile_file_list:
            sample_chunk = np.load(os.path.join(self.state_profile_folder, f))
            sample_chunk_pca = []
            for sample in sample_chunk:
                sample_pca = pca.transform(np.array(sample))
                sample_chunk_pca.append(sample_pca)
            np.save(os.path.join(self.pca_trans_dir, f), sample_chunk_pca)
        print('pca_trans finished.')

    def get_quantization_matrix(self):
        """
        Read the PCA-transformed data, and calculate the auxiliary matrix for quantization
        """
        fit_data = self.get_pca_trans_data()
        fit_data = np.array([s for seq in fit_data for s in seq])
        print('fit data shape:')
        print(fit_data.shape)

        diag_array = []  # holding the reciprocal of each dimension on the diagonal
        min_array = []  # holding the minimum value of each dimension
        for i in range(fit_data.shape[1]):
            proj_i = [e[i] for e in fit_data]
            diag_array.append(1 / (max(proj_i) - min(proj_i)))
            min_array.append(min(proj_i))
            # print('%s--%s' % (min(proj_i), max(proj_i)))
        diag_matrix = np.diag(diag_array)

        np.save(self.diag_matrix_f, diag_matrix)
        np.save(self.min_array_f, min_array)

    def data_transform(self, seq, pca_transform=False):
        """
        return the sequence of abstracted state name
        """
        if pca_transform:
            seq = self.pca_model.transform(np.array(seq))
        seq = seq[:, range(self.dimension)]  # take the dimension
        my_min = np.repeat(self.min_array, len(seq))
        my_min = my_min.reshape(self.dimension, len(seq)).transpose()
        seq = seq - my_min  # each vector minus the lower bound
        pca_fit_partition = np.floor(seq.dot(self.diag_matrix)).astype(int)  # (vec-min)/(range/par_k) and take the floor value
        pca_fit_partition = pca_fit_partition + self.n_step  # to avoid negative encoding
        transition_seq_name = [self.coder.encode(a) for a in pca_fit_partition]  # encode the abstracted vectors
        # print(transition_seq_name)
        # transition_seq_name = [self.fake_initial] + transition_seq_name  # fake initial as starting state
        del my_min
        del seq
        del pca_fit_partition
        return transition_seq_name

    def pca_transform(self, seq):
        return self.pca_model.transform(np.array(seq[0]))

    def get_pca_trans_data(self):
        pca_fit = []
        data_fs = get_all_file(self.pca_trans_dir)
        for f in data_fs:
            chunk = np.load(os.path.join(self.pca_trans_dir, f))
            pca_fit.extend(chunk)
            # break
        return pca_fit


def get_all_file(target_dir):
    """
    A util function to return all files under a dir
    :param target_dir: the target folder
    :return: the set of files with name
    """
    onlyfiles = [f for f in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, f))]
    return onlyfiles


def load_graph_pkl(pkl_dir):
    with open(pkl_dir, 'rb') as f:
        g = pickle.load(f)
    return g





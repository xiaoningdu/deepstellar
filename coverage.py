import numpy as np
import pickle
from Abstraction.GraphWrapper import GraphWrapper


class Coverage(object):

    def __init__(self, pkl_dir, mode, k_step):
        self.par_wrap = load_graph_pkl(pkl_dir)
        if mode == 'state':
            self.total_size = self.par_wrap.graph.get_major_states_num()
            print('There are %s major states in total.' % self.total_size)
        elif mode == 'k-step':
            if k_step > self.par_wrap.stateAbst.n_step:
                print('this step is larger than the steps kept, please rebuild the model.')
                exit(0)
            self.par_wrap.graph.init_k_step_idx(k_step)
            self.total_size = self.par_wrap.graph.get_k_step_states_num()
            print('There are %s k-step states in total with k = %s.' % (self.total_size, k_step))
        elif mode == 'transition':
            self.total_size = self.par_wrap.graph.get_transition_num()
            print('There are %s transitions in total.' % self.total_size)
        else:
            self.total_size = 0
        self.mode = mode

    def update_coverage(self, outputs):
        seed_num = len(outputs)
        ptrs = np.tile(np.zeros(self.total_size, dtype=np.uint8), (seed_num, 1))

        for i in range(len(ptrs)):
            self.par_wrap.visit_graph(outputs[i], ptrs[i], self.mode)

        return ptrs

    def get_coverage(self, outputs):
        result = []

        for i in range(len(outputs)):
            tmp = []
            self.par_wrap.visit_graph(outputs[i], tmp, self.mode, return_set=True)
            result.append(tmp)

        return result

    def get_coverage_criteria(self, outputs):
        result = set()

        for i in range(len(outputs)):
            tmp = []
            self.par_wrap.visit_graph(outputs[i], tmp, self.mode, return_set=True)
            result = result.union(set(tmp))

        return result

    def get_total(self):
        return self.total_size

    def get_weight_dic(self, reverse=False):
        if reverse:
            return self.par_wrap.graph.get_index_weight_dic(type=self.mode, reverse=True)
        return self.par_wrap.graph.get_index_weight_dic(type=self.mode)


def load_graph_pkl(pkl_dir):
    with open(pkl_dir, 'rb') as f:
        g = pickle.load(f)
    return g


if __name__ == '__main__':
    # print("main Test.")
    # g = load_graph_pkl('wrapper_deepspeech2_3_20.pkl')
    # sys.path.append('/media/lyk/DATA/deepstellar/deepstellar/AbstractModel/GraphWrapper.py')
    import deepstellar.AbstractModel.GraphWrapper
    abst_m = pickle.load(open('/media/lyk/DATA/dlfuzzer_data/ds1_pkls/wrapper_deepspeech2_2_5.pkl', 'rb'))
    print(len(g.graph.states))
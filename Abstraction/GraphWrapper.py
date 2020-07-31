import numpy as np
from Abstraction.DTMCGraph import DTMCGraph
import json


class GraphWrapper:
    def __init__(self, stateAbst, fake_initial=-1):
        self.graph = DTMCGraph(fake_initial)
        self.stateAbst = stateAbst

    def build_model(self, label_dir=None):
        """
        Build model for a specific configuration
        :label_dir: file of the label profiling, currently not used.
        """
        pca_fit = self.stateAbst.get_pca_trans_data()

        if label_dir:
            with open(label_dir) as f:
                translation_all = json.load(f)
        else:
            translation_all = None

        if translation_all:  # if with labels
            for i in range(len(pca_fit)):
                seq = pca_fit[i]
                trans = translation_all[i]
                assert len(seq) == len(trans)
                self.build_step(seq, trans)
        else:  # if without labels
            for i in range(len(pca_fit)):
                seq = pca_fit[i]
                self.build_step(seq, None)
                # break
        # del pca_fit
        # del translation_all
        # self.graph.draw_graph("0", "DTMC")
        # g_warp.graph.transitions = None
        self.extend_to_k_step()  # extend the graph to the steps
        self.graph.init_k_step_idx(self.stateAbst.n_step)
        # g_warp.visit_graph('', [0]*500, 'k-step')
        # g_warp.visit_graph(pca_fit[0], [0]*2000, 'transition')
        # os.makedirs(save2folder, exist_ok=True)

    def build_step(self, seq, labels=None):
        """
        Add a sequence of state vectors to the graph, the vectors are usually transformed by PCA model
        :param seq: the sequence of state vectors
        :param labels: labels for the transitions, currently not used
        """
        transition_seq_name = self.stateAbst.data_transform(seq)  # make abstraction without PCA transformation
        if labels is None:
            labels = ['-']*len(seq)
        self.graph.add_ordered_transitions(transition_seq_name, labels)
        del transition_seq_name

    def extend_to_k_step(self):
        """
        Extend the graph to k step states
        """
        if self.stateAbst.n_step <= 0:
            return
        moves = enumerate_manhattan(self.stateAbst.dimension, self.stateAbst.n_step)
        step_out_dic = {}
        for state_name, _ in self.graph.states.items():
            if state_name != -1:
                decoded_vec = self.stateAbst.coder.decode(state_name)
                for move in moves:
                    step_out = list(np.array(decoded_vec)+np.array(move))
                    step_out = self.stateAbst.coder.encode(step_out)
                    step = abs_sum(move)
                    if step_out in step_out_dic:
                        if step_out_dic[step_out] > step:
                            step_out_dic[step_out] = step
                    else:
                        step_out_dic[step_out] = step
        step_out_seq = []
        step_seq = []
        for step_out, step in step_out_dic.items():
            step_out_seq.append(step_out)
            step_seq.append(step)

        self.graph.add_other_states(step_out_seq, step_seq)

    def visit_graph(self, state_seq, cnt_states, mode, return_set=False):
        """
        Update the coverage for a specific sequence
        :param state_seq: the state vector sequence
        :param cnt_states: current coverage
        :param mode: which coverage criteria
        :param return_set: whether to return the set of covered state/transition id
        :return: the cnt_states will be updated
        """
        transition_seq_name = self.stateAbst.data_transform(state_seq, pca_transform=True)
        if mode == 'state':
            self.graph.to_cover_major_states(transition_seq_name, cnt_states, return_set=return_set)
        elif mode == 'k-step':
            self.graph.to_cover_k_step(transition_seq_name, cnt_states, return_set=return_set)
        elif mode == 'transition':
            self.graph.to_cover_transitions(transition_seq_name, cnt_states, return_set=return_set)


def enumerate_manhattan(dim, k):
    """
    :param dim: dimension of the space
    :param k: max step-out
    :return: the set of all possible moves with in k steps
    """
    vec = [0] * dim
    covered_list = []
    queue = [vec]
    while queue:
        cur_vec = queue.pop(0)
        if cur_vec not in covered_list:
            covered_list.append(cur_vec)
            for i in range(len(cur_vec)):
                tmp = cur_vec.copy()
                tmp[i] += 1
                if abs_sum(tmp) <= k:
                    queue.append(tmp)
                tmp = cur_vec.copy()
                tmp[i] -= 1
                if abs_sum(tmp) <= k:
                    queue.append(tmp)
    covered_list.remove(vec)
    return covered_list


def abs_sum(vec):
    return sum([abs(i) for i in vec])

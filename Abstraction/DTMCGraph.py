from collections import OrderedDict
import math
from graphviz import Digraph
import os


class State(object):
    def __init__(self, name, id):
        self.name = name  # name is the encoded vector
        self.freq = 0
        self.id = id
        self.section = 0  # to which section the state belongs

    def add_freq(self):
        self.freq += 1


class Transition(object):
    def __init__(self, src, dest, label, id):
        self.src = src
        self.dest = dest
        self.label = label  # for input/output label, currently not used
        self.freq = 0
        self.prob = 0.0
        self.id = id

    def add_freq(self):
        self.freq += 1


class DTMCGraph(object):
    def __init__(self, fake_ini):
        self.states = OrderedDict()
        self.other_states = OrderedDict()
        self.transitions = {}
        self.fake_ini = fake_ini
        self.states[fake_ini] = State(fake_ini, 0)
        self.next_transition_id = 0
        self.next_state_id = 1
        self.k_step_idx = {}  # keep a state_name:idx mapping for k-step coverage

    def _add_state(self, state_name):
        """
        add a state to the graph
        :param state_name: the encoded int64
        """
        if state_name not in self.states:
            state = State(state_name, self.next_state_id)
            self.next_state_id += 1
            self.states[state_name] = state
            print('STATE ADDED: %s with id %s' % (state_name, state.id))
        else:
            print('You are trying to add a duplicate state with name %s' % state_name)

    def _add_other_state(self, state_name, section):
        """
        add a k-step state to the graph
        :param state_name: the encoded int64
        :param section: 1 for 1 step, 2 for step and so on
        """
        if state_name not in self.other_states:
            state = State(state_name, self.next_state_id)
            state.section = section
            self.next_state_id += 1
            self.other_states[state_name] = state
            print('OTHER STATE ADDED: %s with id %s' % (state_name, state.id))
        else:
            print('You are tring to add a duplicate state with name %s' % state_name)

    def _add_transition(self, src, dst):
        """
        add a transition
        :param src: name of the source state
        :param dst: name of the destination state
        """
        if src not in self.states:
            print('ERROR: src state can not be found in the graph.')
            return -1
        if dst not in self.states:
            self._add_state(dst)

        self.states[dst].add_freq()
        src = self.states[src].id
        dst = self.states[dst].id

        if src in self.transitions:
            if dst in self.transitions[src]:
                self.transitions[src][dst].add_freq()
            else:
                trans = Transition(src, dst, '', self.next_transition_id)
                print('TRANSITION ADDED: with id %s' % self.next_transition_id)
                self.next_transition_id += 1
                trans.add_freq()
                self.transitions[src][dst] = trans
        else:
            self.transitions[src] = {}
            trans = Transition(src, dst, '', self.next_transition_id)
            self.next_transition_id += 1
            trans.add_freq()
            self.transitions[src][dst] = trans

    def add_ordered_transitions(self, trans_seq, label_seq):
        """
        add a set of transitions with a sequence of states
        :param trans_seq: sequence of states specifying the transitions
        :param output_seq: transition label, but currently not used
        """
        trans_seq = [self.fake_ini] + trans_seq
        for i in range(len(trans_seq)-1):
            src = trans_seq[i]
            dest = trans_seq[i+1]
            self._add_transition(src, dest)

    def add_other_states(self, state_seq, section_seq):
        """
        add a set of other states
        :param state_seq: a list of states
        :param section_seq: a list of corresponding sections
        """
        for i in range(len(state_seq)):
            if state_seq[i] not in self.states:
                self._add_other_state(state_seq[i], section_seq[i])

    def cal_trans_prob(self):
        for _, state in self.states.items():
            if state.id in self.transitions:
                out_trans = self.transitions[state.id]
                total = sum([tr.freq for _, tr in out_trans.items()])
                for _, tr in out_trans.items():
                    tr.prob = tr.freq/total

    def draw_graph(self, folder, type):
        self.cal_trans_prob()
        dot = Digraph(comment='RNN state transition graph')
        for state in self.states.values():
            dot.node(str(state.id), '%s' % state.id)
        for src, dlist in self.transitions.items():
            for dest, transition in dlist.items():
                lab = '%.2f' % transition.prob
                dot.edge(str(src), str(dest), label=lab)
        dot.render(os.path.join(folder, '%s.gv' % type), view=False)
        # print(dot.source)

    def to_cover_major_states(self, transition_seq_name, cnt_states, return_set=False):
        """
        update the cnt_states with coverage triggered by the sequence of transitions
        :param transition_seq_name: a name sequence of states
        :param cnt_states: a coverage vector with same length of the self.states, it is
        indexed by the state.id
        :param  return_set: whether to return the set of ids of covered states/transitions
        :return: the cnt_states is updated
        """
        for i in range(len(transition_seq_name)-1):
            dst = transition_seq_name[i+1]
            if dst in self.states:
                idx = self.states[dst].id
                if not return_set:
                    num = cnt_states[idx]
                    if num < 255:
                        num += 1
                        cnt_states[idx] = num
                else:
                    cnt_states.append(idx)

    def to_cover_k_step(self, transition_seq_name, cnt_states, return_set=False):
        for i in range(len(transition_seq_name)-1):
            dst = transition_seq_name[i+1]
            if dst in self.k_step_idx:
                idx = self.k_step_idx[dst]
                if not return_set:
                    num = cnt_states[idx]
                    if num < 255:
                        num += 1
                        cnt_states[idx] = num
                else:
                    cnt_states.append(idx)

    def init_k_step_idx(self, k):
        """
        initialize a mapping between k-step state name and vector index
        :param k: maximum step to calculate the coverage, i.e., only consider states within k step
        """
        if k <= 0:
            return
        self.k_step_idx = {}
        i = 0
        for state_name, state in self.other_states.items():
            if state.section <= k:
                self.k_step_idx[state_name] = i
                i += 1

    def to_cover_transitions(self, transition_seq_name, cnt_states, return_set=False):
        for i in range(len(transition_seq_name)-1):
            src = transition_seq_name[i]
            dst = transition_seq_name[i+1]
            if src not in self.states or dst not in self.states:
                continue
            src = self.states[src].id
            dst = self.states[dst].id
            # tran = trans[i]
            if src in self.transitions:
                if dst in self.transitions[src]:
                    idx = self.transitions[src][dst].id
                    if not return_set:
                        num = cnt_states[idx]
                        if num < 255:
                            num += 1
                            cnt_states[idx] = num
                    else:
                        cnt_states.append(idx)

    def get_index_weight_dic(self, type="state", reverse=False):
        cri_dic = self.states
        if type == "transition":
            cri_dic = dict()
            for src, entry in self.transitions.items():
                for dst, tran in entry.items():
                    cri_dic[tran.id] = tran

        total = 0
        for name, entry in cri_dic.items():
            total += entry.freq
        # print(self.next_transition_id)
        result = dict()
        for name, entry in cri_dic.items():
            result[entry.id] = entry.freq/total

        if not reverse:
            return result
        else:
            rev_dic = dict()
            for k, w in result.items():
                rev_dic[k] = (1-w)/(self.next_transition_id-1)
            return rev_dic

    def get_major_states_num(self):
        return len(self.states)

    def get_transition_num(self):
        return self.next_transition_id

    def get_k_step_states_num(self):
        return len(self.k_step_idx)
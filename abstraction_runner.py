import argparse
import os
import pickle
from mnist_demo.mnist_lstm import MnistLSTMClassifier
from Abstraction.StateAbstraction import StateAbstraction
from Abstraction.GraphWrapper import GraphWrapper


if __name__ == '__main__':
    parse = argparse.ArgumentParser("Generate abstract model")
    parse.add_argument('-dl_model', help='path of dl model', required=True)
    # parse.add_argument('-profile_data', help="path of data to do the profiling")
    parse.add_argument('-profile_save_path', help="dir to save profiling raw data", required=True)
    parse.add_argument('-comp_num', help="number of component when fitting pca", type=int, required=True)  # can select a larger number
    parse.add_argument('-k', help='number of dimension to keep', type=int, required=True)
    parse.add_argument('-m', help="number of intervals on each dimension", type=int, required=True)
    parse.add_argument('-bits', help="number of bits for encoding", type=int, required=True)
    parse.add_argument('-name_prefix', help="name prefix when save the abstract model", required=True)
    parse.add_argument('-abst_save_path', help="path to save abstract model", required=True)
    parse.add_argument('-n_step', help="extend the graph to n_step", type=int, default=0)

    args = parse.parse_args()

    lstm_classifier = MnistLSTMClassifier()
    lstm_classifier.load_hidden_state_model(args.dl_model)

    if not os.path.exists(args.profile_save_path):
        lstm_classifier.profile_train_data(args.profile_save_path)
        print("profiling done...")
    else:
        print("profiling is already done...")

    par_k = [args.m]*args.k
    stateAbst = StateAbstraction(args.profile_save_path, args.comp_num, args.bits, [args.m]*args.k, args.n_step)
    wrapper = GraphWrapper(stateAbst)
    wrapper.build_model()

    save_file = 'wrapper_%s_%s_%s.pkl' % (args.name_prefix, len(par_k), par_k[0])
    save_file = os.path.join(args.abst_save_path, save_file)
    os.makedirs(args.abst_save_path, exist_ok=True)
    with open(save_file, 'wb') as f:
        pickle.dump(wrapper, f)

    print('finish')


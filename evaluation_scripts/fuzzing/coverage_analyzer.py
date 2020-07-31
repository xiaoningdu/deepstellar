import os
import argparse
from coverage import Coverage
import numpy as np
from mnist_demo.mnist_lstm import MnistLSTMClassifier


def read_inputs_from_folder(folder, type="queue"):
    files = os.listdir(folder)
    tests = []
    for file in files:
        data = np.load(os.path.join(folder, file))
        if type == "crash":
            x_test = np.expand_dims(data, 0)
        elif type == "queue":
            x_test = data[1:2]
        else:
            x_test = data
        tests.extend(x_test)

    return np.asarray(tests)


def fuzzing_analyzer(classifier, folder, dtmc_wrapper_f, type):
    if type == "queue":
        inputs = read_inputs_from_folder(folder, type="queue")
    else:  # type == "seeds"
        inputs = read_inputs_from_folder(folder, type="seed")

    states = classifier.get_state_profile(inputs)
    coverage_handlers = []

    for criteria, k_step in [("state", 0), ("transition", 0)]:  # , ("k-step", 3), ("k-step", 6)
        cov = Coverage(dtmc_wrapper_f, criteria, k_step)
        coverage_handlers.append(cov)

    for coverage_handler in coverage_handlers:
        cov = coverage_handler.get_coverage_criteria(states)
        total = coverage_handler.get_total()
        print(len(cov) / total)
        if coverage_handler.mode != "k-step":  # to printout the weighted coverage metrics
            weight_dic = coverage_handler.get_weight_dic()
            print(sum([weight_dic[e] for e in cov]))
            rev_weight_dic = coverage_handler.get_weight_dic(reverse=True)
            print(sum([rev_weight_dic[e] for e in cov]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='analyzing the fuzzing results')
    parser.add_argument('-dl_model', help='path to the dl model', required=True)
    parser.add_argument('-wrapper', help='path to the abstract graph wrapper', required=True)
    parser.add_argument('-inputs_folder', help='path to the inputs folder', required=True)
    parser.add_argument('-type', choices=['seeds', 'queue'], default='queue')
    args = parser.parse_args()

    classifier = MnistLSTMClassifier()
    classifier.load_hidden_state_model(args.dl_model)
    fuzzing_analyzer(classifier, args.inputs_folder, args.wrapper, args.type)


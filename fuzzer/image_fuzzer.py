import argparse, pickle
import shutil

import tensorflow as tf
import os
from coverage import Coverage
from keras.applications.vgg16 import preprocess_input
import random
import time
import numpy as np
from fuzzer.image_queue import ImageInputCorpus
from fuzzer.fuzzone import build_fetch_function

from fuzzer.lib.fuzzer import Fuzzer
from fuzzer.mutators import Mutators
from fuzzer.image_queue import Seed
from mnist_demo.mnist_lstm import MnistLSTMClassifier


def imagenet_preprocessing(input_img_data):
    temp = np.copy(input_img_data)
    temp = np.float32(temp)
    qq = preprocess_input(temp)  # final input shape = (1,224,224,3)
    return qq


def mnist_preprocessing(x_test):
    temp = np.copy(x_test)
    temp = temp.reshape(temp.shape[0], 28, 28)
    temp = temp.astype('float32')
    temp /= 255
    return temp


def cifar_preprocessing(x_test):
    temp = np.copy(x_test)
    temp = temp.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        temp[:, :, :, i] = (temp[:, :, :, i] - mean[i]) / std[i]
    return temp


preprocess_dic = {
    'cifar10': cifar_preprocessing,
    'mnist': mnist_preprocessing,
    'imagenet': imagenet_preprocessing
}

shape_dic = {
    'cifar10': (32, 32, 3),
    'mnist': (28, 28),
    'imagenet': (224, 224, 3)
}

execlude_layer_dic = {
    'vgg16': ['input', 'flatten', 'activation', 'batch', 'dropout'],
    'resnet20': ['input', 'flatten', 'activation', 'batch', 'dropout'],
    'lenet1': ['input', 'flatten', 'activation', 'batch', 'dropout'],
    'lenet4': ['input', 'flatten', 'activation', 'batch', 'dropout'],
    'lenet5': ['input', 'flatten', 'activation', 'batch', 'dropout'],
    'mobilenet': ['input', 'flatten', 'padding', 'activation', 'batch', 'dropout',
                  'bn', 'reshape', 'relu', 'pool', 'concat', 'softmax', 'fc'],
    'vgg19': ['input', 'flatten', 'padding', 'activation', 'batch', 'dropout', 'bn',
              'reshape', 'relu', 'pool', 'concat', 'softmax', 'fc'],
    'resnet50': ['input', 'flatten', 'padding', 'activation', 'batch', 'dropout', 'bn',
                 'reshape', 'relu', 'pool', 'concat', 'add', 'res4', 'res5']
}


def metadata_function(meta_batches):
    return meta_batches


def image_mutation_function(batch_num, deeptest=False):
    def func(seed):
        if deeptest:
            return Mutators.image_random_mutate(seed, batch_num)
        else:
            return Mutators.image_random_mutate(seed, batch_num)

    return func


def objective_function(seed):
    """Checks if the metadata is inf or NaN."""
    metadata = seed.metadata
    ground_truth = seed.ground_truth
    return metadata[0] != ground_truth


def iterate_function():
    def func(queue, root_seed, parent, mutated_coverage_list, mutated_data_batches, mutated_metadata_list,
             objective_function):
        ori_batches, batches, cl_batches = mutated_data_batches
        successed = False
        bug_found = False
        for idx in range(len(mutated_coverage_list)):
            # 1000 for placeholder
            input = Seed(cl_batches[idx], 1000, mutated_coverage_list[idx], root_seed, parent,
                         mutated_metadata_list[:, idx],
                         parent.ground_truth)
            is_adv = objective_function(input)
            if is_adv:
                suf = 'g_' + str(input.ground_truth) + 'c_' + str(input.metadata[0]) + '-' + root_seed
                queue.save_if_interesting(input, batches[idx], True, suffix=suf)
            else:
                new_img = np.append(ori_batches[idx:idx + 1], batches[idx:idx + 1], axis=0)
                successed = queue.save_if_interesting(input, new_img, False) or successed
        return bug_found, successed

    return func


def dry_run(indir, fetch_function, coverage_function, queue):
    seed_lis = os.listdir(indir)
    if len(seed_lis) == 0:
        print('Empty dir')
        exit(0)
    for seed_name in seed_lis:
        tf.logging.info("Attempting dry run with '%s'...", seed_name)
        path = os.path.join(indir, seed_name)
        img = np.load(path)
        # input_batches = img
        coverage_batches, metadata_batches = fetch_function((0, img, 0))
        coverage_list = coverage_function(coverage_batches)
        metadata_list = metadata_function(metadata_batches)
        input = Seed(0, 1000, coverage_list[0], seed_name, None, metadata_list[0][0], metadata_list[0][0])
        new_img = np.append(img, img, axis=0)
        queue.save_if_interesting(input, new_img, False, True, seed_name)


if __name__ == '__main__':

    start_time = time.time()
    # Log more
    tf.logging.set_verbosity(tf.logging.INFO)
    random.seed(time.time())

    parser = argparse.ArgumentParser(description='coverage guided fuzzing')

    parser.add_argument('-i', help='input seed dir')
    parser.add_argument('-o', help='seed output')

    parser.add_argument('-model_type', help="target model fuzz", choices=['mnist', 'cifar10', 'imagenet'])
    parser.add_argument('-dl_model', help="path to the dl model", required=True)
    parser.add_argument('-criteria', help="set the criteria to guide",
                        choices=['state', 'k-step', 'transition'], default='state')
    parser.add_argument('-k_step', help="how many outer step to check", type=int, default=0)
    parser.add_argument('-batch_num', help="set mutation batch number", type=int, default=20)
    parser.add_argument('-iterations', help="total regression tests tried", type=int, default=10000000)
    parser.add_argument('-cri_parameter', help="set the parameter of criteria", type=float)
    parser.add_argument('-quantize', help="fuzzer for quantize", default=0, type=int)
    parser.add_argument('-quantize_models', help="fuzzer for quantize")
    parser.add_argument('-random', help="set mutation batch number", type=int, default=0)
    parser.add_argument('-select', help="select next",
                        choices=['random2', 'random', 'tensorfuzz', 'deeptest', 'deeptest2', 'prob'], default='prob')
    parser.add_argument('-pkl_path', help='pkl path')

    args = parser.parse_args()

    if os.path.exists(args.o):
        shutil.rmtree(args.o)
    os.makedirs(os.path.join(args.o, 'queue'))
    os.makedirs(os.path.join(args.o, 'crashes'))

    lstm_classifier = MnistLSTMClassifier()
    lstm_classifier.load_hidden_state_model(args.dl_model)
    model = lstm_classifier.model
    preprocess = preprocess_dic[args.model_type]

    coverage_handler = Coverage(args.pkl_path, args.criteria, args.k_step)

    plot_file = open(os.path.join(args.o, 'plot.log'), 'a+')

    fetch_function_1 = build_fetch_function(model, preprocess)

    dry_run_fetch = build_fetch_function(model, preprocess)

    coverage_function = coverage_handler.update_coverage

    mutation_function = image_mutation_function(args.batch_num)

    queue = ImageInputCorpus(args.o, args.random, args.select, coverage_handler.total_size, args.criteria)

    dry_run(args.i, dry_run_fetch, coverage_function, queue)

    image_iterate_function = iterate_function()

    fuzzer = Fuzzer(queue, coverage_function, metadata_function, objective_function, mutation_function,
                    fetch_function_1, image_iterate_function, args.select)

    fuzzer.loop(args.iterations)
    # queue.log()

    print('finish', time.time() - start_time)

import argparse
import pickle
import os
import sys
import random
from keras.datasets import mnist

from keras.models import load_model
import numpy as np
from keras.datasets import cifar10

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

model_weight_path = {
    'mnist': "test/rnn_model/model.h5"
}
def createBatch(x_batch, batch_size, output_path, prefix):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    batch_num = len(x_batch) / batch_size
    batches = np.split(x_batch, batch_num, axis=0)
    for i, batch in enumerate(batches):
        test = batch
        saved_name = prefix + str(i) + '.npy'
        np.save(os.path.join(output_path, saved_name), test)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='control experiment')


    parser.add_argument('-model_type', help='Model type', choices=['mnist', 'cifar10', 'imagenet'], default='mnist')
    parser.add_argument('-output_path', help='Out path',default='/home/xiaoning/Desktop/ml_demo/fuzz_data/initialseeds')
    parser.add_argument('-batch_size', type=int, help='Number of images in one batch', default=10)
    parser.add_argument('-batch_num', type=int, help='Number of batches', default=10)
    args = parser.parse_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    if args.model_type == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        batch = mnist_preprocessing(x_test)
        model = load_model(model_weight_path['mnist'])
        x_test = x_test.reshape(x_test.shape[0], 28, 28)
    elif args.model_type == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        batch = cifar_preprocessing(x_test)
        model = load_model(model_weight_path['cifar10'])
    else:
        assert (False)

    num_in_each_class = (args.batch_size * args.batch_num) / 10

    result = np.argmax(model.predict(batch), axis=1) # [0],axis=1


    new_label = np.reshape(y_test, result.shape)

    idx_good = np.where(new_label == result)[0]


    for cl in range(10):
        cl_indexes  = [i for i in idx_good if new_label[i] == cl]
        selected = random.sample(cl_indexes, int(num_in_each_class))
        createBatch(x_test[selected], args.batch_size, args.output_path, str(cl)+'_')
    print('finish')
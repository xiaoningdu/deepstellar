import argparse
import os
import random
from keras.datasets import mnist

from keras.models import load_model
import numpy as np


def mnist_preprocessing(x_test):
    temp = np.copy(x_test)
    temp = temp.reshape(temp.shape[0], 28, 28)
    temp = temp.astype('float32')
    temp /= 255
    return temp


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

    parser.add_argument('-dl_model', help='path to model')
    parser.add_argument('-output_path', help='Out path')
    parser.add_argument('-batch_size', type=int, help='Number of images in one batch', default=1)
    parser.add_argument('-batch_num', type=int, help='Number of batches', default=100)
    args = parser.parse_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    batch = mnist_preprocessing(x_test)
    model = load_model(args.dl_model)
    x_test = x_test.reshape(x_test.shape[0], 28, 28)

    num_in_each_class = (args.batch_size * args.batch_num) / 10

    result = np.argmax(model.predict(batch), axis=1)  # [0],axis=1

    new_label = np.reshape(y_test, result.shape)

    idx_good = np.where(new_label == result)[0]

    for cl in range(10):
        cl_indexes = [i for i in idx_good if new_label[i] == cl]
        selected = random.sample(cl_indexes, int(num_in_each_class))
        createBatch(x_test[selected], args.batch_size, args.output_path, str(cl) + '_')
    print('finish')

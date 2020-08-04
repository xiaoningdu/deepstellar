from __future__ import print_function
import cv2
import numpy as np
import random
import time
import copy


class Mutators():
    def image_translation(img, params):

        rows, cols, ch = img.shape
        # rows, cols = img.shape

        # M = np.float32([[1, 0, params[0]], [0, 1, params[1]]])
        M = np.float32([[1, 0, params], [0, 1, params]])
        dst = cv2.warpAffine(img, M, (cols, rows))
        return dst

    def image_scale(img, params):

        # res = cv2.resize(img, None, fx=params[0], fy=params[1], interpolation=cv2.INTER_CUBIC)
        rows, cols, ch = img.shape
        res = cv2.resize(img, None, fx=params, fy=params, interpolation=cv2.INTER_CUBIC)
        res = res.reshape((res.shape[0], res.shape[1], ch))
        y, x, z = res.shape
        if params > 1:  # need to crop
            startx = x // 2 - cols // 2
            starty = y // 2 - rows // 2
            return res[starty:starty + rows, startx:startx + cols]
        elif params < 1:  # need to pad
            sty = int((rows - y) / 2)
            stx = int((cols - x) / 2)
            return np.pad(res, [(sty, rows - y - sty), (stx, cols - x - stx), (0, 0)], mode='constant',
                          constant_values=0)
        return res

    def image_shear(img, params):
        rows, cols, ch = img.shape
        # rows, cols = img.shape
        factor = params * (-1.0)
        M = np.float32([[1, factor, 0], [0, 1, 0]])
        dst = cv2.warpAffine(img, M, (cols, rows))
        return dst

    def image_rotation(img, params):
        rows, cols, ch = img.shape
        # rows, cols = img.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), params, 1)
        dst = cv2.warpAffine(img, M, (cols, rows), flags=cv2.INTER_AREA)
        return dst

    def image_contrast(img, params):
        alpha = params
        new_img = cv2.multiply(img, np.array([alpha]))  # mul_img = img*alpha
        # new_img = cv2.add(mul_img, beta)                                  # new_img = img*alpha + beta

        return new_img

    def image_brightness(img, params):
        beta = params
        new_img = cv2.add(img, beta)  # new_img = img*alpha + beta
        return new_img

    def image_blur(img, params):

        # print("blur")
        blur = []
        if params == 1:
            blur = cv2.blur(img, (3, 3))
        if params == 2:
            blur = cv2.blur(img, (4, 4))
        if params == 3:
            blur = cv2.blur(img, (5, 5))
        if params == 4:
            blur = cv2.GaussianBlur(img, (3, 3), 0)
        if params == 5:
            blur = cv2.GaussianBlur(img, (5, 5), 0)
        if params == 6:
            blur = cv2.GaussianBlur(img, (7, 7), 0)
        if params == 7:
            blur = cv2.medianBlur(img, 3)
        if params == 8:
            blur = cv2.medianBlur(img, 5)
        # if params == 9:
        #     blur = cv2.blur(img, (6, 6))
        if params == 9:
            blur = cv2.bilateralFilter(img, 6, 50, 50)
            # blur = cv2.bilateralFilter(img, 9, 75, 75)
        return blur

    def image_pixel_change(img, params):
        # random change 1 - 5 pixels from 0 -255
        img_shape = img.shape
        img1d = np.ravel(img)
        arr = np.random.randint(0, len(img1d), params)
        for i in arr:
            img1d[i] = np.random.randint(0, 256)
        new_img = img1d.reshape(img_shape)
        return new_img

    def image_noise(img, params):
        if params == 1:  # Gaussian-distributed additive noise.
            row, col, ch = img.shape
            mean = 0
            var = 0.1
            sigma = var ** 0.5
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
            noisy = img + gauss
            return noisy.astype(np.uint8)
        elif params == 2:  # Replaces random pixels with 0 or 1.
            s_vs_p = 0.5
            amount = 0.004
            out = np.copy(img)
            # Salt mode
            num_salt = np.ceil(amount * img.size * s_vs_p)
            coords = [np.random.randint(0, i, int(num_salt))
                      for i in img.shape]
            out[tuple(coords)] = 1

            # Pepper mode
            num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i, int(num_pepper))
                      for i in img.shape]
            out[tuple(coords)] = 0
            return out
        elif params == 3:  # Multiplicative noise using out = image + n*image,where n is uniform noise with specified mean & variance.
            row, col, ch = img.shape
            gauss = np.random.randn(row, col, ch)
            gauss = gauss.reshape(row, col, ch)
            noisy = img + img * gauss
            return noisy.astype(np.uint8)

    '''    
    TODO: Add more mutators, current version is from DeepTest, https://arxiv.org/pdf/1708.08559.pdf

    Also check,   https://arxiv.org/pdf/1712.01785.pdf, and DeepExplore

    '''

    # TODO: Random L 0

    # TODO: Random L infinity

    # more transformations refer to: http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html#geometric-transformations
    # http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_table_of_contents_imgproc/py_table_of_contents_imgproc.html

    transformations = [image_translation, image_scale, image_shear, image_rotation,
                       image_contrast, image_brightness, image_blur, image_pixel_change, image_noise]

    # these parameters need to be carefullly considered in the experiment
    # to consider the feedbacks
    params = []
    params.append(list(range(-3, 3)))  # image_translation
    params.append(list(map(lambda x: x * 0.1, list(range(8, 11)))))  # image_scale
    params.append(list(map(lambda x: x * 0.1, list(range(-5, 5)))))  # image_shear
    params.append(list(range(-30, 30)))  # image_rotation
    params.append(list(map(lambda x: x * 0.1, list(range(6, 12)))))  # image_contrast
    params.append(list(range(-20, 20)))  # image_brightness
    params.append(list(range(1, 10)))  # image_blur
    params.append(list(range(1, 10)))  # image_pixel_change
    params.append(list(range(1, 4)))  # image_noise

    classA = [7, 8]
    classB = [0, 1, 2, 3, 4, 5, 6]

    # classB = [5, 6]
    # classB = []
    @staticmethod
    def mutate_one(ori_img, img, cl, try_num=50):
        x, y, z = img.shape

        a = 0.02
        b = 0.30
        l0 = int(a * x * y * z)
        l_infinity = int(b * 255)
        ori_shape = ori_img.shape
        for ii in range(try_num):
            random.seed(time.time())
            if cl == 0:  # 0 can choose class A and B
                tid = random.sample(Mutators.classA + Mutators.classB, 1)[0]
                transformation = Mutators.transformations[tid]

                params = Mutators.params[tid]
                param = random.sample(params, 1)[0]
                img_new = transformation(copy.deepcopy(img), param)
                img_new = img_new.reshape(ori_shape)

                if tid in Mutators.classA:
                    sub = ori_img - img_new
                    if np.sum(sub != 0) < l0 or np.max(abs(sub)) < l_infinity:
                        return ori_img, img_new, 0, 1
                else:  # B, C
                    # print(transformation)
                    ori_img = transformation(copy.deepcopy(ori_img), param)  # original image need to be updated
                    # print('Original changed with %s',transformation)
                    ori_img = ori_img.reshape(ori_shape)
                    return ori_img, img_new, 1, 1
            if cl == 1:
                tid = random.sample(Mutators.classA, 1)[0]
                transformation = Mutators.transformations[tid]
                params = Mutators.params[tid]
                param = random.sample(params, 1)[0]
                img_new = transformation(copy.deepcopy(img), param)
                sub = ori_img - img_new
                if np.sum(sub != 0) < l0 or np.max(abs(sub)) < l_infinity:
                    return ori_img, img_new, 1, 1
        return ori_img, img, cl, 0

    @staticmethod
    def image_random_mutate(seed, batch_num):
        '''
        This is the interface to perform random mutation on input image, random select
        an mutator and perform a random mutation with a random parameter predefined.

        :param img: input image cl: class
        :param params:
        :return:
        '''

        # randomly sample
        # tid = random.sample([0, 1, 2, 3, 4, 5, 6], 1)[0]
        # l0 = 300
        # l_infinity = 150

        test = np.load(seed.fname)
        test = np.expand_dims(test, axis=-1)
        ori_img = test[0]
        img = test[1]
        cl = seed.clss
        ori_batches = []
        batches = []
        cl_batches = []
        for i in range(batch_num):
            ori_out, img_out, cl_out, changed = Mutators.mutate_one(ori_img, img, cl)
            if changed:
                ori_batches.append(ori_out)
                batches.append(img_out)
                cl_batches.append(cl_out)
        # ori_batches = np.squeeze(ori_batches)
        # batches = np.squeeze(batches)
        if len(ori_batches) > 0:
            ori_batches = np.squeeze(np.asarray(ori_batches), axis=-1)
            batches = np.squeeze(np.asarray(batches), axis=-1)
        return (ori_batches, batches, cl_batches)

    @staticmethod
    def mutate_two(seed, batch_num):
        '''
        This is the interface to perform random mutation on input image, random select
        an mutator and perform a random mutation with a random parameter predefined.

        :param img: input image cl: class
        :param params:
        :return:
        '''

        # randomly sample
        # tid = random.sample([0, 1, 2, 3, 4, 5, 6], 1)[0]
        # l0 = 300
        # l_infinity = 150

        test = np.load(seed.fname)
        ori_img = test[0]
        img = test[1]
        cl = seed.clss
        ori_batches = []
        batches = []
        cl_batches = []
        for i in range(batch_num):
            ori_out, img_out, cl_out, changed = Mutators.mutate_one(ori_img, img, cl)
            # ori_out, img_out, cl_out, changed = Mutators.mutate_one(ori_out, img_out, cl_out)
            if changed:
                ori_batches.append(ori_out)
                batches.append(img_out)
                cl_batches.append(cl_out)

        return (np.asarray(ori_batches), np.asarray(batches), cl_batches)

import json
import math
import numpy as np
import os
import sys
import random
from tqdm import trange
import math

NUM_USER = 30


def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum(np.exp(x))
    return ex / sum_ex


def generate_synthetic(alpha, beta, iid, sample_per_user = 50):
    dimension = 60
    NUM_CLASS = 10

    # samples_per_user = np.random.lognormal(4, 1.5, (NUM_USER)).astype(int) + sample_per_user
    # print(samples_per_user)
    # print('mean:', np.mean(samples_per_user))

    samples_per_user = [sample_per_user for _ in range(NUM_USER)]

    X_split = [[] for _ in range(NUM_USER)]
    y_split = [[] for _ in range(NUM_USER)]

    #### define some eprior ####
    mean_W = np.random.normal(0, alpha, NUM_USER)
    mean_b = mean_W
    B = np.random.normal(0, beta, NUM_USER)
    mean_x = np.zeros((NUM_USER, dimension))

    diagonal = np.zeros(dimension)
    for j in range(dimension):
        diagonal[j] = np.power((j + 1), -1.2)
    cov_x = np.diag(diagonal)

    for i in range(NUM_USER):
        if iid == 1:
            mean_x[i] = np.ones(dimension) * B[i]  # all zeros
        else:
            mean_x[i] = np.random.normal(B[i], 1, dimension)
        print(mean_x[i])

    if iid == 1:
        W_global = np.random.normal(0, 1, (dimension, NUM_CLASS))
        b_global = np.random.normal(0, 1, NUM_CLASS)

    for i in range(NUM_USER):

        W = np.random.normal(mean_W[i], 1, (dimension, NUM_CLASS))
        b = np.random.normal(mean_b[i], 1, NUM_CLASS)

        if iid == 1:
            W = W_global
            b = b_global

        xx = np.random.multivariate_normal(mean_x[i], cov_x, samples_per_user[i])
        yy = np.zeros(samples_per_user[i])

        for j in range(samples_per_user[i]):
            tmp = np.dot(xx[j], W) + b
            yy[j] = np.argmax(softmax(tmp))

        X_split[i] = xx.tolist()
        y_split[i] = yy.tolist()

        print("{}-th users has {} exampls".format(i, len(y_split[i])))

    return X_split, y_split


def main():

    # train_path = "data/train/mytrain.json"
    # test_path = "data/test/mytest.json"

    alpha = 0.
    beta = 0.
    iid = 0.


    # original_l = [60, 100, 194, 116, 90, 100, 57, 49, 60, 762, 515, 808, 126, 47, 224, 40, 65, 58, 358, 43, 192, 141, 45, 49, 66, 42, 92, 100, 426, 109]
    # print('mean ori: ', np.mean(original_l))

    sample_per_user = 100

    # s = np.random.lognormal(3., 1., 1000)
    # print(s)

    X, y = generate_synthetic(alpha=alpha, beta=beta, iid=iid, sample_per_user = sample_per_user)  # synthetic (0,0)

    train_path = f"train/train_{alpha}_{beta}_{iid}_{sample_per_user}.json"
    test_path = f"test/test_{alpha}_{beta}_{iid}_{sample_per_user}.json"

    # Create data structure
    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}

    for i in trange(NUM_USER, ncols=120):
        uname = 'f_{0:05d}'.format(i)
        combined = list(zip(X[i], y[i]))
        random.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)
        num_samples = len(X[i])
        train_len = int(0.8 * num_samples)
        test_len = num_samples - train_len

        train_data['users'].append(uname)
        train_data['user_data'][uname] = {'x': X[i][:train_len], 'y': y[i][:train_len]}
        train_data['num_samples'].append(train_len)
        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x': X[i][train_len:], 'y': y[i][train_len:]}
        test_data['num_samples'].append(test_len)

    with open(train_path, 'w') as outfile:
        json.dump(train_data, outfile)
    with open(test_path, 'w') as outfile:
        json.dump(test_data, outfile)


if __name__ == "__main__":
    main()
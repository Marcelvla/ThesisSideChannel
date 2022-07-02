## This file contains all different methods for feature extration
## for a multivariate template attack
# Imports
import scipy.io, scipy.stats
import numpy as np
import sys
import pdb
from collections import Counter
import struct
import pandas as pd
import pdb
from sklearn.decomposition import NMF, FactorAnalysis
import time

def load_train(key, split):
    ''' Open the training part of the dataset
    '''
    key_data = []
    with open("Data/Group_" + str(key) + "_Grizzly.raw", "rb") as f:
        for _ in range(split):
            trace = np.array(struct.unpack_from(">2500H",f.read(5000)))
            key_data.append(trace)
        f.close()

    return np.array(key_data)

def load_test(key, split, U_red):
    ''' Open the test part of the dataset and use U_red to create the reduced
        set
    '''
    key_data = []
    with open("Data/Group_" + str(key) + "_Grizzly.raw", "rb") as f:
        for _ in range(split):
            # trace = np.array(struct.unpack_from(">2500H",f.read(5000)))
            f.read(5000)

        for _ in range(split, 3072):
            trace = np.array(struct.unpack_from(">2500H",f.read(5000)))
            key_data.append(trace)

        f.close()

    test_red = np.dot(np.array(key_data), U_red)

    return test_red

def load_test_mod(key, split, model):
    ''' Open the test part of the dataset and use U_red to create the reduced
        set
    '''
    key_data = []
    with open("Data/Group_" + str(key) + "_Grizzly.raw", "rb") as f:
        for _ in range(split):
            # trace = np.array(struct.unpack_from(">2500H",f.read(5000)))
            f.read(5000)

        for _ in range(split, 3072):
            trace = np.array(struct.unpack_from(">2500H",f.read(5000)))
            key_data.append(trace)

        f.close()

    test_red = model.transform(key_data)

    return test_red

def trainMeans(split, keys):
    ''' Incrementally calculate means for training data
    '''
    t_means = []
    if split == 2072:
        data = pd.read_csv('Data/means2072.csv').T
        for k in range(keys):
            print(f"Loading key {k}", end ="\r")
            t_means.append(list(data[k][1:]))
    else:
        for k in range(keys):
            print(f"Loading key {k}", end ="\r")
            data = load_train(k, split)
            t_means.append(np.mean(data, axis=0))

    # pdb.set_trace()

    tbar = np.mean(t_means, axis=0)

    return t_means, tbar

def PCA(n_features, keys, split=2072):
    ''' Perfom PCA and return projected matrix
    '''
    # train_means = [np.mean(t, axis=0) for t in train]
    # tbar = np.mean(train_means, axis=0)
    train_means, tbar = trainMeans(split, keys)
    print("Calculated means")

    B_temp = 0
    for key in range(keys):
        B_temp += np.outer(train_means[key] - tbar, (train_means[key] - tbar).T)

    B = B_temp/keys
    u, s, vh = np.linalg.svd(B)
    # print(f"shape of u {u.shape}")
    U_red = u[0:n_features].T

    return U_red

def FE_NMF(n_features, keys, split=2072):
    ''' Perform NMF for every key
    '''
    Hlist = []
    for key in range(keys):
        print(f"Loading and fitting data for key {key}", end='\r')
        data = load_train(key, split)
        model = NMF(n_components = n_features)
        # print("Fitted")
        W = model.fit_transform(data)
        H = model.components_
        Hlist.append(H)

    # pdb.set_trace()
    # U, S, V = np.linalg.svd(np.mean(Hlist, axis=0))
    # pdb.set_trace()
    # s_diag = np.diag(S)
    U_red = np.linalg.pinv(np.mean(Hlist, axis=0))
    # np.dot(np.dot(V, np.linalg.inv(s_diag)), U.T)

    return U_red

def FE_NMF_2(n_features, keys, split=2072):
    ''' Perform NMF for every key average all training
    '''
    data = np.array(split * [np.zeros(2500)])
    for key in range(keys):
        print(f"Loading and fitting data for key {key}", end='\r')
        data += load_train(key, split)

    data = data / keys

    model = NMF(n_components = n_features)
    W = model.fit_transform(data)
    H = model.components_
    U_red = np.linalg.pinv(H)

    return U_red

def FE_FA(n_features, keys, split=2072):
    ''' Dictionary learning
    '''
    data = np.array(split * [np.zeros(2500)])
    for key in range(keys):
        print(f"Loading and fitting data for key {key}", end='\r')
        data += load_train(key, split)

    data = data / keys

    model = FactorAnalysis(n_components = n_features)
    model_fit = model.fit(data)

    return model_fit

def dataTransform(data, model):
    ''' Do factor analysis on the data
    '''
    return model.transform(data)
# TO DO Implement more FE methods

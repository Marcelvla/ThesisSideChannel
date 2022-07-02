## This file contains the code to run a multivariate template attack
# Imports
import scipy.io, scipy.stats
import numpy as np
import sys
import pdb
from collections import Counter
from feature_extraction import *
import time
import enquiries
import matplotlib.pyplot as plt
import operator

## Multivariate template functions
def multivarTemplate(U_red, keys, split):
    ''' Build multivariate templates for all keys in the training data
        returns two lists of len # keys for avg and covariance
    '''
    mu_k = []
    sigma_k = []
    for key in range(keys):
        print(f"Creating template for key {key}", end ="\r")
        train_key = load_train(key, split)
        train_red = np.dot(train_key, U_red)
        mu_k.append(np.mean(train_red, axis=0))
        sigma_k.append(np.cov(train_red.T))

    # mu_t = [np.mean(t, axis=0) for t in train]
    # sigma_t = [np.cov(t.T) for t in train]

    return (mu_k, sigma_k)

def multivarTemplateMod(model, keys, split):
    ''' Build multivariate templates for all keys in the training data
        returns two lists of len # keys for avg and covariance
    '''
    mu_k = []
    sigma_k = []
    for key in range(keys):
        print(f"Creating template for key {key}", end ="\r")
        train_key = load_train(key, split)
        train_red = dataTransform(train_key, model)
        mu_k.append(np.mean(train_red, axis=0))
        sigma_k.append(np.cov(train_red.T))

    # mu_t = [np.mean(t, axis=0) for t in train]
    # sigma_t = [np.cov(t.T) for t in train]

    return (mu_k, sigma_k)


def traceLMBDA(mu, inv, trace, key):
    ''' Calculate lambda for trace with candidate key
    '''
    return 0.5*np.dot(np.dot((trace - mu[key]).T, inv[key]), (trace - mu[key]))

def matchMultivarTemplate(mu, inv, trace, keys):
    ''' Match the trace to the templates and return the rankings of the keys
        for given trace
    '''
    # pdb.set_trace()
    lmbda = [traceLMBDA(mu, inv, trace, key) for key in range(keys)]

    return list(np.argsort(lmbda))

def misCLF(ct_clf):
    ''' Calculate misclassification rate for given counter
    '''
    ## Misclassification rate ##
    mis_clf = []
    # print(f"Multivariate Gauss Template for {n_features} features")
    for ct in range(len(ct_clf)):
        k = ct_clf[ct]
        mis = (sum(k.values()) - k[ct]) / sum(k.values())
        mis_clf.append(mis)
        # pdb.set_trace()
        # print(f"Key {ct} classifications:{k}")
        # print(f"Misclassification rate for O_{ct}: {mis}")

    return mis_clf

def classifyMultivarTemplate(n_features, keys, method, split=2072):
    ''' Classify the testsets with multivariate guassian distributions.
    '''
    ############ Dimensionality reduction ############
    # FE = ['PCA', 'NMF', 'FA', 'Other FE N/A']
    # choice = enquiries.choose('Choose FE method: ', FE)
    choice = method

    start = time.time()
    if choice == 'PCA':
        U_red = PCA(n_features, keys, split)
        end = time.time()
        print(f"Applied PCA for {n_features} features in {end - start} seconds")
    elif choice == 'NMF':
        U_red = FE_NMF(n_features, keys)
        end = time.time()
        print(f"Applied NMF for {n_features} features in {end - start} seconds")
    elif choice == 'FA':
        model = FE_FA(n_features, keys)
    else:
        print('No other FE methods implemented yet')

    #########################################################
    ############ Creating templates for all keys ############
    #########################################################
    start = time.time()
    try:
        mu, sigma = multivarTemplate(U_red, keys, split)
    except:
        mu, sigma = multivarTemplateMod(model, keys, split)
    # pdb.set_trace()

    # plot = enquiries.choose('Plot average of traces?: ', ['Yes', 'No'])
    # if plot == 'Yes':
    #     [plt.plot(mu[k]) for k in range(len(mu))]
    #     plt.show()

    inv = [np.linalg.inv(s) for s in sigma]
    print(f"Created templates in {(time.time() - start)} seconds")

    # pdb.set_trace()

    #################################################
    ############ Averaging num of traces ############
    #################################################
    # len_test = 3072 - split
    # n = [i for i in range(1, len_test) if len_test % i == 0]
    # num = int(enquiries.choose('Take the average of n traces?', n))

    ## Take for average of:
    num = [1, 5, 10, 20, 25]

    ######################################################
    ############ Classification for templates ############
    ######################################################
    # ct_clf = []
    ranks = []
    for key in range(keys):
        print(f"Classifying testset of key {key}", end ="\r")
        try:
            test_red = load_test(key, split, U_red)
        except:
            test_red = load_test_mod(key, split, model)

        ranks_num = []
        for n in num:
            print(f"Averaging {n} traces", end="\r")
            test_red_num = np.array([np.array(np.mean(test_red[n*i:n + n*i], axis=0)) for i in range(int(len(test_red) / n))])
            clf = [matchMultivarTemplate(mu, inv, t, keys) for t in test_red_num]
            # ct_clf.append(Counter([c[0] for c in clf]))
            ranks_num.append([c.index(key) for c in clf])

        ranks.append(ranks_num)

    # Misclassification rate
    # mis_clf = misCLF(ct_clf)

    return ranks

if __name__ == "__main__":
    # Do stuff
    n_features, keys, split = sys.argv[1:]
    print(f"Multivariate template for {n_features} features, split of {split}")
    ranks = classifyMultivarTemplate(int(n_features), int(keys), int(split))

    pdb.set_trace()
    print("done")


    # for m in [150, 300, 600, 800, 850, 900, 950]:
    #     classifyMultivarTemplate(train, test, m)

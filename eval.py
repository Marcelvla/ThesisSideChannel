# Imports
import numpy as np
from multivariateTA import classifyMultivarTemplate
import matplotlib.pyplot as plt
from scipy.stats import chi2
import seaborn as sns
import pdb
import pandas as pd
import sys
import enquiries
import time

NFEAT = [75,100,150,200,250]
NTRAC = [1,5,10,20,25]

## Plotting functions
def plotHM(data, n, num, method):
    ''' Plot the heatmap with MC for all keys against number of features
        for method
    '''
    ax = sns.heatmap(data, annot=True, square=True, linewidth=0.5, cmap="YlGnBu", yticklabels=n, xticklabels=num)
    ax.set_xlabel('Number of traces at a time')
    ax.set_ylabel(f'Number of features used for {method}')
    plt.show()

def plotSR(data, n, ft, method, legend=['PCA', 'NMF', 'FA']):
    ''' Plot average success rate graph for n_features
    '''
    colors = ["green", "greenyellow", "yellow", "aquamarine", "blue"]
    for d in range(len(data)):
        plt.plot(n, data[d], color=colors[d])
    plt.xlabel(f"Number of {ft} for FE with {method}")
    plt.ylabel("Average Success rate")
    plt.legend(legend)
    plt.show()

def plot(SR_ranks, method, n=NFEAT, num=NTRAC):
    ''' Function for plotting all kinds of results, asks which
    '''
    choices = ["Heatmap", "SR-features", "SR-traces", "Multi-FE"]
    plot = enquiries.choose("Heatmap/SR line/Multi?", choices)

    if plot == "Heatmap":
        plotHM(SR_ranks, n, num, method)
    elif plot == "SR-features":
        datalist = []
        for nu in range(len(num)):
            datalist.append([SR_ranks[feat][nu] for feat in range(len(n))])
        plotSR(datalist, n, "features", method)
    elif plot == "SR-traces":
        datalist = []
        for feat in range(len(n)):
            datalist.append([SR_ranks[feat][nu] for nu in range(len(num))])
        plotSR(datalist, num, "traces", method)
    elif plot == "Multi-FE":
        choices2 = ["Traces", "Features"]
        plot2 = enquiries.choose("Traces/Features?", choices2)
        ord = [1,5,10,20]
        ord_choice = enquiries.choose("Choose order", ord)

        if plot2 == "Traces":
            plot3 = enquiries.choose("Num traces", NTRAC)
            num_i = num.index(plot3)
            datalist = []

            for method in ['PCA', 'NMF', 'FA']:
                datalist.append([SR_ranks[method][ord_choice][f][num_i] for f in range(5)])

            plotSR(datalist, n, f"Features for {plot3} traces", "all FE's")

        elif plot2 == "Features":
            plot3 = enquiries.choose("Num features", NFEAT)
            feat_i = n.index(plot3)
            datalist = []

            for method in ['PCA', 'NMF', 'FA']:
                datalist.append(SR_ranks[method][ord_choice][feat_i])

            # pdb.set_trace()
            plotSR(datalist, num, f"Traces for {plot3} features", "all FE's")

    return "Done"

## Open results and create dicts for evaluation
def removeNAN(list):
    return [i for i in list if np.isnan(i) == False]

def rankScore(data, num, keys, order):
    ''' return the average successrate of given data
    '''
    num_ranks = []
    for i in range(len(num)):
        allranks = [data[k][i] for k in range(keys)]
        num_rankscore = [sum([1 for x in r if x <= order])/len(r) for r in allranks]
        num_ranks.append(np.mean(num_rankscore))

    return num_ranks

def openRes(method, n_features=NFEAT, num=NTRAC, keys=256, order=[10], score=True):
    ''' Read excel files with the results
    '''
    ranks_dict = {o:[] for o in order}
    # SR_ranks = []
    for n in n_features:
        filename = f"Output/{method}_{n}_feat.xlsx"
        print(f"Opening {filename}")
        read = [pd.read_excel(filename, sheet_name=str(k)) for k in range(keys)]
        print("Removing nan values... ")
        results = [[removeNAN(row[1].values) for row in r.iterrows()] for r in read]
        # pdb.set_trace()
        for o in order:
            ranks_temp = ranks_dict[o]
            if score:
                ranks_temp.append(rankScore(results, num, keys, o))
            else:
                ranks_temp.append(results)

    return ranks_dict

## McNemar's test functions
def mcNemars(ranks_1, ranks_2, n_feat, n_trace, m, keys=256, order=10):
    ''' Makes confusion matrices for two method rankings and returns McNemars
        X^2 values
        ranks[nfeatures][key][ntraces]
    '''
    DP, SP, SN, DN = 0, 0, 0, 0

    for k in range(keys):
        check1 = ranks_1[NFEAT.index(n_feat)][k][NTRAC.index(n_trace)]
        check2 = ranks_2[NFEAT.index(n_feat)][k][NTRAC.index(n_trace)]

        for i in range(int(1000/n_trace)):
            if check1[i] <= order:
                if check2[i] <= order:
                    DP += 1
                else:
                    SP += 1
            else:
                if check2[i] <= order:
                    SN += 1
                else:
                    DN += 1

    ctable = {f"{m[0]} positive":[DP, SP], f"{m[0]} negative":[SN, DN]}

    print(f"Contingency table for {m}, with {n_feat} features and {n_trace} traces, order {order}")
    print(pd.DataFrame(ctable, index=[f"{m[1]} positive", f"{m[1]} negative"]))

    chi_sq = (SN - SP) ** 2 / (SN + SP)
    P = chi2.pdf(chi_sq, df=1)

    print(f"McNemar's X^2 score for PCA and NMF is {chi_sq}")
    print(f"Probability of null hypothesis being true: {P}")

    return chi_sq

def nemar4all(ranks_1, ranks_2, m, keys=256, order=[10]):
    ''' Create nemar for all combinations of features and traces
    '''
    nemar_dict = {o:None for o in order}

    for o in order:
        nemar_matrix = [[mcNemars(ranks_1[o], ranks_2[o], feat, trac, m, keys, o) for trac in NTRAC] for feat in NFEAT]
        nemar_dict[o] = nemar_matrix

    return nemar_dict

## Function for feature extraction testing
def FEtest(n_features, keys, num, method, split=2072):
    ''' Test method for n_features and save the results to excel files
    '''
    for n in n_features:
        start_n = time.time()
        print(f"MTA for {n} features, with {method}")
        results = classifyMultivarTemplate(n, keys, method)
        writer = pd.ExcelWriter(f"Output/{method}_{n}_feat.xlsx", engine='xlsxwriter')
        for k in range(len(results)):
            res_k = pd.DataFrame(results[k])
            res_k.to_excel(writer, sheet_name=f"{k}", index=False)
        writer.save()
        print(f"Classified all traces in {(time.time() - start_n) / 60} minutes")

    return "Done"

if __name__ == "__main__":
    # DO STUFF LELELE
    # testdata = np.array([[1,2], [3,4]])
    # plotHM(testdata)

    ########### PCATEST ###########
    # PCAtest([75, 100, 125, 150, 175, 200], 8, 2072, [1, 4, 8, 10, 20, 25])
    # print("Done")

    ############ NMFTEST ###########

    # n_features, keys, split = sys.argv[1:]
    # print(f"Multivariate template for {n_features} features, split of {split}")
    # ranks = classifyMultivarTemplate(int(n_features), int(keys), int(split))

    # num_ranks = []
    # for i in range(len(num)):
    #     allranks = [ranks[k][i] for k in range(int(keys))]
    #     num_rankscore = [sum([1 for x in r if x <=10])/len(r) for r in allranks]
    #     num_ranks.append(np.mean(num_rankscore))
    # SR_ranks.append(num_ranks)

    ############### FATEST ###########

    ### MultiplotTest
    # from SRvars import *
    # SR_ranks = {'PCA':PCA, 'NMF':NMF, 'FA':FA}
    # plot(SR_ranks, 'all')

    ################ McNemars Test #############
    rPCA = openRes('PCA', score=False, order=NTRAC[:-1])
    rNMF = openRes('NMF', score=False, order=NTRAC[:-1])
    rFA = openRes('FA', score=False, order=NTRAC[:-1])
    # score = nemar4all(rPCA, rNMF, ['PCA', 'NMF'], keys=4, order=NTRAC[:-1])

    pdb.set_trace()
    print("done")

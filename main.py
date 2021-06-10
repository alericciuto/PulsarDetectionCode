import numpy
import matplotlib.pyplot as plt
from numpy import inf
from utils import *
from classifiers.MultivariateGaussianClassifier import *
from classifiers.NaiveBayesClassifier import *
from classifiers.TiedCovarianceGaussianClassifier import *
from classifiers.TiedDiagCovGaussianClassifier import *
from classifiers.LogisticRegression import *
from classifiers.LinearSVM import *
from classifiers.KernelSVM import *
from classifiers.GaussianMixtureModel import *
from transformers.PCA import *
from transformers.Gaussianizer import *
from tabulate import tabulate
from itertools import combinations
import time
import concurrent.futures
from tqdm import tqdm
import sklearn.model_selection
from scipy.interpolate import interp1d

labels_map = {
    0: 'Not a Pulsar',
    1: 'Pulsar'
}

features_map = {
    0: 'Mean of the integrated profile',
    1: 'Standard deviation of the integrated profile',
    2: 'Excess kurtosis of the integrated profile',
    3: 'Skewness of the integrated profile',
    4: 'Mean of the DM-SNR curve',
    5: 'Standard deviation of the DM-SNR curve',
    6: 'Excess kurtosis of the DM-SNR curve',
    7: 'Skewness of the DM-SNR curve'
}


def load_data(filepath):
    data = []
    labels = []
    with open(filepath) as f:
        for line in f:
            fields = line.split(',')
            data.append([float(feature) for feature in fields[0: 8]])
            labels.append(int(fields[8]))
    data = numpy.array(data).T  # transpose to have the features on the rows and the samples on the columns
    labels = numpy.array(labels)
    return data, labels


def plot_hist(D, L, folder='hist'):
    D_index_L = [D[:, L == i] for i in set(L)]

    for i in features_map.keys():
        plt.figure()
        plt.xlabel(features_map[i])
        for index, data in enumerate(D_index_L):
            plt.hist(data[i, :], bins=20, density=True, ec='black', alpha=0.5, label=labels_map[index])
        plt.legend()
        plt.tight_layout()
        plt.savefig('./plots/' + folder + '/hist_%d.png' % i)
    plt.show()


def plot_scatter(D, L, folder='scatter'):
    D_index_L = [D[:, L == i] for i in set(L)]

    for i in features_map.keys():
        for j in features_map.keys():
            if i == j:
                continue
            plt.figure()
            plt.xlabel(features_map[i])
            plt.ylabel(features_map[j])
            for index, data in enumerate(D_index_L):
                plt.scatter(data[i, :], data[j, :], label=labels_map[index], alpha=0.5)  # red
            plt.legend()
            plt.tight_layout()
            plt.savefig('./plots/' + folder + '/scatter_%d_%d.png' % (i, j))
        plt.show()


def plot_heatmap(D, folder='heatmap', subtitle='', color='YlGn'):
    corr_coef = numpy.corrcoef(D)

    fig, ax = plt.subplots()
    ax.imshow(corr_coef, cmap=color)
    for i in range(len(features_map)):
        for j in range(len(features_map)):
            ax.text(j, i, str(round(corr_coef[i, j], 1)), ha="center", va="center", color="r")

    fig.tight_layout()
    plt.savefig('./plots/' + folder + '/corr_coeff_' + subtitle + '.png')
    plt.show()


def compute_confusion_matrix(true, predicted):
    K = numpy.unique(numpy.concatenate((true, predicted))).size
    confusion_matrix = numpy.zeros((K, K), dtype=numpy.int64)

    # for i in range(len(true)):
    #     confusion_matrix[predicted[i], true[i]] += 1

    # 6 times speed up with respect to the previous code
    labels = numpy.hstack((vcol(predicted), vcol(true)))
    for indexes in set(combinations(tuple(list(range(K)) + list(range(K))), K)):
        equals = numpy.array(labels == indexes, dtype=numpy.int8).sum(axis=1) == K
        confusion_matrix[indexes] = numpy.array(equals, dtype=numpy.int8).sum()

    return confusion_matrix


def DCFu(prior, cfn, cfp, confusion_matrix):
    FNR = confusion_matrix[0, 1] / sum(confusion_matrix[:, 1])
    FPR = confusion_matrix[1, 0] / sum(confusion_matrix[:, 0])
    DCFu = prior * cfn * FNR + (1 - prior) * cfp * FPR
    return DCFu


def DCF(prior, cfn, cfp, confusion_matrix):
    DCFu_ = DCFu(prior, cfn, cfp, confusion_matrix)
    Bdummy = min(prior * cfn, (1 - prior) * cfp)
    return DCFu_ / Bdummy


def min_DCF(llr, labels, prior, cfn, cfp):
    scores = llr  # numpy.sort(llr)  # without sort improve performance

    mindcf = None
    for i, threshold in enumerate(scores):
        predicted = 0 + (llr > threshold)
        confusion_matrix_min_dcf = compute_confusion_matrix(labels, predicted)
        DCF_ = DCF(prior, cfn, cfp, confusion_matrix_min_dcf)
        mindcf = mindcf if mindcf is not None and mindcf <= DCF_ else DCF_

    return mindcf


def k_fold_min_DCF(D, L, K, Classifier, prior, class_args=(), transformers=[], transf_args=[]):
    if K <= 0 or K > D.shape[1]:
        raise Exception("K-Fold : K should be > 1 and <= " + str(D.shape[1]))
    nTest = int(D.shape[1] / K)
    nTrain = D.shape[1] - nTest
    numpy.random.seed(0)
    idx_1 = numpy.random.permutation(D.shape[1])
    # duplicate idx
    idx = numpy.concatenate((idx_1, idx_1))

    n_classes = len(set(L))
    llr = numpy.zeros(D.shape[1])
    for i in range(K):
        start = i * nTest
        idxTrain = idx[start: start + nTrain]
        idxTest = idx[start + nTrain: start + nTrain + nTest]

        DTR = D[:, idxTrain]
        DTE = D[:, idxTest]
        LTR = L[idxTrain]
        LTE = L[idxTest]

        for j, T in enumerate(transformers):
            transformer = T().fit(DTR, *transf_args[j])
            DTR = transformer.transform(DTR)
            DTE = transformer.transform(DTE)

        classifier = Classifier(DTR, LTR, *class_args)
        llr[idxTest] = classifier.llr(DTE)

    mindcf = min_DCF(llr, L, prior, 1, 1)

    return mindcf


def gaussianize(D):
    return Gaussianizer().fit(D).transform(D)


if __name__ == '__main__':
    DTR, LTR = load_data('./data/Train.txt')

    # DTR, _, LTR, _ = sklearn.model_selection.train_test_split(DTR.T, LTR, train_size=1 / 8, random_state=42)
    # DTR = DTR.T

    print_plots = False
    load_precomputed_data = [True, True, True, False]  # [False, False, False]
    store_computed_data = [False, False, False, True]  # [True, True, True]

    if load_precomputed_data[0]:
        DTR_G = numpy.load('./data/TrainGAU.npy')
    else:
        DTR_G = Gaussianizer().fit(DTR).transform(DTR)
        if store_computed_data[0]:
            numpy.save('./data/TrainGAU.npy', DTR_G)

    if print_plots:
        plot_hist(DTR, LTR)
        plot_scatter(DTR, LTR)

        plot_hist(DTR_G, LTR, folder='hist_GAU')
        plot_scatter(DTR_G, LTR, folder='scatter_GAU')

        plot_heatmap(DTR, subtitle='all', color='binary')
        plot_heatmap(DTR[:, LTR == 1], subtitle='pulsar', color='Blues')
        plot_heatmap(DTR[:, LTR == 0], subtitle='not_pulsar', color='Greens')

    #######################################################################################
    # Gaussian Models
    #######################################################################################

    classifier_name = numpy.array([
        'Full-Cov',
        'Diag-Cov',
        'Tied Full-Cov',
        'Tied Diag-Cov'
    ])
    classifiers = numpy.array([
        MultivariateGaussianClassifier,
        NaiveBayesClassifier,
        TiedCovarianceGaussianClassifier,
        TiedDiagCovGaussianClassifier
    ])

    priors = numpy.array([0.5, 0.1, 0.9])
    data = [DTR for i in range(6)]
    mindcf = numpy.zeros((len(data), classifiers.shape[0], priors.shape[0]))
    transformers = [
        [Gaussianizer],
        [PCA, Gaussianizer],
        [PCA, Gaussianizer],
        [PCA, Gaussianizer],
        [],
        [PCA]
    ]
    transf_args = [
        [()],
        [(7,), ()],
        [(6,), ()],
        [(5,), ()],
        [()],
        [(7,)]
    ]

    if len(data) != len(transformers) or len(transformers) != len(transf_args):
        raise Exception("Length of data/transformers/transf_args incorrect")
    elif classifiers.shape[0] != classifier_name.shape[0]:
        raise Exception("Length of classifiers/classifier_name incoherent")

    if load_precomputed_data[1]:
        mindcf = numpy.load('./data/minDCF_GAU_models.npy')

    results = []
    for d, D in enumerate(data):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            if not load_precomputed_data[1]:
                for i, c in enumerate(classifiers):
                    for j, p in enumerate(priors):
                        print(classifier_name[i] + " - prior = " + str(p) + " - data id = " + str(d))
                        results.append(executor.submit(k_fold_min_DCF, D, LTR, 5, c, p, (), transformers[d], transf_args[d]))
                        # print("min_DCF = " + str(mindcf[i, j]))
            for i, r in enumerate(tqdm(results)):
                mindcf[numpy.unravel_index(i, mindcf.shape, 'C')] = round(r.result(), 3)
            table = numpy.hstack((vcol(classifier_name), mindcf[d]))
            print(tabulate(table, headers=[""] + list(priors), tablefmt='fancy_grid'))

    if not store_computed_data[1]:
        numpy.save('./data/minDCF_GAU_models.npy', mindcf)

    #######################################################################################
    # Logistic Regression
    #######################################################################################

    classifier_name = numpy.array([
        'Log Reg',
        'Log Reg'
    ])
    classifiers = numpy.array([
        LogisticRegression,
        LogisticRegression
    ])
    transformers = [
        [Gaussianizer],
        []
    ]
    transf_args = [
        [()],
        [()]
    ]
    data = [DTR for i in range(2)]

    lamb = numpy.array([10 ** i for i in range(-6, 6)])
    lamb = numpy.array([numpy.linspace(lamb[i], lamb[i + 1], 10) for i in range(lamb.shape[0] - 1)]).reshape(-1)
    priors = numpy.array([0.5, 0.1, 0.9])

    if load_precomputed_data[2]:
        mindcf = numpy.load('./data/minDCF_LogReg_lamb.npy')
    else:
        mindcf = numpy.zeros((len(data), classifiers.shape[0], priors.shape[0], lamb.shape[0]))

    if len(data) != len(transformers) or len(transformers) != len(transf_args):
        raise Exception("Length of data/transformers/transf_args incoherent")
    elif classifiers.shape[0] != classifier_name.shape[0]:
        raise Exception("Length of classifiers/classifier_name incoherent")

    results = []
    for d, D in enumerate(data):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            if not load_precomputed_data[2]:
                for i, c in enumerate(classifiers):
                    for j, p in enumerate(priors):
                        print(classifier_name[i] + " - prior = " + str(p) + " - data id = " + str(d))
                        for k, l in enumerate(lamb):
                            results.append(executor.submit(k_fold_min_DCF, D, LTR, 5, c, p, (l,), transformers[d], transf_args[d]))
            for i, r in enumerate(tqdm(results)):
                mindcf[numpy.unravel_index(i, mindcf.shape, 'C')] = round(r.result(), 3)
            table = numpy.hstack((vcol(classifier_name), mindcf[d].min(axis=2, initial=inf)))
            print(tabulate(table, headers=[""] + list(priors), tablefmt='fancy_grid'))

    if store_computed_data[2]:
        numpy.save('./data/minDCF_LogReg_lamb.npy', mindcf)

    for d in range(len(data)):
        for i in range(mindcf[d].shape[0]):
            plt.figure()
            for j, p in enumerate(priors):
                plt.plot(lamb, mindcf[d, i, j], label='minDCF (π = ' + str(p) + ')')
            plt.xlabel('λ')
            plt.ylabel('min DCF')
            plt.legend()
            plt.xscale('log')
            plt.tight_layout()
            name = 'Raw' if i == 0 else 'Gaussianized'
            if store_computed_data[2]:
                plt.savefig('./plots/mindcf_training/LogReg_lamb_' + str(d) + '_' + str(i) + '.png')
            plt.show()

    ##################################################################################
    # Linear SVM
    ##################################################################################

    classifier_name = numpy.array([
        'SVM (no class balancing)',
        'SVM (with class balancing)'
    ])
    classifiers = numpy.array([
        LinearSVM,
        LinearSVM
    ])
    transformers = [
        [],
    ]
    transf_args = [
        [()],
    ]
    data = [DTR]

    Ci = numpy.array([10 ** i for i in range(-3, 3)])
    priors = numpy.array([0.5, 0.1, 0.9])

    if load_precomputed_data[3]:
        mindcf = numpy.load('./data/minDCF_SVM_C.npy')
    else:
        mindcf = numpy.zeros((len(data), classifiers.shape[0], priors.shape[0], Ci.shape[0]))

    if len(data) != len(transformers) or len(transformers) != len(transf_args):
        raise Exception("Length of data/transformers/transf_args incoherent")
    elif classifiers.shape[0] != classifier_name.shape[0]:
        raise Exception("Length of classifiers/classifier_name incoherent")

    results = []
    for d, D in enumerate(data):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            if not load_precomputed_data[3]:
                for i, c in enumerate(classifiers):
                    for j, p in enumerate(priors):
                        print(classifier_name[i] + " - prior = " + str(p) + " - data id = " + str(d))
                        for k, C in enumerate(Ci):
                            results.append(
                                executor.submit(k_fold_min_DCF, D, LTR, 5, c, p, (1, C, p, None,), transformers[d], transf_args[d]))
            for i, r in enumerate(tqdm(results)):
                mindcf[numpy.unravel_index(i, mindcf.shape, 'C')] = round(r.result(), 3)
            table = numpy.hstack((vcol(classifier_name), mindcf[d].min(axis=2, initial=inf)))
            print(tabulate(table, headers=[""] + list(priors), tablefmt='fancy_grid'))

    if store_computed_data[3]:
        numpy.save('./data/minDCF_SVM_C.npy', mindcf)

    for d in range(len(data)):
        for i in range(mindcf[d].shape[0]):
            plt.figure()
            for j, p in enumerate(priors):
                plt.plot(lamb, mindcf[d, i, j], label='minDCF (π = ' + str(p) + ')')
            plt.xlabel('λ')
            plt.ylabel('min DCF')
            plt.legend()
            plt.xscale('log')
            plt.tight_layout()
            if store_computed_data[2]:
                plt.savefig('./plots/mindcf_training/SVM_C_' + str(d) + '_' + str(i) + '.png')
            plt.show()

import numpy
import matplotlib.pyplot as plt
from numpy import inf
from scipy.stats import norm

from utils import *
from classifiers.MultivariateGaussianClassifier import *
from classifiers.NaiveBayesClassifier import *
from classifiers.TiedCovarianceGaussianClassifier import *
from classifiers.TiedDiagCovGaussianClassifier import *
from classifiers.LogisticRegression import *
from classifiers.LinearSVM import *
from classifiers.KernelSVM import *
from classifiers.GaussianMixtureModel import *
from tabulate import tabulate
from itertools import combinations
import time

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


def gaussianize(X, Z):
    # Compute rank transformed features of Z over X (training samples)
    Y = []
    N = X.shape[1]
    M = Z.shape[1]
    for i, x in enumerate(X):
        y = numpy.array([x[Z[i, j] < x].shape[0] for j in range(M)])  # equivalent to sum ones if Z[i, j] < X[i, :]
        y = (y + 1) / (N + 2)  # this avoid +inf and -inf with ppf
        Y.append(y)
    # Return percent point function (inverse of the cumulative distribution function)
    return norm.ppf(numpy.array(Y))


def covariance(D):
    N = D.shape[1]
    mu = vcol(D.mean(axis=1))
    DC = D - mu
    C = numpy.dot(DC, DC.T) / N
    return C


def PCA(D, m):
    C = covariance(D)
    s, U = numpy.linalg.eigh(C)
    P = U[:, ::-1][:, 0:m]
    DP = numpy.dot(P.T, D)
    return DP


def k_fold(D, L, K, Classifier, prior):
    if K <= 1 or K > D.shape[1]:
        raise Exception("K-Fold : K should be > 1 and <= " + str(D.shape[1]))
    nTest = int(D.shape[1] / K)
    nTrain = D.shape[1] - nTest
    numpy.random.seed(0)
    idx = numpy.random.permutation(D.shape[1])
    # duplicate idx
    idx = numpy.concatenate((idx, idx))

    n_classes = len(set(L))
    errors = 0
    for i in range(K):
        start = i * nTest
        idxTrain = idx[start: start + nTrain]
        idxTest = idx[start + nTrain: start + nTrain + nTest]

        DTR = D[:, idxTrain]
        DTE = D[:, idxTest]
        LTR = L[idxTrain]
        LTE = L[idxTest]

        classifier = Classifier(DTrain=DTR, LTrain=LTR)
        predicted, _ = classifier.predict(DTest=DTE, LTest=LTE, prior=prior)
        errors += predicted[LTE != predicted].shape[0]

    return errors / (nTest * K)


def compute_confusion_matrix(true, predicted):
    K = numpy.unique(numpy.concatenate((true, predicted))).size
    confusion_matrix = numpy.zeros((K, K), dtype=numpy.int64)

    # for i in range(len(true)):
    #     confusion_matrix[predicted[i], true[i]] += 1

    # 6 times speed up with respect to up
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
    for threshold in scores:
        predicted = 0 + (llr > threshold)
        confusion_matrix_min_dcf = compute_confusion_matrix(labels, predicted)
        DCF_ = DCF(prior, cfn, cfp, confusion_matrix_min_dcf)
        mindcf = mindcf if mindcf is not None and mindcf <= DCF_ else DCF_

    return mindcf


def k_fold_min_DCF(D, L, K, Classifier, prior, args=()):
    if K <= 0 or K > D.shape[1]:
        raise Exception("K-Fold : K should be > 1 and <= " + str(D.shape[1]))
    nTest = int(D.shape[1] / K)
    nTrain = D.shape[1] - nTest
    numpy.random.seed(0)
    idx = numpy.random.permutation(D.shape[1])
    # duplicate idx
    idx = numpy.concatenate((idx, idx))

    n_classes = len(set(L))
    mindcf = 0
    for i in range(K):
        start = i * nTest
        idxTrain = idx[start: start + nTrain]
        idxTest = idx[start + nTrain: start + nTrain + nTest]

        DTR = D[:, idxTrain]
        DTE = D[:, idxTest]
        LTR = L[idxTrain]
        LTE = L[idxTest]

        classifier = Classifier(DTR, LTR, *args)
        mindcf += min_DCF(classifier.llr(DTE), LTE, prior, 1, 1)

    return mindcf / K


if __name__ == '__main__':
    DTR, LTR = load_data('./data/Train.txt')

    print_plots = False

    try:
        DTR_G = numpy.load('./data/TrainGAU.npy')
    except FileNotFoundError:
        DTR_G = gaussianize(DTR, DTR)
        numpy.save('./data/TrainGAU.npy', DTR_G)

    if print_plots:
        plot_hist(DTR, LTR)
        plot_scatter(DTR, LTR)

        plot_hist(DTR_G, LTR, folder='hist_GAU')
        plot_scatter(DTR_G, LTR, folder='scatter_GAU')

        plot_heatmap(DTR, subtitle='all', color='binary')
        plot_heatmap(DTR[:, LTR == 1], subtitle='pulsar', color='Blues')
        plot_heatmap(DTR[:, LTR == 0], subtitle='not_pulsar', color='Greens')

    DTR_G_PCA_7 = PCA(DTR_G, 7)
    DTR_G_PCA_6 = PCA(DTR_G, 6)
    DTR_G_PCA_5 = PCA(DTR_G, 5)
    DTR_G_PCA_4 = PCA(DTR_G, 4)

    # print(k_fold(DTR_PCA_4, LTR, 5, MultivariateGaussianClassifier, prior=vcol(numpy.array([0.1, 0.9]))))
    # print(k_fold(DTR_PCA_4, LTR, 5, NaiveBayesClassifier, prior=vcol(numpy.array([0.1, 0.9]))))
    # print(k_fold(DTR_PCA_4, LTR, 5, TiedCovarianceGaussianClassifier, prior=vcol(numpy.array([0.1, 0.9]))))
    # print(k_fold(DTR_PCA_4, LTR, 5, TiedDiagCovGaussianClassifier, prior=vcol(numpy.array([0.1, 0.9]))))

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
    mindcf = numpy.zeros((classifiers.shape[0], priors.shape[0]))
    data = []  # [DTR, DTR_G]  # [DTR_G, DTR_G_PCA_7, DTR_G_PCA_6, DTR_G_PCA_5, DTR]

    for d, D in enumerate(data):
        for i, c in enumerate(classifiers):
            for j, p in enumerate(priors):
                print(classifier_name[i] + " - prior = " + str(p) + " - data id = " + str(d))
                mindcf[i, j] = round(k_fold_min_DCF(D, LTR, K=5, Classifier=c, prior=p), 3)
                print("min_DCF = " + str(mindcf[i, j]))
        table = numpy.hstack((vcol(classifier_name), mindcf))
        print(tabulate(table, headers=[""] + list(priors), tablefmt='fancy_grid'))

    ##################################################################################

    classifier_name = numpy.array([
        'Log Reg',
        'Log Reg'
    ])
    classifiers = numpy.array([
        LogisticRegression,
        LogisticRegression
    ])

    lamb = numpy.array([10 ** i for i in range(-5, 5)])
    lamb = numpy.array([numpy.linspace(lamb[i], lamb[i + 1], 5) for i in range(lamb.shape[0] - 1)]).reshape(-1)
    priors = numpy.array([0.5, 0.1, 0.9])

    try:
        mindcf = numpy.load('./data/minDCF_LogReg_lamb.npy')
    except FileNotFoundError:
        mindcf = numpy.zeros((classifiers.shape[0], priors.shape[0], lamb.shape[0]))
        data = [DTR, DTR_G]

        for d, D in enumerate(data):
            for i, c in enumerate(classifiers):
                for j, p in enumerate(priors):
                    print(classifier_name[i] + " - prior = " + str(p) + " - data id = " + str(d))
                    for k, l in enumerate(lamb):
                        mindcf[i, j, k] = round(k_fold_min_DCF(D, LTR, K=5, Classifier=c, args=(l, None,), prior=p), 3)
                        print("min_DCF = " + str(mindcf[i, j]), end='\r')
                    print()
            table = numpy.hstack((vcol(classifier_name), mindcf.min(axis=2, initial=inf)))
            print(tabulate(table, headers=[""] + list(priors), tablefmt='fancy_grid'))

        numpy.save('./data/minDCF_LogReg_lamb.npy', mindcf)

    for i in range(mindcf.shape[0]):
        table = numpy.hstack((vcol(classifier_name[i:i+1]), vrow(mindcf[i].min(axis=1, initial=inf))))
        print(tabulate(table, headers=[""] + list(priors), tablefmt='fancy_grid'))
        plt.figure()
        for j, p in enumerate(priors):
            plt.plot(lamb, mindcf[i, j, :], label='minDCF (piT = ' + str(p) + ')')
        plt.xlabel('λ')
        plt.ylabel('min DCF')
        plt.legend()
        plt.xscale('log')
        plt.tight_layout()
        name = 'Raw' if i == 0 else 'Gaussianized'
        plt.savefig('./plots/mindcf_training/' + name + '_LogReg_lamb.png')
        plt.show()


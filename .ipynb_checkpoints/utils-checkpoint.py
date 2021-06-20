import numpy
import matplotlib.pyplot as plt
import scipy.special
import time
from itertools import combinations
import concurrent.futures
from tqdm import tqdm


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

def vcol(vec):
    return vec.reshape(vec.shape[0], 1)


def vrow(vec):
    return vec.reshape(1, -1)


def GAU_pdf(x, mu, var):
    return (2 * numpy.pi * var) ** (-0.5) * \
           numpy.exp(- ((x - mu) ** 2) / (2 * var))


def covariance(D):
    N = D.shape[1]
    mu = vcol(D.mean(axis=1))
    DC = D - mu
    C = numpy.dot(DC, DC.T) / N
    return C


def GAU_logpdf(x, mu, var):
    D = x.shape[1]
    invC = numpy.linalg.inv(var)
    _, logDetC = numpy.linalg.slogdet(var)
    return (- D / 2 * numpy.log(2 * numpy.pi)) \
           - logDetC / 2 \
           - (numpy.dot((x - mu).T, invC).T * (x - mu)).sum(axis=0) / 2


def logpdf_GAU_ND(x, mu, C):
    M = x.shape[0]
    invC = numpy.linalg.inv(C)
    _, logDetC = numpy.linalg.slogdet(C)
    result = numpy.diag(
        - M / 2 * numpy.log(2 * numpy.pi) - \
        logDetC / 2 - \
        numpy.dot(numpy.dot((x - mu).T, invC), (x - mu)) / 2
    )
    resultWithoutDiag = - M / 2 * numpy.log(2 * numpy.pi) - \
                        logDetC / 2 - \
                        (numpy.dot((x - mu).T, invC).T * (x - mu)).sum(axis=0) / 2
    return resultWithoutDiag


def k_fold_err_rate(D, L, K, Classifier, prior):
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


def plot_dens_gmm(data, gmm):
    plt.figure()
    kwargs = dict(histtype='stepfilled', alpha=0.5, bins=30, density=True)
    plt.hist(data.reshape(-1), **kwargs)
    GAU = numpy.zeros(data.shape)
    x = numpy.linspace(min(data.reshape(-1)), max(data.reshape(-1)), GAU.shape[1])
    for i in range(len(gmm)):
        GAU += gmm[i][0] * GAU_pdf(x, gmm[i][1], gmm[i][2])
    plt.plot(x, GAU.reshape(-1), linewidth=4)
    plt.tight_layout()
    plt.show()


def logpdf_GMM(X, gmm):
    S = []
    for w, mu, C in gmm:
        pdf = logpdf_GAU_ND(X, mu, C)
        pdf += numpy.log(w)
        S.append(pdf)
    S = numpy.array(S)

    logdens = scipy.special.logsumexp(S, axis=0)
    log_SPost = numpy.array([S[i, :] - logdens for i in range(len(S))])
    SPost = numpy.exp(log_SPost)
    return logdens, SPost


def EM_algorithm(data, gmm_init, threshold=1e-6, diag=False, tied=False, psi=0.01):
    posterior = logpdf_GMM(data, gmm_init)[1]

    prev_avg_ll = None
    n_iter = 0
    while True:
        n_iter += 1
        Z = vcol(numpy.sum(posterior, axis=1))
        F = numpy.dot(posterior, data.T)
        S = numpy.array([numpy.dot(posterior[i, :] * data, data.T) for i in range(len(Z))])

        w = Z / numpy.sum(Z.reshape(-1))
        mu = (F / Z).T

        C = [S[i] / Z[i] - numpy.dot(vcol(mu[:, i]), vcol(mu[:, i]).T) for i in range(len(Z))]
        if diag:
            C = [C[i] * numpy.eye(C[i].shape[0]) for i in range(len(Z))]
        elif tied:
            tied_C = 1 / data.shape[1] * sum([Z[i] * C[i] for i in range(len(Z))])
            C = [tied_C for _ in range(len(Z))]

        gmm = [(w[i, 0], vcol(mu[:, i]), contraint_eigenvalues(C[i], psi)) for i in range(len(Z))]

        ll, posterior = logpdf_GMM(data, gmm)
        avg_ll = numpy.average(ll)
#         print((prev_avg_ll, avg_ll))
        if prev_avg_ll is not None and avg_ll - prev_avg_ll < threshold:
            break
        prev_avg_ll = avg_ll

    return gmm


def contraint_eigenvalues(C, psi=0.01):
    U, s, _ = numpy.linalg.svd(C)
    s[s < psi] = psi
    C = numpy.dot(U, vcol(s) * U.T)
    return C


def LBG_algorithm(mu, C, n_g, alpha):
    w = 1.0
    gmm = [(w, mu, C)]

    for i in range(n_g // 2):
        w = [gmm[j][0] / 2 for j in numpy.arange(len(gmm) * 2) // 2]

        U, s, Vh = numpy.linalg.svd(C)
        d = [- (U[:, 0:1] * s[0] ** 0.5 * alpha) * (-1) ** k for k in range(2)]
        mu = [gmm[j][1] + d_i for j in numpy.arange(len(gmm)) for d_i in d]

        gmm = [(w[j], mu[j], C) for j in range(len(w))]

    return gmm


def LBG_and_EM(data, n_g, alpha, gmm_init=None, threshold=1e-6, diag=False, tied=False, psi=0.01):
    if gmm_init is None:
        mu = vcol(numpy.mean(data, axis=1))
        C = numpy.dot(data - mu, (data - mu).T) / data.shape[1]
        C = contraint_eigenvalues(C, psi)
        gmm_init = [(1, mu, C)]

    gmm = gmm_init
    gmm_all = [gmm]
    while len(gmm) < n_g:
        gmm = []
#         print("LBG")
        for j, gmm_i in enumerate(gmm_init):
            gmm += LBG_algorithm(gmm_i[1], gmm_i[2], 2, alpha)
#         print("EM")
        gmm = EM_algorithm(data, gmm, threshold, diag, tied)
        gmm_all.append(gmm)
        gmm_init = gmm

    return gmm_all
    # return gmm



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


def min_DCF(llr, labels, prior, cfn, cfp, returnThreshold=False):
    scores = llr  # numpy.sort(llr)  # without sort improve performance

    opt_threshold = None
    mindcf = None
    for i, threshold in enumerate(scores):
        predicted = 0 + (llr > threshold)
        confusion_matrix_min_dcf = compute_confusion_matrix(labels, predicted)
        DCF_ = DCF(prior, cfn, cfp, confusion_matrix_min_dcf)
        if mindcf is None or mindcf > DCF_:
            mindcf = DCF_
            opt_threshold = threshold

    if returnThreshold:
        return mindcf, opt_threshold
    else:
        return mindcf



def plot_mindcf(D, mindcf, priors, hyperparams, xlabel, store, path, names=None):
    for d in range(len(D)):
        for i in range(mindcf[d].shape[0]):
            plt.figure()
            for j, p in enumerate(priors):
                plt.plot(hyperparams, mindcf[d, i, j], label='minDCF (π = ' + str(p) + ')')
            plt.xlabel(xlabel)
            plt.ylabel('min DCF')
            plt.legend()
            plt.grid(True)
            plt.xscale('log')
            plt.tight_layout()
            if store:
                plt.savefig(path + (names[d] if names is not None else '') + '.png')
            plt.show()


def compute_llr(DTR, LTR, Classifier, class_args, DTE, transformers, transf_args):
    for j, T in enumerate(transformers):
        transformer = T().fit(DTR, *transf_args[j])
        DTR = transformer.transform(DTR)
        DTE = transformer.transform(DTE)

    classifier = Classifier(DTR, LTR, *class_args)
    return classifier.llr(DTE)


def train_GMM(DTR, LTR, classifier, class_args, transformers, transf_args):
    for j, T in enumerate(transformers):
        transformer = T().fit(DTR, *transf_args[j])
        DTR = transformer.transform(DTR)
        # DTE = transformer.transform(DTE)

    classifier = classifier(DTR, LTR, *class_args)
    return classifier


def k_fold_min_DCF(D, L, K, Classifier, prior, class_args=(), transformers=[], transf_args=[]):
    if K <= 0 or K > D.shape[1]:
        raise Exception("K-Fold : K should be > 1 and <= " + str(D.shape[1]))
    nTest = int(D.shape[1] / K)
    nTrain = D.shape[1] - nTest
    numpy.random.seed(0)
    idx_1 = numpy.random.permutation(D.shape[1])
    # duplicate idx in order to do a circular array
    idx = numpy.concatenate((idx_1, idx_1))

    llr = numpy.zeros(D.shape[1])

    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for i in range(K):
            start = i * nTest
            idxTrain = idx[start: start + nTrain]
            idxTest = idx[start + nTrain: start + nTrain + nTest]
    
            DTR = D[:, idxTrain]
            DTE = D[:, idxTest]
            LTR = L[idxTrain]
    
            future = executor.submit(compute_llr, DTR, LTR, Classifier, class_args, DTE, transformers, transf_args)
            results.append(future)
    
        for i, r in enumerate(results):
            start = i * nTest
            idxTest = idx[start + nTrain: start + nTrain + nTest]
            llr[idxTest] = r.result()

#     for i in range(K):
#         start = i * nTest
#         idxTrain = idx[start: start + nTrain]
#         idxTest = idx[start + nTrain: start + nTrain + nTest]

#         DTR = D[:, idxTrain]
#         DTE = D[:, idxTest]
#         LTR = L[idxTrain]

#         llr[idxTest] = compute_llr(DTR, LTR, Classifier, class_args, DTE, transformers, transf_args)

    mindcf = min_DCF(llr, L, prior, 1, 1)

    return mindcf


def k_fold_llr(D, L, K, Classifier, class_args=(), transformers=[], transf_args=[]):
    if K <= 0 or K > D.shape[1]:
        raise Exception("K-Fold : K should be > 1 and <= " + str(D.shape[1]))
    nTest = int(D.shape[1] / K)
    nTrain = D.shape[1] - nTest
    numpy.random.seed(0)
    idx_1 = numpy.random.permutation(D.shape[1])
    # duplicate idx in order to do a circular array
    idx = numpy.concatenate((idx_1, idx_1))

    llr = numpy.zeros(D.shape[1])

    for i in range(K):
        start = i * nTest
        idxTrain = idx[start: start + nTrain]
        idxTest = idx[start + nTrain: start + nTrain + nTest]

        DTR = D[:, idxTrain]
        DTE = D[:, idxTest]
        LTR = L[idxTrain]

        llr[idxTest] = compute_llr(DTR, LTR, Classifier, class_args, DTE, transformers, transf_args)

    return llr


def optimal_binary_bayes_decisions(prior, cfn, cfp, llr, useThreshold=False):
    if useThreshold:
        predicted = 0 + (llr > prior)
    else:
        predicted = 0 + (llr > - numpy.log(prior * cfn / ((1 - prior) * cfp)))
    return predicted


def optimal_threshold(llr, labels, prior):
    numpy.random.seed(0)
    ids = numpy.random.permutation(llr.shape[0])
    N = llr.shape[0] // 2
    ids_min = ids[0: N]
    ids_act = ids[N:]
    _, threshold = min_DCF(llr[ids_min], labels[ids_min], prior, 1, 1, returnThreshold=True)
    predicted = optimal_binary_bayes_decisions(threshold, 1, 1, llr[ids_act], useThreshold=True)
    confusion_matrix = compute_confusion_matrix(labels[ids_act], predicted)
    actdcf = DCF(prior, 1, 1, confusion_matrix)
    mindcf = min_DCF(llr[ids_act], labels[ids_act], prior, 1, 1)
    return mindcf, actdcf


def k_fold_GMM(D, L, K, prior, classifiers=[], class_args=(), transformers=[], transf_args=[], mode=None):
    if K <= 0 or K > D.shape[1]:
        raise Exception("K-Fold : K should be > 1 and <= " + str(D.shape[1]))
    nTest = int(D.shape[1] / K)
    nTrain = D.shape[1] - nTest
    numpy.random.seed(0)
    idx_1 = numpy.random.permutation(D.shape[1])
    # duplicate idx in order to do a circular array
    idx = numpy.concatenate((idx_1, idx_1))

    llr = numpy.zeros(D.shape[1])
    if len(classifiers) == 1 and mode == 'train':
        classifier = classifiers[0]
        classifiers = []

#     results = []
#     with concurrent.futures.ProcessPoolExecutor() as executor:
#         with tqdm(total=K, colour='blue') as progress:
#             for i in range(K):
#                 start = i * nTest
#                 idxTrain = idx[start: start + nTrain]
#                 idxTest = idx[start + nTrain: start + nTrain + nTest]

#                 DTR = D[:, idxTrain]
#                 DTE = D[:, idxTest]
#                 LTR = L[idxTrain]

#                 if mode == 'train':
#                     future = executor.submit(train_GMM, DTR, LTR, classifier, class_args, transformers, transf_args)
#                     future.add_done_callback(lambda _: progress.update())
#                     results.append(future)
#                 else:
#                     for j, T in enumerate(transformers):
#                         transformer = T().fit(DTR, *transf_args[j])
#                         DTE = transformer.transform(DTE)
#                     llr[idxTest] = classifiers[i].llr(DTE, *class_args)

#             if mode == 'train':
#                 for i, r in enumerate(results):
#                     classifiers.append(r.result())
#                     return classifiers
#             else:
#                 mindcf = min_DCF(llr, L, prior, 1, 1)
#                 return mindcf

    for i in range(K):
        start = i * nTest
        idxTrain = idx[start: start + nTrain]
        idxTest = idx[start + nTrain: start + nTrain + nTest]

        DTR = D[:, idxTrain]
        DTE = D[:, idxTest]
        LTR = L[idxTrain]

        if mode == 'train':
            classifiers.append(train_GMM(DTR, LTR, classifier, class_args, transformers, transf_args))
        else:
            for j, T in enumerate(transformers):
                transformer = T().fit(DTR, *transf_args[j])
                DTE = transformer.transform(DTE)
            llr[idxTest] = classifiers[i].llr(DTE, *class_args)

    if mode == 'train':
        return classifiers
    else:
        mindcf = min_DCF(llr, L, prior, 1, 1)
        return mindcf



class Timer:

    def __init__(self):
        print("Timer - Start")
        self.start = time.perf_counter()

    def print_time_passed(self):
        print("Timer - time passed = " + str(time.perf_counter() - self.start))
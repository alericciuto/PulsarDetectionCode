import numpy
import matplotlib.pyplot as plt
import scipy.special


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
    print("EM_algorithm START")
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
        print(str(n_iter) + "° step - LL = " + str(avg_ll), end='\r')
        if prev_avg_ll is not None and avg_ll - prev_avg_ll < threshold:
            print(str(n_iter) + "° step - LL = " + str(avg_ll))
            break
        prev_avg_ll = avg_ll

    print("EM_algorithm END")
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
    while len(gmm) < n_g:
        gmm = []
        for j, gmm_i in enumerate(gmm_init):
            gmm += LBG_algorithm(gmm_i[1], gmm_i[2], 2, alpha)
        gmm = EM_algorithm(data, gmm, threshold, diag, tied)
        gmm_init = gmm

    return gmm
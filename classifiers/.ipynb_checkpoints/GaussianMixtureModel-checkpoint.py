import scipy.special
from utils import *


class GaussianMixtureModel:

    def __init__(self, DTrain, LTrain, diag=False, tied=False, n_g=1, alpha=0.1, threshold=1e-3, psi=0.01):
        self.DTrain = DTrain
        self.LTrain = LTrain
        self.n_classes = len(set(LTrain))
        self.NTrain = self.DTrain.shape[1]
        self.n_g = n_g
        self.diag = diag
        self.tied = tied
        self.alpha = alpha
        self.psi = psi
        self.threshold = threshold
        self.gmm = self.estimate_params()

    def estimate_params(self):
        gmm = [[] for _ in range(int(numpy.log2(self.n_g)))]
        # gmm = []
        for i in set(self.LTrain):
            DTRi = self.DTrain[:, self.LTrain == i]
            print("starting LBG_and_EM")
            gmm_per_class = LBG_and_EM(
            # gmm += LBG_and_EM(
                DTRi,
                n_g=self.n_g,
                threshold=self.threshold,
                alpha=self.alpha,
                diag=self.diag,
                tied=self.tied,
                psi=self.psi
            )
            print("end")
            for j, gmm_j in enumerate(gmm_per_class):
                if len(gmm) > j:
                    gmm[j] += gmm_j
                else:
                    gmm.append(gmm_j)
        return gmm

    def predict(self, DTest, LTest=None, prior=None, n_g=None):  # n_g=None
        if n_g is None:
            n_g = self.n_g
        prior = numpy.log(prior if prior is not None else 1.0 / self.n_classes)
        SComponent = logpdf_GMM(DTest, self.gmm[int(numpy.log2(n_g))])[1]
        # SComponent = logpdf_GMM(DTest, self.gmm)[1]  # self.gmm[int(numpy.log2(n_g))]
        if SComponent.shape[0] > self.n_classes:
            SComponent = numpy.array([numpy.sum(SComponent[i * n_g:(i + 1) * n_g, :], axis=0) for i in range(self.n_classes)])
        SJoint = SComponent + prior
        marg_log_dens = sum([scipy.special.logsumexp(SJoint[c]) for c in range(self.n_classes)])
        SPost = numpy.exp(SJoint - marg_log_dens)
        predicted = numpy.argmax(SPost, axis=0)
        err_rate = predicted[LTest != predicted].shape[0] / LTest.shape[0] if LTest is not None else None
        return predicted, err_rate

    def llr(self, DTest, n_g=None, class0=0, class1=1):  # n_g=None
        if n_g is None:
            n_g = self.n_g
        SComponent = logpdf_GMM(DTest, self.gmm[int(numpy.log2(n_g))])[1]  # self.gmm[int(numpy.log2(n_g))]
        if SComponent.shape[0] > self.n_classes:
            SComponent = numpy.array([numpy.sum(SComponent[i * n_g:(i + 1) * n_g, :], axis=0) for i in range(self.n_classes)])
        llr = SComponent[class1, :] - SComponent[class0, :]
        return llr

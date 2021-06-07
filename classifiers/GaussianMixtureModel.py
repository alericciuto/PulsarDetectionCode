import scipy.special
from utils import *


class GaussianMixtureModel:

    def __init__(self, DTrain, LTrain, diag=False, tied=False, n_g=1, alpha=0.1, threshold=1e-6, psi=0.01):
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
        gmm = []
        for i in set(self.LTrain):
            DTRi = self.DTrain[:, self.LTrain == i]
            gmm += LBG_and_EM(
                DTRi,
                n_g=self.n_g,
                threshold=self.threshold,
                alpha=self.alpha,
                diag=self.diag,
                tied=self.tied,
                psi=self.psi
            )
        return gmm

    def predict(self, DTest, LTest=None, prior=None):
        prior = numpy.log(prior if prior is not None else 1.0 / self.n_classes)
        SComponent = logpdf_GMM(DTest, self.gmm)[1]
        SJoint = SComponent + prior
        marg_log_dens = scipy.special.logsumexp(SJoint, axis=0)
        SPost = numpy.exp(SJoint - marg_log_dens)
        predicted = numpy.argmax(SPost, axis=0) // self.n_g
        err_rate = predicted[LTest != predicted].shape[0] / LTest.shape[0] if LTest is not None else None
        return predicted, err_rate

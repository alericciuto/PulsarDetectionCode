import numpy
import scipy.special
from utils import vcol, vrow, GAU_logpdf


class MultivariateGaussianClassifier:

    def __init__(self, DTrain, LTrain):
        self.DTrain = DTrain
        self.LTrain = LTrain
        self.n_classes = len(set(LTrain))
        self.NTrain = self.DTrain.shape[1]
        self.params = [self.estimate_params(c) for c in range(self.n_classes)]

    def estimate_params(self, c):
        D = self.DTrain[:, self.LTrain == c]
        N = D.shape[1]
        mu = vcol(D.mean(1))
        var = numpy.dot(D - mu, (D - mu).T) / N
        return mu, var

    def predict(self, DTest, LTest=None, prior=None):
        prior = numpy.log(prior if prior is not None else 1.0 / self.n_classes)
        S = numpy.array([GAU_logpdf(DTest, self.params[c][0], self.params[c][1]) for c in range(self.n_classes)])
        SJoint = S + prior
        marg_log_dens = scipy.special.logsumexp(SJoint, axis=0)
        SPost = numpy.exp(SJoint - marg_log_dens)
        predicted = numpy.argmax(SPost, axis=0)
        err_rate = predicted[LTest != predicted].shape[0] / LTest.shape[0] if LTest is not None else None
        return predicted, err_rate

    def llr(self, DTest, class0=0, class1=1):
        S = numpy.array([GAU_logpdf(DTest, self.params[c][0], self.params[c][1]) for c in [class0, class1]])
        llr = S[class1, :] - S[class0, :]
        return llr

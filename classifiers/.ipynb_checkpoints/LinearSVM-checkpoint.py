import scipy.special
import scipy.optimize
from utils import *


class LinearSVM:

    def __init__(self, DTrain, LTrain, K=1, C=0.1, prior=None, alpha0=None):
        self.n_samples = DTrain.shape[1]
        self.K = K
        self.C = C
        self.DTrain = numpy.vstack((DTrain, numpy.ones(
            self.n_samples) * self.K))  # increase K, means decrease the effect of regularization on b bias.
        self.LTrain = LTrain
        self.alpha0 = numpy.zeros(self.n_samples) if alpha0 is None else alpha0
        if prior is not None:
            emp_prior_F = DTrain[:, LTrain == 0].shape[1] / DTrain.shape[1]
            emp_prior_T = DTrain[:, LTrain == 1].shape[1] / DTrain.shape[1]
            Cf = self.C * (1 - prior) / emp_prior_F
            Ct = self.C * prior / emp_prior_T
        else:
            Cf = Ct = C
        self.bounds = numpy.array([(0, Cf if LTrain[i] == 0 else Ct) for i in range(self.n_samples)])
        # H = sum of zi * zj * xi * xj
        self.H = numpy.dot(vcol(2 * self.LTrain - 1), vrow(2 * self.LTrain - 1)) * numpy.dot(self.DTrain.T, self.DTrain)
        self.w_best, self.alpha = self.compute_params()  # b_best is inside w_best

    def compute_params(self):
        alpha, f, d = scipy.optimize.fmin_l_bfgs_b(
            self.dual_SVM,
            self.alpha0,
            bounds=self.bounds
        )
        return numpy.dot(self.DTrain, vcol(alpha * (2 * self.LTrain - 1))), alpha  # b_best is inside w_best

    def dual_SVM(self, alpha=None):
        alpha = self.alpha if alpha is None else alpha
        return (
            1 / 2 * numpy.dot(numpy.dot(vrow(alpha), self.H), vcol(alpha))[0, 0] - numpy.sum(alpha),
            (numpy.dot(self.H, vcol(alpha)) - 1).reshape(-1)
        )

    def primal_SVM(self):
        G = 1 - (2 * self.LTrain - 1) * numpy.dot(self.w_best.T, self.DTrain).reshape(-1)
        zero = numpy.zeros(G.shape[0])
        maximum = numpy.maximum(zero, G)
        return 1 / 2 * (self.w_best * self.w_best).sum() + self.C * numpy.sum(maximum)

    def predict(self, DTest, LTest=None):
        DTest = numpy.vstack((DTest, numpy.ones(DTest.shape[1]) * self.K))
        scores = numpy.dot(self.w_best.T, DTest).ravel()
        predicted = (scores > 0).astype(int)
        err_rate = predicted[predicted != LTest].shape[0] / predicted.shape[0] if LTest is not None else None
        return predicted, err_rate

    def llr(self, DTest):
        DTest = numpy.vstack((DTest, numpy.ones(DTest.shape[1]) * self.K))
        S = numpy.dot(self.w_best.T, DTest).ravel()
        return S

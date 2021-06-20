import scipy.special
import scipy.optimize
from utils import *


def poly_kernel(x1, x2, K, args=(0, 2)):
    c = args[0]
    d = args[1]
    return numpy.power(numpy.dot(x1.T, x2) + c, d) + K * K


def RBF_kernel(x1, x2, K, args):
    gamma = args[0]
    diff = numpy.zeros((x1.shape[1], x2.shape[1]))
    for i in range(x1.shape[1]):
        diff[i, :] = ((vcol(x1[:, i]) - x2) ** 2).sum(axis=0)
    return numpy.exp(- gamma * diff) + K * K


class KernelSVM:

    def __init__(self, DTrain, LTrain, K=1, C=0.1, prior=None, alpha0=None, kernel='poly', kernel_args=()):
        self.n_samples = DTrain.shape[1]
        self.K = K
        self.C = C
        self.DTrain = DTrain
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
        self.kernel_args = kernel_args
        self.kernel = poly_kernel if kernel == 'poly' else RBF_kernel
        # H = sum of zi * zj * xi * xj
        self.H = numpy.dot(vcol(2 * self.LTrain - 1), vrow(2 * self.LTrain - 1)) * \
                 self.kernel(DTrain, DTrain, self.K, self.kernel_args)
        self.w_best, self.alpha = self.compute_params()  # b_best is inside w_best

    def compute_params(self):
        alpha, f, d = scipy.optimize.fmin_l_bfgs_b(
            self.dual_SVM,
            self.alpha0,
            bounds=self.bounds,
            factr=1e6
        )
        return numpy.dot(self.DTrain, vcol(alpha * (2 * self.LTrain - 1))), alpha  # b_best is inside w_best

    def dual_SVM(self, alpha=None):
        alpha = self.alpha if alpha is None else alpha
        return (
            1 / 2 * numpy.dot(numpy.dot(vrow(alpha), self.H), vcol(alpha))[0, 0] - numpy.sum(alpha),
            (numpy.dot(self.H, vcol(alpha)) - 1).reshape(-1)
        )

    def predict(self, DTest, LTest=None):
        scores = numpy.dot((vcol(self.alpha) * vcol(2 * self.LTrain - 1)).T,
                           self.kernel(self.DTrain, DTest, self.K, self.kernel_args)).ravel()
        predicted = (scores > 0).astype(int)
        err_rate = predicted[predicted != LTest].shape[0] / predicted.shape[0] if LTest is not None else None
        return predicted, err_rate

    def llr(self, DTest):
        scores = numpy.dot((vcol(self.alpha) * vcol(2 * self.LTrain - 1)).T,
                      self.kernel(self.DTrain, DTest, self.K, self.kernel_args)).ravel()
        return scores

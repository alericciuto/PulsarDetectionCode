import numpy
import scipy.special
import scipy.optimize


def vrow(vec):
    return vec.reshape(1, -1)


def vcol(vec):
    return vec.reshape(vec.shape[0], 1)


def poly_kernel(x1, x2, K, args=(0, 2)):
    c = args[0]
    d = args[1]
    return numpy.power(numpy.dot(x1.T, x2) + c, d) + K * K


def RBF_kernel(x1, x2, K, args):
    gamma = args[0]
    diff = numpy.zeros((x1.shape[1], x2.shape[1]))
    for i in range(x1.shape[1]):
        for j in range(x2.shape[1]):
            diff[i, j] = ((x1[:, i] - x2[:, j]) * (x1[:, i] - x2[:, j])).sum()
    return numpy.exp(- gamma * diff) + K * K


class KernelSVM:

    def __init__(self, DTrain, LTrain, K=1, C=0.1, alpha0=None, kernel='poly', kernel_args=()):
        self.n_samples = DTrain.shape[1]
        self.K = K
        self.C = C
        self.DTrain = DTrain
        self.LTrain = LTrain
        self.alpha0 = numpy.zeros(self.n_samples) if alpha0 is None else alpha0
        self.bounds = numpy.array([(0, self.C) for i in range(self.n_samples)])
        if kernel is not None:
            self.kernel_args = kernel_args
            if kernel == 'poly':
                self.kernel = poly_kernel
            elif kernel == 'rbf':
                self.kernel = RBF_kernel
        # H = sum of zi * zj * xi * xj
        self.H = numpy.dot(vcol(2 * self.LTrain - 1), vrow(2 * self.LTrain - 1)) * \
                 self.kernel(DTrain, DTrain, self.K, self.kernel_args)
        self.w_best, self.alpha = self.compute_params()  # b_best is inside w_best

    def compute_params(self):
        alpha, f, d = scipy.optimize.fmin_l_bfgs_b(
            self.dual_SVM,
            self.alpha0,
            bounds=self.bounds,
            factr=1.0
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

    def llr(self, DTest, class0=0, class1=1):
        S = numpy.dot((vcol(self.alpha) * vcol(2 * self.LTrain - 1)).T,
                      self.kernel(self.DTrain, DTest, self.K, self.kernel_args)).ravel()
        llr = S[class0, :] / S[class1, :]
        return llr

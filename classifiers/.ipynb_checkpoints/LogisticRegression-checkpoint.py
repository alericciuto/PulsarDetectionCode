import scipy.special
import scipy.optimize
from utils import *


class LogisticRegression:

    def __init__(self, DTrain, LTrain, lamb=1e-3, x0=None, prior=None):
        self.DTrain = DTrain
        self.LTrain = LTrain
        self.n_classes = len(set(LTrain))
        self.n_samples = self.DTrain.shape[1]
        self.n_samples_per_class = numpy.array([LTrain[LTrain == i].shape[0] for i in range(self.n_classes)])
        self.prior = numpy.array([LTrain[LTrain == i].shape[0] / LTrain.shape[0] for i in range(self.n_classes)] if prior is None else prior)
        self.n_features = self.DTrain.shape[0]
        self.lamb = lamb
        self.x0 = numpy.zeros(self.n_features + 1) if x0 is None else x0
        self.w_best, self.b_best = self.compute_params()

    def compute_params(self):
        x, f, d = scipy.optimize.fmin_l_bfgs_b(
            self.logreg_obj,
            self.x0,
            approx_grad=True)
        return x[0: -1], \
               x[-1]

    def logreg_obj(self, v=None):
        if v is None:
            w, b = self.w_best, self.b_best
        else:
            w, b = vrow(v[0:-1]), v[-1]
        y = (numpy.dot(w, self.DTrain) + b).reshape(-1)
        z = 2 * self.LTrain - 1
        exp = - (z * y).astype(numpy.float128)
        loss = sum([numpy.sum(numpy.log1p(numpy.exp(exp[self.LTrain == i]))) * self.prior[i] / self.n_samples_per_class[i] for i in range(self.n_classes)])

        return self.lamb * (w * w).sum() / 2 + loss

    def predict(self, DTest, LTest=None):
        scores = (numpy.dot(vrow(self.w_best), DTest) + self.b_best).ravel()
        predicted = (scores > 0).astype(int)
        err_rate = predicted[predicted != LTest].shape[0] / predicted.shape[0] if LTest is not None else None
        return predicted, err_rate

    def llr(self, DTest):
        S = (numpy.dot(vrow(self.w_best), DTest) + self.b_best).ravel()
        return S
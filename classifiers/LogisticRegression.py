import numpy
import scipy.special
import scipy.optimize


class LogisticRegression:

    def __init__(self, DTrain, LTrain, lamb=1e-3, x0=None):
        self.DTrain = DTrain
        self.LTrain = LTrain
        self.n_classes = len(set(LTrain))
        self.n_samples = self.DTrain.shape[1]
        self.n_features = self.DTrain.shape[0]
        self.lamb = lamb
        self.x0 = numpy.zeros((self.n_features + 1) * self.n_classes) if x0 is None else x0
        self.w_best, self.b_best = self.compute_params()

    def compute_params(self):
        x, f, d = scipy.optimize.fmin_l_bfgs_b(
            self.logreg_obj,
            self.x0,
            approx_grad=True)
        return x[0:self.n_features * self.n_classes].reshape(self.n_features, self.n_classes), \
               x[-self.n_classes:].reshape(self.n_classes, 1)

    def logreg_obj(self, v):
        w, b = v[0:self.n_features * self.n_classes].reshape(self.n_features, self.n_classes), \
               v[-self.n_classes:].reshape(self.n_classes, 1)
        S = numpy.dot(w.T, self.DTrain) + b
        y_log = S - scipy.special.logsumexp(S, axis=0)
        T = numpy.zeros((self.n_classes, self.n_samples))
        for i in range(self.n_samples):
            T[self.LTrain[i], i] = 1
        loss = numpy.sum(T * y_log) / self.n_samples
        return self.lamb * (w * w).sum() / 2 - loss

    def predict(self, DTest, LTest=None):
        scores = numpy.dot(self.w_best.T, DTest) + self.b_best
        predicted = numpy.argmax(scores, axis=0)
        err_rate = predicted[predicted != LTest].shape[0] / predicted.shape[0] if LTest is not None else None
        return predicted, err_rate

    def llr(self, DTest, class0=0, class1=1):
        S = numpy.dot(self.w_best.T, DTest) + self.b_best
        llr = S[class0, :] / S[class1, :]
        return llr

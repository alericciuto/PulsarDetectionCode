from utils import *
from scipy.stats import norm


class Gaussianizer:

    def __init__(self):
        self.D = None
        self.N = 0

    def fit(self, D):
        self.D = D
        self.N = D.shape[1]
        return self

    def transform(self, Z):
        F = Z.shape[0]
        M = Z.shape[1]
        Y = numpy.empty((F, M))
        for j in range(M):
            zj = vcol(Z[:, j])
            Y[:, j] = numpy.array(zj < self.D, dtype=numpy.int).sum(axis=1)
        Y = (Y + 1) / (self.N + 2)
        return norm.ppf(Y)

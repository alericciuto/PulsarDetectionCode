from utils import *


class PCA:

    def __init__(self):
        self.P = None

    def fit(self, D, m):
        C = covariance(D)
        s, U = numpy.linalg.eigh(C)
        self.P = U[:, ::-1][:, 0:m]
        return self

    def transform(self, Z):
        return numpy.dot(self.P.T, Z)

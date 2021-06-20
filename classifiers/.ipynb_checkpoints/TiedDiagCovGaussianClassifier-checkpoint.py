import numpy
import scipy.special
from utils import vcol, vrow, GAU_logpdf


# Different from MultivariateGaussianClassifier because the covariance matrice is unique for all the classes
# The covariance matrice is equal to the within-class covariance matrix
class TiedDiagCovGaussianClassifier:

    def __init__(self, DTrain, LTrain):
        self.DTrain = DTrain
        self.LTrain = LTrain
        self.n_classes = len(set(LTrain))
        self.NTrain = self.DTrain.shape[1]
        self.means = [self.estimate_means(c) for c in range(self.n_classes)]
        self.covariance = self.estimate_within_class_covariance()

    def estimate_means(self, c):
        D = self.DTrain[:, self.LTrain == c]
        mu = vcol(D.mean(1))
        return mu

    def estimate_within_class_covariance(self):
        DC = [self.DTrain[:, self.LTrain == c] - self.means[c] for c in range(self.n_classes)]
        SW = numpy.array(sum([numpy.dot(DC[c], DC[c].T) for c in range(self.n_classes)]) / self.NTrain)
        return SW * numpy.eye(SW.shape[0])

    def predict(self, DTest, LTest=None, prior=None):
        prior = numpy.log(prior if prior is not None else 1.0 / self.n_classes)
        S = numpy.array([GAU_logpdf(DTest, self.means[c], self.covariance) for c in range(self.n_classes)])
        SJoint = S + prior
        marg_log_dens = scipy.special.logsumexp(SJoint, axis=0)
        SPost = numpy.exp(SJoint - marg_log_dens)
        predicted = numpy.argmax(SPost, axis=0)
        err_rate = predicted[LTest != predicted].shape[0] / LTest.shape[0] if LTest is not None else None
        return predicted, err_rate

    def llr(self, DTest, class0=0, class1=1):
        S = numpy.array([GAU_logpdf(DTest, self.means[c], self.covariance) for c in [class0, class1]])
        llr = S[class1, :] - S[class0, :]
        return llr


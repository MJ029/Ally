# Import required libraries
from sklearn.linear_model import LogisticRegression
from Core.PreProcessing import Skeleton
from Core.FeatureExtraction import DimensionalityReduction


class LogSR(Skeleton, DimensionalityReduction):

    def __init__(self):
        self.estimator = LogisticRegression()
        Skeleton.__init__()
        DimensionalityReduction.__init__(self, self.estimator)

    def build_model(self, x_train, y_train, **kwargs):
        try:
            self.estimator = LogisticRegression()
            self.estimator.predict(x_train, y_train)
            return self.estimator
        except Exception as exp:
            raise Exception('CLASS: LogisticRegression: {}'.format(exp))

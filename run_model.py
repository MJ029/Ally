# Import required libraries
# Core Usage
from Core.Model import Workflow

# Regression Usage
from Regression.OrdinaryLeastSquare import MLR, POLY
from Regression.SupportVectorRegression import SVR
from Regression.DecisionTreeRegression import DTR
from Regression.RandomForestRegression import RFR

# Classification Usage
from Classification.LinearClassifiers import LogSR

import warnings

# Ignore warnings
warnings.filterwarnings('ignore')


class RunModel(Workflow):

    def __init__(self, input_path, job_type, target, features=['ALL'], algorithms=['ALL'], file_format='csv',
                 missing_value=False, search_value=None, feature_selection=None, feature_scaling=False,
                 encode_categorical=False, binary_transform=False, categorical_features=None, binary_classifier=False
                 ):
        """
        Object Initialization for run_model
        :param job_type: type of model that you are trying to apply
            It could be any one of below
            - Regression
            - Classification
        :param input_path: path to load input file
        :param target: Target variable [Dependent Variable(Y)]
        :param features: Features of matrix of X [Independent Variable]
        :param algorithms: list of algorithms to be applied according to job_type, it can be anyone from below
            - ALL --> ALL IN
            - MLR --> MultiLinearRegression
            - POLY --> PolynomialRegression
            - SVR --> SupportVectorRegression
            - DTREE --> DecisionTreeRegression & DecisionTreeClassification
            - RFR --> RandomForestRegression
            - SIGMOID --> LogisticRegression
            - KNN --> K-NearestNeighbours Classifier
            - SVM --> SupportVectorMachine Classifier
            - RFC --> RandomForestClassifier
            - BAYESIAN --> NaiveBayesClassifier
        :param file_format: input file format to be read
        :param missing_value: True, if dataset contains any missing value, default to False
        :param search_value: applicable when missing_value is True, default to None
        :param feature_selection: Apply Feature selection on Regression model which follows the below feature selection
            procedures. for feature selection we are using statsmodel library, it is a iterative approach,
            by default it is None
                - BE: BackWard-Elimination
                - FS: Forward-Selection
        :param feature_scaling: scale matix of feature od X, default to False, it must be true when feature_selection
            is enabled
        :param encode_categorical: encodes categorical data to integer format when it sets True, default to False
        :param binary_transform: binzrie encoded categorical data to binary format, default to False
        :param categorical_features: list of categorical features to be encoded
        :param binary_classifier: defines whether the target variable is Binary or not, defaulted to False.
            If true Logistic Regression would take into consideration.
            If false remaining classifiers only considered
        """
        super().__init__(input_path, file_format, target, features, missing_value, search_value, job_type,
                         feature_selection, feature_scaling, encode_categorical, binary_transform,
                         categorical_features)
        self.__job_type = job_type
        self.__binary_classifier = binary_classifier
        self.__algorithms = self.evaluate_algorithms(algorithms)
        self.estimator_summary = dict()

    def evaluate_algorithms(self, algorithms):
        """
        Function will evaluate input parameters and drives the model accordingly
        :param algorithms: list of algorithms that you wanted to apply on dataset
        :return:
        """
        # Raise Exception if algorithms is not a list
        if not isinstance(algorithms, list):
            raise TypeError('"algorithms" must be a type of list not a type of {}'.format(type(algorithms)))

        # Checks Regression algorithms in supported list
        if self.__job_type.lower() == 'regression':
            supported_algorithms = ['ALL', 'MLR', 'POLY', 'SVR', 'DTREE', 'RFR']
            for alg in algorithms:
                if alg not in supported_algorithms:
                    raise Exception('Given regression algorithm "{}" not supported currently, '
                                    'please use any one from the list {}'.format(alg, supported_algorithms))

        # Checks Classification algorithms in supported list
        elif self.__job_type.lower() == 'classification':
            supported_algorithms = ['ALL', 'SIGMOID', 'KNN', 'SVM', 'DTREE', 'RFC', 'BAYESIAN']
            for alg in algorithms:
                if alg not in supported_algorithms:
                    raise Exception('Given classification algorithm "{}" not supported currently, '
                                    'please use any one from the list {}'.format(alg, supported_algorithms))

        # Raise Exception if the give job_type is not applicable
        else:
            raise TypeError('Given job_type "{}" not supported currently'.format(self.__job_type))

        return algorithms

    def run(self):
        """
        Function to run models based on the algorithms parameter
        :return: Status summary of job
        """
        for algo in self.__algorithms:
            if algo.upper() == 'MLR':
                print('Running Multi-linear Regression...!')
                estimator = MLR()
                self.estimator_summary[algo] = super().run_estimator(estimator)
            elif algo.upper() == 'POLY':
                print('Running Polynomial Regression...!')
                estimator = POLY()
                self.estimator_summary[algo] = super().run_estimator(estimator)
            elif algo.upper() == 'SVR':
                print('Running Support Vector Regression...!')
                estimator = SVR()
                self.estimator_summary[algo] = super().run_estimator(estimator)
            elif algo.upper() == 'DTREE':
                if self.__job_type.lower() == 'regression':
                    print('Running Decision Tree Regression...!')
                    estimator = DTR()
                    self.estimator_summary[algo] = super().run_estimator(estimator)
                else:
                    raise TypeError('DTree for Classification not found')
            elif algo.upper() == 'RFR':
                print('Running Random Forest Regression...!')
                estimator = RFR()
                self.estimator_summary[algo] = super().run_estimator(estimator)
            elif algo.upper() == 'SIGMOID':
                if self.__binary_classifier:
                    print('Running Logistic Regression...!')
                    estimator = LogSR()
                    self.estimator_summary[algo] = super().run_estimator(estimator)

    def get_summary(self, model_name='ALL'):
        """
        Function will return model summary for specified mode, default to ALL
        :param model_name: name of model which we are trying to get summary
        :return: best fit model summary
        """
        if not isinstance(model_name, str):
            raise TypeError('model_name must be string, not a type of {}'.format(type(model_name)))

        if model_name != 'ALL':
            print('Model Summary: {}'.format(model_name))
            print(self.estimator_summary[model_name])
        else:
            print('Model Summary:')
            print(self.estimator_summary)


if __name__ == '__main__':
    # Test Car-MPG Prediction
    # Load input dataset [carMPG, CarPricePrediction]
    file_path = 'E:\\Project Work\\SpyderWorkspace\\UCI-UseCases\\dataset\\Regression\\carMPG.csv'

    # Initialize object
    obj = RunModel(job_type='Regression',
                   input_path=file_path,
                   target='MPG',
                   features=['Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model_year',
                             'Origin'],
                   algorithms=['MLR', 'POLY', 'SVR', 'DTREE', 'RFR'],
                   file_format='csv',
                   missing_value=True,
                   search_value='?',
                   feature_selection=None,
                   feature_scaling=True,
                   encode_categorical=None,
                   binary_transform=False,
                   categorical_features=None
                   )
    # Run models
    obj.run()

    # Get Model Summary
    obj.get_summary('MLR')
    obj.get_summary('POLY')
    obj.get_summary('SVR')
    obj.get_summary('DTREE')
    obj.get_summary('RFR')

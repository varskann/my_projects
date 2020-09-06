"""
implement linear regression class
- Using sklearn / available modules
- From scratch, doing all calculations
"""

__author__ = "Kanishk Varshney"


import numpy as np
import pandas as pd
from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split


class LinearRegression():
    """Implements Linear Regression API"""
    def __init__(self, training_file="calcofi/bottle.csv", test_file=""):
        self.training_file = training_file
        self.test_file = test_file

        self.x_train = []
        self.y_train = []
        self.x_val = []
        self.y_val = []

        self._read_data()

        self.num_train_examples = self.x_train.shape[0]
        self.num_val_examples = self.x_val.shape[0]


        self.num_features = self.x_train.shape[1]
        print("Number of training features = %s\n\n"%self.num_features)

        self.num_iterations = 1000
        self.alpha = 0.03
        self.model = None

    def _read_data(self, y_value="Apparent Temperature (C)"):

        ## Read training data
        self.x_train, self.y_train = self._read_training_data(y_value)
        print("Training examples available = %s" % (len(self.y_train)))

        ## Read test data
        self.x_val, self.y_val = self._read_test_data(y_value)
        print("Validation examples available = %s" % (len(self.y_val)))

        if len(self.y_val) == 0:
            print("No validation file, creating split from training data")
            self._create_split(test_size=0.2)
            print("Updated train-val split :\n"
                  "\tTrain = %s\n"
                  "\tValidation = %s\n"%(len(self.y_train), len(self.y_val)))



    def _create_split(self, test_size=0.2):
        """
        create train val split, if no validation file is present
        Args:
            test_size (float): percentage of data to split as validation data

        Returns:

        """
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.x_train, self.y_train,
                                                                              test_size=test_size)


    def _read_training_data(self, y_value):
        """Read training file """
        df = pd.read_csv(self.training_file)

        df = df.select_dtypes(include=[np.float])

        df_y = df[y_value]
        df_x = df.drop([y_value], axis=1)

        return np.array(df_x.values.tolist()), np.array([[y] for y in df_y.values.tolist()])

    def _read_test_data(self, y_value):
        """Read test file """
        if self.test_file == "":
            return np.array([]), np.array([])

        df = pd.read_csv(self.test_file)

        df = df.select_dtypes(include=[np.float])

        df_y = df[y_value]
        df_x = df.drop([y_value], axis=1)

        return np.array(df_x.values.tolist()), np.array([[y] for y in df_y.values.tolist()])

    def _normalize_data(self):
        x_train_normalized = (self.x_train - np.mean(self.x_train, 0)) / np.std(self.x_train, 0)
        x_train_normalized = x_train_normalized[:, ~np.all(np.isnan(x_train_normalized), axis=0)]
        self.x_train = np.hstack((np.ones((self.num_train_examples, 1)), x_train_normalized))

        x_val_normalized = (self.x_val - np.mean(self.x_val, 0)) / np.std(self.x_val, 0)
        x_val_normalized = x_val_normalized[:, ~np.all(np.isnan(x_val_normalized), axis=0)]
        self.x_val = np.hstack((np.ones((self.num_val_examples, 1)), x_val_normalized))

        self.num_features = self.x_train.shape[1]


    def fit(self):
        """train the linear regression"""

        self._normalize_data()

        self.w_ = np.zeros((self.num_features, 1))

        for _ in range(self.num_iterations):
            y_pred = np.dot(self.x_train, self.w_)
            error = y_pred - self.y_train
            gradient = np.dot(self.x_train.T, error)
            self.w_ -= (self.alpha/self.num_train_examples) * gradient

    def fit_sklearn(self):
        """fit sklearn LinearRegression model"""
        self.model = linear_model.LinearRegression()
        self.model.fit(self.x_train, self.y_train)

    def predict(self):
        """Predict output based on model learnt"""
        if self.model:
            y_pred = self.model.predict(self.x_val)
        else:
            y_pred = np.dot(self.x_val, self.w_)

        self._performance(y_pred)

    def plot_regression(self):


    def _performance(self, y_pred):

        print("=========== Linear Regression Performance Benchmark ==========")
        print('Mean Absolute Error:', metrics.mean_absolute_error(self.y_val, y_pred))
        print('Mean Squared Error:', metrics.mean_squared_error(self.y_val, y_pred))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(self.y_val, y_pred)))
        print('R2 Score:', metrics.r2_score(self.y_val, y_pred))
        print("______________________________________________________________")



if __name__ == "__main__":
    regression_model = LinearRegression(training_file="szeged-weather/weatherHistory.csv")
    regression_model.fit_sklearn()
    regression_model.predict()

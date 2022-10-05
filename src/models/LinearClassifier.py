from itertools import combinations, product
import logging

import numpy as np

import preprocessing as pp
from functions import implementations as linear_models
from functions.helpers import calculate_mse, predict_labels


class LinearClassifier:
    def __init__(self,
                 jets_column=22,
                 column_names=None,
                 merge_jet_2_3=True,
                 with_bias_column=True,
                 with_feature_expansion=True,
                 test_split_ratio=0.33,
                 with_feature_selection=False,
                 remove_highly_correlated_features=True,
                 remove_highly_correlated_features_treshold=0.9,
                 lasso_lambda=50,
                 feature_selection_threshold=1e-5,
                 base_model='ridge_regression',
                 with_oversampling=False,
                 initial_w = None,
                 max_iters = 1000,
                 gamma = 1e-6,
                 tol = 1e-3,
                 ridge_lambda = 1.0,
                 remove_columns_with_many_undefined_values=True,
                ):


        # TODO group these parameters, this is a mess
        # Data preprocesing parameters
        self.jets_column = jets_column
        self.with_bias_column = with_bias_column
        self.with_feature_expansion = with_feature_expansion
        self.merge_jet_2_3 = merge_jet_2_3
        self.test_split_ratio = test_split_ratio
        self.with_oversampling = with_oversampling
        self.remove_highly_correlated_features = remove_highly_correlated_features
        self.remove_highly_correlated_features_treshold = remove_highly_correlated_features_treshold
        self.remove_columns_with_many_undefined_values = remove_columns_with_many_undefined_values

        self.model_evaluation = None
        self.column_mean_values = None
        self.standardisation_params = None
        self.min_per_column = {}
        self.feature_selection = None

        self.base_model = base_model

        # Feature selection based on lasso
        self.with_feature_selection = with_feature_selection
        self.lasso_lambda = lasso_lambda
        self.feature_selection_threshold = feature_selection_threshold

        # Default parameters for iterative schemes
        self.initial_w = initial_w
        self.max_iters = max_iters
        self.gamma = gamma
        self.tol = tol

        # Penalizer
        self.ridge_lambda = ridge_lambda

        self.logistic = False
        self.column_names = np.array(column_names)


    def _prepare_data_(self, X, fit = False):
        """General data preparation.
        It saves what it does in the model with `fit == True`,
        and reapplies it to the given X with `fit == False`

        Steps:
            1. dummy encoding of jet variable
            2. feature selection based on correlation
            3. feature expansion
            4. standardisation (0 mean, 1 std)
            5. handling of null values (setting them to the mean)
            6. adding bias column
        """
        X = X.copy()

        # handling of categorical column PRI_jet_num
        if self.merge_jet_2_3:
            X[X[:, self.jets_column] == 3, self.jets_column] = 2

        # this handles the multi-model-case
        do_dummy_encoding = len(np.unique(X[:, self.jets_column])) > 1
        if do_dummy_encoding:
            # we drop the last column since we the information is already included in the other three (each jet is pairwise disjoint)
            dummy_encoding_of_jet_column = pp.dummy_encoding(X, self.jets_column)[:, :-1]

        #X = pp.handle_undefined_values_for_jets(X, self.jets_column) # this is still different for each jet

        # keeping track of the column name is a huge pain...
        # here we remove the jets_column, it just makes everything way easier
        jets_column = X[:, self.jets_column]
        X = X[:, pp.without(range(X.shape[1]), [self.jets_column])]
        if fit:
            self.column_names = [c for idx, c in enumerate(self.column_names) if idx != self.jets_column]

        # removal of columns which have too many undefined values
        if self.remove_columns_with_many_undefined_values:
            if fit:
                self.too_many_defined_values = (X == -999).mean(axis=0) > 0.70
                self.column_names = [c for idx, c in enumerate(self.column_names) \
                                     if idx not in np.where(self.too_many_defined_values)[0]]

            logging.debug('removed {0} features, because too many values where undefined, {1} are left'\
                          .format(np.sum(self.too_many_defined_values),
                                  np.sum(~self.too_many_defined_values)))
            X = X[:, ~self.too_many_defined_values]

        if self.remove_highly_correlated_features:
            if fit:
                self.highly_correlated_columns = \
                    pp.highly_correlated_columns(X, threshold=self.remove_highly_correlated_features_treshold)
                logging.debug("feature selection through correlation removed {} features"\
                      .format(len(self.highly_correlated_columns)))
                self.column_names = [c for idx, c in enumerate(self.column_names) \
                                     if idx not in self.highly_correlated_columns]

            X = X[:, pp.without(range(X.shape[1]), self.highly_correlated_columns)]

        if self.with_feature_expansion:
            if fit:
                X, self.column_names = pp.feature_base_expansion(X, self.column_names)
            else:
                X = pp.feature_base_expansion(X)

        if fit:
            self.column_mean_values = pp.default_values_for_each_column(X)
            X, *self.standardisation_params = pp.standardize(X)
        else:
            X, *_ = pp.standardize(X, *self.standardisation_params)

        X = pp.overwrite_null_values_with_mean(X, self.column_mean_values)

        logging.debug('mean of X_({0}) (without dummy):{1}'.format('train' if fit else 'test', np.mean(X)))

        if do_dummy_encoding:
            X = np.hstack((X, dummy_encoding_of_jet_column))
            if fit:
                self.column_names += ['dummy_encoding_class_{}'.format(i) \
                                      for i in range(dummy_encoding_of_jet_column.shape[1])]


        if self.with_bias_column:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
            if fit:
                self.column_names += ['bias']

        return X


    def _prepare_labels_(self, y):
        # Prepare labels for fitting with logistic regression
        return (y + 1) / 2

    def _model_parameters_(self):
        return {k: v for k, v in self.__dict__.items() if k in ['jets_column',
                                                                'with_bias_column',
                                                                'with_feature_expansion',
                                                                'merge_jet_2_3',
                                                                'test_split_ratio',
                                                                'with_oversampling',
                                                                'remove_highly_correlated_features',
                                                                'remove_highly_correlated_features_treshold',
                                                                'base_model',
                                                                'with_feature_selection',
                                                                'lasso_lambda',
                                                                'max_iters',
                                                                'gamma',
                                                                'tol',
                                                                'ridge_lambda'
                                                                ]}


    def _fit_model_(self, X, y):
        # Fit with the selected model, if not specified use ridge_regression
        if self.base_model == "least_squares_GD":
            w, _ = linear_models.least_squares_GD(y, X, self.initial_w, self.max_iters, self.gamma, self.tol)
        elif self.base_model == "least_squares_SGD":
            w, _ = linear_models.least_squares_SGD(y, X, self.initial_w, self.max_iters, self.gamma, self.tol)
        elif self.base_model == "least_squares":
            w, _ = linear_models.least_squares(y, X)
        elif self.base_model == "logistic_regression":
            w, _ = linear_models.logistic_regression(self._prepare_labels_(y),
                                                     X,
                                                     self.initial_w,
                                                     self.max_iters,
                                                     self.gamma,
                                                     self.tol)
            self.logistic = True
        elif self.base_model == "reg_logistic_regression":
            w, _ = linear_models.reg_logistic_regression(self._prepare_labels_(y),
                                                         X,
                                                         self.ridge_lambda,
                                                         self.initial_w,
                                                         self.max_iters,
                                                         self.gamma,
                                                         self.tol)
            self.logistic = True
        elif self.base_model == "ridge_regression":
            w, _ = linear_models.ridge_regression(y, X, self.ridge_lambda)
        else:
            raise ValueError("no model in store, sorry. you tried: {}".format(self.base_model))

        return w


    def fit(self, X, y, column_names=None):
        logging.info('fitting with: {}'.format(self._model_parameters_()))
        with_test_set = self.test_split_ratio > 0

        train_idx = pp.train_test_split(X, y, test_ratio = self.test_split_ratio)
        if with_test_set:
            test_idx = ~train_idx

        X_train = self._prepare_data_(X[train_idx], fit=True)
        if with_test_set:
            X_test = self._prepare_data_(X[test_idx], fit=False)

        y_train = y[train_idx]
        if with_test_set:
            y_test  = y[test_idx]

        if self.with_oversampling:
            # we only oversample the training set as we want the test
            # to reflect the real data as much as possible.
            # a different scoring method than accuracy would be more fitting for this task, e.g. F1-score
            oversampled_train_idx = pp.oversample(X_train, y_train)
            X_train = X_train[oversampled_train_idx]
            y_train = y_train[oversampled_train_idx]

        if self.with_feature_selection:
            coeffs, _ = linear_models.lasso_shooting(y_train, X_train, lambda_=self.lasso_lambda, max_iters=self.max_iters, tol=self.tol)
            self.feature_selection = np.abs(coeffs) > self.feature_selection_threshold

            self.column_names = np.array(self.column_names)[self.feature_selection]

            logging.debug("feature selection through lasso picked {}% of all features".format(self.feature_selection.mean()))

            X_train = X_train[:, self.feature_selection]
            if with_test_set:
                X_test = X_test[:, self.feature_selection]

        self.model = self._fit_model_(X_train, y_train)


        # Evalute predictions
        self.model_evaluation = {'train': (predict_labels(X_train, self.model) == y_train).mean(),
                                 'test': (predict_labels(X_test, self.model) == y_test).mean() if with_test_set else 'NO TEST SET SELECTED'}

        return self


    def predict(self, X):
        preds = predict_labels(self._prepare_data_(X, fit=False)[:, self.feature_selection], self.model, self.logistic)

        return preds

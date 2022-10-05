import logging

import numpy as np

from .LinearClassifier import LinearClassifier


class MultiClassifierModel:
    def __init__(self, base_classifier=None, jets_column=22, **submodel_parameters):
        self.models = {}

        if base_classifier is None:
            self.base_classifier = LinearClassifier
        else:
            self.base_classifier = base_classifier

        self.jets_column = jets_column
        self.submodel_parameters = {"jets_column": jets_column, **submodel_parameters}
        self.jets = None


    @property
    def model_evaluation(self):
        # just an extraction and combination of the results, they should be balanced for the size of jet-splits
        a =[m.model_evaluation for m in self.models.values()]
        me = dict(zip(['test', 'train'], np.array([list(me.values()) for me in a]).mean(axis=0)))
        me['submodels'] = a

        return me


    def weighted_score(self, X):
        """Use the same X you used for training. I know, not optimal..."""
        def _wa_(scores):
            return np.sum(np.array(scores) * np.array([np.sum(idxs)/len(X) for _, idxs in self._jet_indices_(X)]))

        test_scores, train_scores = zip(*(me.values() for me in self.model_evaluation['submodels']))

        return {'test': _wa_(test_scores), 'train': _wa_(train_scores)}


    def _jet_indices_(self, X, fit=False):
        if fit:
            self.jets = np.unique(X[:, self.jets_column])

        jet_indices = [(i, X[:, self.jets_column] == i) for i in self.jets]

        return jet_indices


    def _prepare_data_(self, X):
        X = X.copy()
        if self.submodel_parameters.get("merge_jet_2_3", False):
            X[X[:, self.jets_column] == 3, self.jets_column] = 2

        return X


    def fit(self, X, y):
        # since we use the jets column to build our multi-model model we remove it
        jet_indices = self._jet_indices_(X, fit=True)

        for jet, jet_index in jet_indices:
            # todo add params
            self.models[jet] = self.base_classifier(**self.submodel_parameters)\
                .fit(self._prepare_data_(X[jet_index, :]), y[jet_index])

        return self


    def predict(self, X, X_idx=None):
        if X_idx is None:
            X_idx = list(range(len(X)))

        jet_indices = self._jet_indices_(X, fit=False)

        preds, indicies = zip(*[(self.models[jet].predict(self._prepare_data_(X[jet_idx])), X_idx[jet_idx]) for jet, jet_idx in jet_indices])

        hulk_smash = lambda xs: np.hstack([x.reshape(-1) for x in xs])

        return hulk_smash(preds), hulk_smash(indicies)

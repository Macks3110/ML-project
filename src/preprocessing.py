# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

from itertools import combinations
from utils.misc import without, flatten


# <codecell>

_UNDEFINED_COLUMNS_FOR_JETS_ = [
    {
        'column_name': 'DER_deltaeta_jet_jet',
        'column_position': 4,
        'jets': [2, 3]
    },
    {
       'column_name': 'DER_mass_jet_jet',
       'column_position': 5,
       'jets': [2, 3]
    },
    {
       'column_name': 'DER_prodeta_jet_jet',
       'column_position': 6,
       'jets': [2, 3]
    },
    {
       'column_name': 'DER_lep_eta_centrality',
       'column_position': 12,
       'jets': [2, 3]
    },
    {
        'column_name': 'PRI_jet_subleading_pt',
        'column_position': 26,
        'jets': [2, 3]
    },
    {
         'column_name': 'PRI_jet_subleading_eta',
        'column_position': 27,
        'jets': [2, 3]
    },
    {
        'column_name': 'PRI_jet_subleading_phi',
        'column_position': 28,
        'jets': [2, 3]
    },
    {
        'column_name': 'PRI_jet_leading_pt',
        'column_position': 23,
        'jets': [1, 2, 3]
    },
    {
        'column_name': 'PRI_jet_leading_eta',
        'column_position': 24,
        'jets': [1, 2, 3]
    },
    {
        'column_name': 'PRI_jet_leading_phi',
        'column_position': 25,
        'jets': [1, 2, 3]
    },
]

# <codecell>

if __name__ == '__main__':
    from utils.jupyter import fix_layout
    fix_layout()

# <codecell>

import numpy as np

def standardize(x, mean=None, std=None):
    """Column based normalisation to mean==0, std==1
    or if mean and std is not None apply the transformations using the given values
    """
    if x.shape[0] == 0:
        raise ValueError("You want to standardize an empty matrix?")

    if mean is None:
        mean = np.mean(x, axis=0)

    x = x - mean

    if std is None:
        std = np.std(x, axis=0)

    idx = std > 0
    # if the data is too much standard...
    if idx.sum() > 0:
        x[:, idx] = x[:, idx] / std[idx]
    return x, mean, std

# <codecell>

def dummy_encoding(X, categorical_column):
    return np.hstack(((X[:, categorical_column] == v).astype(np.float).reshape(X.shape[0], 1)
                     for v in np.unique(X[:, categorical_column])))


# <codecell>

def train_test_split(X, y, test_ratio=0.33, use_indices=True):
    """
    Parameters:
        X              data
        y              labels
        test_ratio     percantage of data used for the test split
        use_indices    if True the return value will be a single boolean index. True corresponding to the training set
    """
    split_assignment = np.random.rand(X.shape[0]) > test_ratio

    if use_indices:
        return split_assignment
    else:
        X_train = X[split_assignment]
        y_train = y[split_assignment]
        X_test =  X[~split_assignment]
        y_test = y[~split_assignment]

        return X_train, y_train, X_test, y_test

# <codecell>

def _add_more_(els, nb):
    return np.hstack((els, np.random.choice(els, nb)))

# <codecell>

def oversample(X, y):
    """Simple oversampling method. See below for an example.

    >>> X = np.random.rand(10, 1)
    >>> y = np.random.rand(10) > 0.3
    >>>
    >>> train_idx = train_test_split(X, y)
    >>>
    >>> y_train = y[train_idx]
    >>> y_test = y[~train_idx]
    >>>
    >>> print('TRAIN')
    >>> print('we have {0} pos examples out of {1}'.format(y_train.sum(), y_train.shape[0]))
    >>> oversampled_train_idx = oversample(X[train_idx], y[train_idx])
    >>> X_train = X[train_idx][oversampled_train_idx]
    >>> y_train = y[train_idx][oversampled_train_idx]
    >>> print('we have {0} pos examples out of {1}'.format(y_train.sum(), y_train.shape[0]))
    >>>
    >>> print('TEST')
    >>> print('we have {0} pos examples out of {1}'.format(y_test.sum(), y_test.shape[0]))
    >>> oversampled_train_idx = oversample(X[~train_idx], y[~train_idx])
    >>> X_test = X[~train_idx][oversampled_train_idx]
    >>> y_test = y[~train_idx][oversampled_train_idx]
    >>> print('we have {0} pos examples out of {1}'.format(y_test.sum(), y_test.shape[0]))
    """

    classes = np.unique(y)
    max_nb_elements = max((y == c).sum() for c in classes)

    class_assignments = {c: ((y == c).sum(), np.where(y == c)[0]) for c in classes}
    class_assignments = {c: _add_more_(c_idx, max_nb_elements - c_amount) if c_amount < max_nb_elements else c_idx
                         for c, (c_amount, c_idx) in class_assignments.items()}

    return np.random.permutation(np.hstack(class_assignments.values()))

# <codecell>

def handle_undefined_values_for_jets(X, jet_column, undefined_columns_for_jets=_UNDEFINED_COLUMNS_FOR_JETS_):
    """ Replaces the undefined values with 0.

    Some columns are undefined for a given 'jet'. See `_UNDEFINED_COLUMNS_FOR_JETS_` for a list of these columns.
    For these columns we replace those values with 0.
    """
    X = X.copy()

    for col in undefined_columns_for_jets:
        _, col_pos, defined_jets = col.values()

        X[:, col_pos] *= (np.isin(X[:, jet_column], defined_jets)).astype(np.float)

    return X

# <codecell>

def default_values_for_each_column(data, fn=np.mean, faulty_value=-999, column_discard_threshold=0.8):
    """Defines for each column a default value, which can then be used to replace the missing values.
    In this case we use it to replace the `-999` values with the mean of the
    column, but taking only in account the values which are not `-999`.
    """
    def _mean_(X, col):
        idx = X[:, col] != faulty_value
        if idx.mean() > column_discard_threshold:
            return fn(X[idx, col])
        else:
            return None

    return [(col, _mean_(data, col)) for col in range(data.shape[1])]

# <codecell>

def overwrite_null_values_with_mean(data, default_values, faulty_value=-999):
    data = data.copy()
    for col, val in filter(lambda x: x[1] is not None, default_values):
        data[data[:, col] == faulty_value, col] = val

    return data

# <codecell>

def select_columns_based_on_default_value(data, default_values):
    """Using the precomputed default values select only the columns for which we have a value"""
    return data[:, [col for col, m in default_values if m is not None]]


def highly_correlated_columns(X, threshold=0.995):
    # For some reason I can't compute this in one go on my machine... but alas I am an engineer!
    correlation_matrix = np.zeros((X.shape[1], X.shape[1]))

    for c1 in range(X.shape[1]):
        for c2 in range(X.shape[1]):
            if c1 != c2:
                correlation_matrix[c1, c2] = np.corrcoef(X[:, c1], X[:, c2])[0, 1]

    return list(set(np.where(np.abs(np.nan_to_num(correlation_matrix)) > threshold)[0]))


def _expand_column_names_(column_names, extension):
    return [c + extension for c in column_names]


def feature_base_expansion(X, base_columns=None):
    # the `1000` in the log transformation is just a random value... it comes from the given "NaN" number which is -999
    X_expanded= np.hstack((X,
                           np.log(X + 1000),
                           *(X ** i for i in range(2, 7)),
                           np.tanh(X),
                           *((X[:, c1] * X[:, c2]).reshape((X.shape[0], 1)) for c1, c2 in combinations(range(X.shape[1]), 2)),
                           *(np.abs(X[:, c1] - X[:, c2]).reshape((X.shape[0], 1)) for c1, c2 in combinations(range(X.shape[1]), 2)),
                         ))

    if base_columns is not None:
        return X_expanded, flatten([base_columns,
                                   _expand_column_names_(base_columns, '_log'),
                                   *(_expand_column_names_(base_columns, '_power_{}'.format(p)) for p in range(2, 7)),
                                   _expand_column_names_(base_columns, '_tanh'),
                                   ['product_{0}_{1}'.format(base_columns[c1], base_columns[c2]) for c1, c2 in combinations(range(X.shape[1]), 2)],
                                   ['abs_diff_{0}_{1}'.format(base_columns[c1], base_columns[c2]) for c1, c2 in combinations(range(X.shape[1]), 2)]])
    else:
        return X_expanded


# <codecell>

if __name__ == '__main__':
    X = np.random.rand(10, 1)
    y = np.random.rand(10) > 0.3

    train_idx = train_test_split(X, y)

    y_train = y[train_idx]
    y_test = y[~train_idx]

    print('TRAIN')
    print('we have {0} pos examples out of {1}'.format(y_train.sum(), y_train.shape[0]))
    oversampled_train_idx = oversample(X[train_idx], y[train_idx])
    X_train = X[train_idx][oversampled_train_idx]
    y_train = y[train_idx][oversampled_train_idx]
    print('we have {0} pos examples out of {1}'.format(y_train.sum(), y_train.shape[0]))

    print('TEST')
    print('we have {0} pos examples out of {1}'.format(y_test.sum(), y_test.shape[0]))
    oversampled_train_idx = oversample(X[~train_idx], y[~train_idx])
    X_test = X[~train_idx][oversampled_train_idx]
    y_test = y[~train_idx][oversampled_train_idx]
    print('we have {0} pos examples out of {1}'.format(y_test.sum(), y_test.shape[0]))

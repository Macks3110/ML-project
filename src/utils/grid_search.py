from functools import partial
from itertools import product
import logging

# Use cross validation to get an unbiased estimation

# reordering the arguments for partial to work
def _grid_search_fit_(clf, base_parameters, grid_search_parameters, X, y):
    try:
        return clf(**{**base_parameters, **grid_search_parameters}).fit(X, y)
    except Exception as e:
        logging.error("exception ! {0}, used arguments {1}".format(e, grid_search_parameters))
        raise
        return None



def grid_search(clf, base_parameters, grid_search_parameters, X, y):
    parameters = list(product(*grid_search_parameters.values()))
    print('testing {} models! make sure that you have the power to run this!'.format(len(parameters)))

    res = [_grid_search_fit_(clf, base_parameters, dict(zip(grid_search_parameters.keys(), p)), X, y)
           for p in parameters]
    return res


def grid_search_parallel(clf, base_parameters, grid_search_parameters, X, y):
    from functional import pseq

    parameters = list(product(*grid_search_parameters.values()))
    print('testing {} models! make sure that you have the power to run this!'.format(len(parameters)))

    return pseq(parameters)\
        .map(lambda p: dict(zip(grid_search_parameters.keys(), p)))\
        .map(lambda p: _grid_search_fit_(clf=clf, base_parameters=base_parameters,
                                         grid_search_parameters=p, X=X, y=y))\
        .list()

# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <markdowncell>

# # imports

# <markdowncell>

# Use this notebook to develop the model. See `run.py` for a running model. Use with care.

# <codecell>

def fix_layout(width:int=95):
    from IPython.core.display import display, HTML
    display(HTML('<style>.container { width:' + str(width) + '% !important; }</style>'))
    
fix_layout()

# <codecell>

import numpy as np

from itertools import combinations, product
    
from IPython.display import display

import preprocessing as pp
from functions import implementations as linear_models 
from functions.helpers import calculate_mse
# todo remove this from final run version
from utils.plots import plot_feature_importance
import utils.csv as ph

from utils.logs import enable_logging, logging
enable_logging(logging.DEBUG)

# do not use %pylab, that inserts packages into the namespace 
# %matplotlib inline

# <codecell>

np.random.seed(42) # for reproductive results

# <markdowncell>

# # data loading & handling
# 
# - loading data
# - clean data (replacing -999 with mean values)

# <codecell>

y, X, X_indices = ph.load_csv_data('../data/train.csv')

# <codecell>

with open('../data/train.csv', 'r') as f:
    csv_header = f.readline()

# <codecell>

X_columns = [c.strip() for c in csv_header.split(',')[2:]]

# <markdowncell>

# # model fitting
# 
# - feature expansion (before or after the standardisation?)
# - standardising
# - train/test split
# - feature selection
# - creating models for each jet
# - train each model

# <codecell>

import models.LinearClassifier as LC
from importlib import reload
reload(LC)

# <codecell>

m.remove_highly_correlated_features_treshold

# <codecell>

run_sanitiy_checks = True

base_parameters = {
    "merge_jet_2_3":  True, 
    "with_bias_column":  True,
    "with_feature_expansion":  False,
    "jets_column":  22,
    "remove_columns_with_many_undefined_values": True,
    "column_names": X_columns,
    "remove_highly_correlated_features_treshold": 0.96
}



if run_sanitiy_checks:
    sanity_simple_models = [LC.LinearClassifier(**{**base_parameters, **model}).fit(X, y) for model in linear_models._TEST_PARAMETERS_]

    for m in sanity_simple_models:
        print(m.base_model, m.model_evaluation)

# <codecell>

for m in sanity_simple_models:
    print(m.base_model, m.model_evaluation)

# <codecell>

if run_sanitiy_checks:
    clf_parameters = {"merge_jet_2_3": True, 
                      "jets_column":  22,
                      "with_bias_column": True,
                      "with_feature_expansion": True,
                      "with_feature_selection": True,
                      "lasso_lambda": 50,
                      "column_names": X_columns
                     }

    clf = LC.LinearClassifier(**clf_parameters).fit(X, y)

    print(clf.model_evaluation)
    plot_feature_importance(clf)

# <markdowncell>

# # Multi model

# <codecell>

from models import MultiClassifierModel as MCM
reload(LC)
reload(MCM)

# <codecell>

if run_sanitiy_checks:
    mclf = MCM.MultiClassifierModel(LC.LinearClassifier, **{**base_parameters, **{'with_oversampling': True}}).fit(X, y)
    print(mclf.model_evaluation['test'])

    mclf = MCM.MultiClassifierModel(LC.LinearClassifier, **base_parameters).fit(X, y)
    print(mclf.model_evaluation['test'])
    
    mclf = MCM.MultiClassifierModel(LC.LinearClassifier, **{**base_parameters, **{'remove_columns_with_many_undefined_values': False}}).fit(X, y)
    print(mclf.model_evaluation['test'])

# <codecell>

final_model_params = {'ridge_lambda': 1e-10, 'with_feature_expansion': True, 'remove_highly_correlated_features_treshold': 0.8, 'column_names': ['DER_mass_MMC', 'DER_mass_transverse_met_lep', 'DER_mass_vis', 'DER_pt_h', 'DER_deltaeta_jet_jet', 'DER_mass_jet_jet', 'DER_prodeta_jet_jet', 'DER_deltar_tau_lep', 'DER_pt_tot', 'DER_sum_pt', 'DER_pt_ratio_lep_tau', 'DER_met_phi_centrality', 'DER_lep_eta_centrality', 'PRI_tau_pt', 'PRI_tau_eta', 'PRI_tau_phi', 'PRI_lep_pt', 'PRI_lep_eta', 'PRI_lep_phi', 'PRI_met', 'PRI_met_phi', 'PRI_met_sumet', 'PRI_jet_num', 'PRI_jet_leading_pt', 'PRI_jet_leading_eta', 'PRI_jet_leading_phi', 'PRI_jet_subleading_pt', 'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi', 'PRI_jet_all_pt'], 'jets_column': 22, 'with_oversampling': False, 'lasso_lambda': 0.001, 'merge_jet_2_3': True, 'with_bias_column': True}

clf = MCM.MultiClassifierModel(**final_model_params).fit(X, y)

# <codecell>

np.mean([len(m.model) for m in clf.models.values()])

# <codecell>

import pandas as pd
feature_importances = pd.DataFrame(columns=['column'])
for jet, m in clf.models.items():
    feature_importance = list(zip(np.abs(m.model), m.column_names))
    feature_importances = feature_importances.merge(pd.DataFrame(sorted(feature_importance, key=lambda x: abs(x[0]), reverse=True), columns=['feature_importance_{}'.format(jet), 'column']), on='column', how='outer')

# <codecell>

feature_importances['mean'] = feature_importances[[c for c in feature_importances.columns if c != 'column']].mean(axis=1)
feature_importances = feature_importances.sort_values('mean', ascending=False)
feature_importances['mean_pct'] = np.cumsum(feature_importances['mean']) / feature_importances['mean'].sum()
plt_data = feature_importances[feature_importances['mean_pct'] < 0.5]

# <codecell>

import seaborn as sns

sns.set_palette(sns.color_palette("BuGn_r"))

# <codecell>

from matplotlib import pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
ax = sns.barplot(x='mean', y='column', data=plt_data, palette=sns.color_palette("BuGn_r", n_colors=plt_data.shape[0]))
ax.set(xlabel='mean of weight', ylabel='feature')
plt.tight_layout()
ax.figure.savefig('../report/plots/feature_importance.eps')

# <codecell>

def plot_feature_importance_multi(clf):


# <markdowncell>

# # Grid Search

# <codecell>

run_sanitiy_checks = False
if run_sanitiy_checks:
    grid_search_parameters = {
        'lasso_lambda': np.logspace(1, 3, 1),
        'ridge_lambda': np.logspace(-10, -3, 2)
    }
else:
    grid_search_parameters= {
        'lasso_lambda': np.logspace(-3, 3, 10),
        'ridge_lambda': np.logspace(-10, 1, 10),
        'merge_jet_2_3': [True, False],
        'remove_highly_correlated_features_treshold': np.linspace(0.8, 0.9, 2)
    }

base_parameters = {
    #"merge_jet_2_3": True, 
    "with_bias_column": True,
    "with_feature_expansion": True,
    "jets_column":  22,
    "column_names": X_columns,
    "with_oversampling": False,
}

# <codecell>

import utils.grid_search as gs
from importlib import reload
reload(gs)
reload(MCM)

grid_search_results = gs.grid_search_parallel(MCM.MultiClassifierModel, base_parameters=base_parameters, 
                     grid_search_parameters=grid_search_parameters,
                     X=X,
                     y=y)

# <codecell>

gsr_dict = [{'model_evaluation': m.model_evaluation, 'parameters': m.submodel_parameters, 'weighted_score': weighted_score(m, X)} for m in grid_search_results]

# <codecell>

gsr_dict

# <codecell>

import json
with open('./grid_search_results.json', 'w') as outfile:
    json.dump(gsr_dict, outfile)

# <codecell>

def weighted_score(mcm, X):
    def _wa_(scores):
        return np.sum(np.array(scores) * np.array([np.sum(idxs)/len(X) for _, idxs in mcm._jet_indices_(X)]))
    
    test_scores, train_scores = zip(*(me.values() for me in best_clf.model_evaluation['submodels']))
    
    return {'test': _wa_(test_scores), 'train': _wa_(train_scores)}

# <codecell>

max_idx = np.argmax(np.array([weighted_score(m, X)['test'] for m in grid_search_results]))
best_clf = grid_search_results[max_idx]

print(best_clf.submodel_parameters)
print(best_clf.model_evaluation)
print(weighted_score(best_clf, X))

# <codecell>

another_one = MCM.MultiClassifierModel(**best_clf.submodel_parameters)
another_one.submodel_parameters['with_oversampling'] = True
another_one.fit(X, y)


print(another_one.model_evaluation)
print(weighted_score(another_one, X))

# <markdowncell>

# # Predicting

# <codecell>

_, X_pred, X_pred_indices = ph.load_csv_data('../data/test.csv')

# <codecell>

from datetime import datetime

preds, preds_idx = best_clf.predict(X_pred, X_pred_indices)
ph.create_csv_submission(preds_idx, preds, '../submissions/' + datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + '.csv')

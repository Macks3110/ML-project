# -*- coding: utf-8 -*-

# This is a run down version of `main.ipynb`. If you want find more
# information, comments and such, please check out that file.

# Please ignore the RuntimeWarning. It's being dealt with.

import numpy as np

from utils.csv import load_csv_data, create_csv_submission
from models import MultiClassifierModel as MCM


def main():
    np.random.seed(42) # for reproductive results

    y, X, X_indices = load_csv_data('../data/train.csv')

    model_parameters = {'ridge_lambda': 1e-10, 'with_feature_expansion': True,
                        'remove_highly_correlated_features_treshold': 0.8,
                        'column_names': ['DER_mass_MMC',
                                         'DER_mass_transverse_met_lep',
                                         'DER_mass_vis', 'DER_pt_h',
                                         'DER_deltaeta_jet_jet',
                                         'DER_mass_jet_jet', 'DER_prodeta_jet_jet',
                                         'DER_deltar_tau_lep', 'DER_pt_tot',
                                         'DER_sum_pt', 'DER_pt_ratio_lep_tau',
                                         'DER_met_phi_centrality',
                                         'DER_lep_eta_centrality', 'PRI_tau_pt',
                                         'PRI_tau_eta', 'PRI_tau_phi',
                                         'PRI_lep_pt', 'PRI_lep_eta',
                                         'PRI_lep_phi', 'PRI_met', 'PRI_met_phi',
                                         'PRI_met_sumet', 'PRI_jet_num',
                                         'PRI_jet_leading_pt',
                                         'PRI_jet_leading_eta',
                                         'PRI_jet_leading_phi',
                                         'PRI_jet_subleading_pt',
                                         'PRI_jet_subleading_eta',
                                         'PRI_jet_subleading_phi',
                                         'PRI_jet_all_pt'], 'jets_column': 22,
                        'with_oversampling': False, 'lasso_lambda': 0.001,
                        'merge_jet_2_3': True, 'with_bias_column': True}

    clf = MCM.MultiClassifierModel(**model_parameters).fit(X, y)

    _, X_pred, X_pred_indices = load_csv_data('../data/test.csv')

    preds, preds_idx = clf.predict(X_pred, X_pred_indices)
    create_csv_submission(preds_idx, preds, '../submissions/final_prediction.csv')


if __name__ == '__main__':
    main()

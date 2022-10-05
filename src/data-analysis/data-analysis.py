# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <markdowncell>

# # Setup

# <markdowncell>

# ## Imports

# <codecell>

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas_profiling
import numpy as np

import sklearn

from IPython.display import display
import pandas_profiling
# do not use %pylab, that inserts packages into the namespace 
%matplotlib inline

# <markdowncell>

# ## Helper methods

# <codecell>

def fix_layout(width:int=95):
    from IPython.core.display import display, HTML
    display(HTML('<style>.container { width:' + str(width) + '% !important; }</style>'))
    
fix_layout()


def group_and_count(df, groupby_column, with_pct=False, with_avg=False):
    result = df.groupby(groupby_column).size().sort_values(ascending=False).reset_index().rename(columns={0: 'count'})
    if with_pct:
        result['count_pct'] = result['count'] / result['count'].sum()
    if with_avg:
        result['count_avg'] = result['count'].mean()
    return result

# <codecell>

from operator import itemgetter 

def plot_feature_importance(feature_importances,
                            column_names,
                            top=None,
                            min_imp_coverage=None):
    
    feature_importance = sorted(zip(column_names, feature_importances), key=itemgetter(1), reverse=True)
    nb_feats = len(feature_importance)
    plt.figure(figsize=(5, np.ceil(nb_feats * 0.35)))
    plt.barh(-1 * np.arange(nb_feats), [item[1] for item in feature_importance]) #, color=bar_colours)
    plt.yticks(-1 * np.arange(nb_feats) + 0.5, [item[0] for item in feature_importance], rotation=0)
    plt.ylim(ymin=-nb_feats - 1, ymax=1)

    title_str = 'Feature Importances'
    if top is not None:
        title_str = title_str + ' of top {0:d} features'.format(top)
    if min_imp_coverage is not None:
        title_str = title_str + ' of features with combined importance coverage of >= {0:.0%}'.format(min_imp_coverage)
    plt.title(title_str)

    return feature_importance

# <codecell>

from sklearn.metrics import confusion_matrix
    
def plot_confusion_matrix(y_true, y_pred, class_labels=None, title='', cmap=plt.cm.Blues):

    class_labels = class_labels or [0,1]

    cm = confusion_matrix(y_true, y_pred)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    ax.set_xticks(np.arange(cm.shape[0]))
    ax.set_yticks(np.arange(cm.shape[1]))
    ax.set_xticklabels(list(class_labels), rotation=0)
    ax.set_yticklabels(list(class_labels))
    plt.tight_layout()
    plt.xlabel('Predicted class')
    plt.ylabel('True class')
    plt.title(title)
    ax.tick_params(axis='both', which='both', length=0)

    width, height = cm.shape

    colour_threshold = np.amax(cm) / 2

    for x in range(width):
        for y in range(height):
            if cm[x][y] > 0:
                if cm[x][y] > colour_threshold:
                    color = 'white'
                else:
                    color = 'black'

                ax.text(y,
                        x,
                        str(cm[x][y]),
                        verticalalignment='center',
                        horizontalalignment='center',
                        color=color,
                        fontsize=15)

# <codecell>

def distribution_comparison_plot(df, compare_col, idx, idx_label=None, cap_quantile=None, floor_quantile=None):
    def remove_nans(values):
        idx = np.isnan(values)
        return values[~idx], idx.sum()

    [floor, cap] = np.percentile(df.loc[df[compare_col].notnull(), compare_col],
                                 [floor_quantile or 0, cap_quantile or 100])
    idx_floor_cap = (df[compare_col] >= floor) & (df[compare_col] <= cap)

    values_idx_true, nb_nans_idx_true = remove_nans(df.loc[idx & idx_floor_cap, compare_col])
    values_idx_false, nb_nans_idx_false = remove_nans(df.loc[~idx & idx_floor_cap, compare_col])

    plt.figure()
    if len(np.unique(values_idx_true)) > 1:
        sns.distplot(values_idx_true, label=(idx_label or 'Index') + ': True')
    if len(np.unique(values_idx_false)) > 1:
        sns.distplot(values_idx_false, label=(idx_label or 'Index') + ': False')
    plt.legend()
    plt.title('distribution comparison plot\n(removed {0} nan from the "True" dist and {1} from the "False" dist)'
              .format(nb_nans_idx_true, nb_nans_idx_false))

# <codecell>

from sklearn.metrics import roc_auc_score, cohen_kappa_score, f1_score, accuracy_score, precision_score, recall_score,\
    log_loss

def display_metrics(y_train, y_pred_train, y_test, y_pred_test):
    scores_df = pd.DataFrame()

    my_metrics = [('f1', f1_score),
                  ('precision', precision_score),
                  ('recall', recall_score),
                  ("cohen's kappa", cohen_kappa_score),
                  ('accuracy', accuracy_score)]

    my_metrics_w_proba = [('log_loss', log_loss),
                          ('auc', roc_auc_score)]

    for cur_metric_name, cur_metric in my_metrics:
        scores_df.loc['train', cur_metric_name] = cur_metric(y_train, y_pred_train)
        scores_df.loc['test', cur_metric_name] = cur_metric(y_test, y_pred_test)

    return scores_df

# <markdowncell>

# ## Reading in data

# <codecell>

train_data = pd.read_csv('../../data/train.csv')

# <codecell>

train_data.groupby('Prediction').size() / train_data.shape[0]

# <markdowncell>

# # Data Analysis

# <codecell>

profile_report = pandas_profiling.ProfileReport(train_data)
profile_report

# <codecell>

train

# <codecell>

profile_report.get_rejected_variables()

# <markdowncell>

# # ML

# <codecell>

from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
import eli5

# <codecell>

train_data.columns

# <markdowncell>

# ## Simple try

# <codecell>

columns_to_ignore = ['Id', 'Prediction']

X = train_data[[c for c in train_data.columns if c not in columns_to_ignore]]
y = train_data['Prediction'].apply(lambda x: 1 if x == 'b' else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# <codecell>

print("we have a total of {nb_samples} elements to train on, out of these {nb_pos_samples}({np_pos_samples_percentage:.1%}%) are positive"\
      .format(nb_samples=y_train.shape[0],
              nb_pos_samples=y_train.sum(),
              np_pos_samples_percentage=y_train.mean()))

# <codecell>

clf = RandomForestClassifier()

clf.fit(X_train, y_train)

y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)

plot_feature_importance(feature_importances=clf.feature_importances_, column_names=X.columns);

plot_confusion_matrix(y_true=y_train, y_pred=y_pred_train, title='train')
plot_confusion_matrix(y_true=y_test, y_pred=y_pred_test, title='test')

display_metrics(y_test=y_test, y_pred_test=y_pred_test, y_train=y_train, y_pred_train=y_pred_train)

# => massive overfitting....

# <codecell>

columns_to_ignore = ['Id', 'Prediction'] + profile_report.get_rejected_variables()

X = train_data[[c for c in train_data.columns if c not in columns_to_ignore]]
y = train_data['Prediction'].apply(lambda x: 1 if x == 'b' else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = RandomForestClassifier()

clf.fit(X_train, y_train)

y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)

plot_feature_importance(feature_importances=clf.feature_importances_, column_names=X.columns);

plot_confusion_matrix(y_true=y_train, y_pred=y_pred_train, title='train')
plot_confusion_matrix(y_true=y_test, y_pred=y_pred_test, title='test')

display_metrics(y_test=y_test, y_pred_test=y_pred_test, y_train=y_train, y_pred_train=y_pred_train)

# => massive overfitting....

# <codecell>

columns_to_ignore = ['Id', 'Prediction'] + profile_report.get_rejected_variables()

X = train_data[[c for c in train_data.columns if c not in columns_to_ignore]]
y = train_data['Prediction'].apply(lambda x: 1 if x == 'b' else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = RandomForestClassifier(n_jobs=-1, n_estimators=200, max_depth=3, min_samples_split=10)

clf.fit(X_train, y_train)

y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)

plot_feature_importance(feature_importances=clf.feature_importances_, column_names=X.columns);

plot_confusion_matrix(y_true=y_train, y_pred=y_pred_train, title='train')
plot_confusion_matrix(y_true=y_test, y_pred=y_pred_test, title='test')

display_metrics(y_test=y_test, y_pred_test=y_pred_test, y_train=y_train, y_pred_train=y_pred_train)

# <codecell>

columns_to_ignore = ['Id', 'Prediction']

X = train_data[[c for c in train_data.columns if c not in columns_to_ignore]]
y = train_data['Prediction'].apply(lambda x: 1 if x == 'b' else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = RandomForestClassifier(n_jobs=-1, n_estimators=200, max_depth=3)

clf.fit(X_train, y_train)

y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)

plot_feature_importance(feature_importances=clf.feature_importances_, column_names=X.columns);

plot_confusion_matrix(y_true=y_train, y_pred=y_pred_train, title='train')
plot_confusion_matrix(y_true=y_test, y_pred=y_pred_test, title='test')

display_metrics(y_test=y_test, y_pred_test=y_pred_test, y_train=y_train, y_pred_train=y_pred_train)

# <markdowncell>

# ## Better try

# <codecell>

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

columns_to_ignore = ['Id', 'Prediction'] + profile_report.get_rejected_variables()

X = train_data[[c for c in train_data.columns if c not in columns_to_ignore]]
y = train_data['Prediction'].apply(lambda x: 1 if x == 'b' else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

params = [{
            'n_estimators': [10, 100, 200],
            'max_depth': [3,4,5],
          }]

scores = ['accuracy', 'precision', 'recall']

clf = GridSearchCV(RandomForestClassifier(), 
                   params,
                   cv=2, # we have enough data points
                   n_jobs=-1,
                   scoring='accuracy')
clf.fit(X_train, y_train)

print("Best parameters set found on development set:")
print(clf.best_params_)
print("Grid scores on development set:")

means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()

print("Detailed classification report:")
y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))

# <codecell>

plot_feature_importance(feature_importances=clf.feature_importances_, column_names=X.columns);

plot_confusion_matrix(y_true=y_train, y_pred=y_pred_train, title='train')
plot_confusion_matrix(y_true=y_test, y_pred=y_pred_test, title='test')

display_metrics(y_test=y_test, y_pred_test=y_pred_test, y_train=y_train, y_pred_train=y_pred_train)

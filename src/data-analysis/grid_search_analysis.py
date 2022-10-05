# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <codecell>

import json
import pandas as pd
from pandas.io.json import json_normalize    

# <codecell>

with open('../../submissions/grid_search_results.json', 'r') as f:
    grid_search_results = json.load(f)

# <codecell>

df = json_normalize(grid_search_results)
df

# <codecell>

import seaborn as sns
sns.set(style="darkgrid")

# <codecell>

sns.lineplot(x="parameters.lasso_lambda", y="weighted_score.test",
             data=df)

# <codecell>

sns.regplot(x="parameters.lasso_lambda", y="weighted_score.test",
             data=df)

# <codecell>

import matplotlib.pyplot as plt

# <codecell>

plt.plot(df['parameters.lasso_lambda'], df['weighted_score.test'])

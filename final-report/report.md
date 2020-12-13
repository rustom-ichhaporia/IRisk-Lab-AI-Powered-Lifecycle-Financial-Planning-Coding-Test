---
title: AI-Powered Socioeconomic Prediction of Lifespan
author: 
    - Rustom Ichhaporia [`rustomi2@illinois.edu`][^*]
date: 2020-12-12
description: Final report for AI-Powered Lifecycle Financial Project research. 
abstract: How long will you live? This age-old question has extensive implications in the billions of risk estimations made by individuals planning for the future every day. Although never certain, a stronger approximation of an indidual's lifespan can enable more reliable future planning and  greater sense of stability

# geometry: margin=1in
documentclass: article
classoption: 
    - twocolumn
numbersections: true
appendix: jfa;slkdfj
---

<!-- Document -->

# Background



# Dataset Selection

# Preprocessing 

The appendix of this report contains the code for training the model and saving the results in a file. It does not include the code for statistical plots. 

# Modeling

# Results

Last week, I registered for HAL access so that I could run the hyperparameter optimization script remotely because my computer overheated and could not run it for the appropriate number of trials. I was able to upload my files and run the first half of the script, but unfortunately when running the `fmin` optimization function, the program crashes after the first of 150 loops with the error: 


## Limitations 

[^1]
One significant drawback of the approach taken to estimating 

\newpage
# Appendix

<!-- 
<style>
  .col1 {
    columns: 1;
  }
</style>

<div class="col1">

#### **`script.py`** {-}

```{.python .numberLines}
'''Imports'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import plot_roc_curve

from imblearn.over_sampling import SMOTE

sns.set()

'''Preprocessing'''

df_raw = pd.read_csv('data/11.csv')
print('Data successfully loaded.')

df_raw = df_raw.drop(columns=['smok100', 'agesmk', 'smokstat', 'smokhome', 'curruse', 'everuse'])

df_raw['indmort'] = df_raw['inddea'][(df_raw['inddea'] == 1) & (df_raw['indalg'] == 1)]
df_raw['indmort'] = df_raw['indmort'].fillna(0)

used_numerical = ['age', 'hhnum']
used_ordinal = ['povpct', 'adjinc']
used_categorical = ['stater', 'pob', 'sex', 'race', 'urban', 'smsast']
used_special = ['wt', 'indmort']

used_features = used_numerical + used_ordinal + used_categorical + used_special

df_raw = df_raw[used_features]

df_raw[used_categorical] = df_raw[used_categorical].astype('category')

df_raw = df_raw.dropna(axis=0)

df = pd.get_dummies(df_raw)

X = df.drop(columns=['indmort'])
y = df['indmort']

'''Sampling'''

X_train, X_test, y_train, y_test = train_test_split(X, y)

print('Proportion of data from minority class before SMOTE:', y_train.sum() / y_train.shape[0])
X_train, y_train = SMOTE().fit_resample(X_train, y_train)
print('Proportion of data from minority class after SMOTE:', y_train.sum() / y_train.shape[0])

'''Modeling'''

model = LogisticRegressionCV(scoring='roc_auc', random_state=0, n_jobs=-1, verbose=1).fit(X_train.drop(columns=['wt']), y_train, sample_weight=X_train['wt'])

print(classification_report(model.predict(X_test.drop(columns=['wt'])), y_test))

pred_probs = model.predict_proba(X_test.drop(columns=['wt']))[:, 1]

print(classification_report(np.round(pred_probs + 0.25), y_test, sample_weight=X_test['wt']))

```
</div> -->


<!-- Footnotes -->

[^*]: This research project was completed during my time as a research intern at the Illinois Risk Lab (https://irisklabuiuc.wixsite.com/) during the Fall of 2020. My research was a part of the AI-Powered Lifecycle Financial Planning project, which is still under development. I appreciate the help of Dr. Runhuan Feng, Dr. Frank Quan, Dr. Yong Xie, Dr. Linfeng Zhang, and my fellow interns throughout the process. Thank you!

[^1]: https://github.com/microsoft/LightGBM/issues/2696
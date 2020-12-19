'''Imports'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

from imblearn.over_sampling import SMOTE

sns.set()

'''Preprocessing'''

df_raw = pd.read_csv('data/11.csv')
print('Data successfully loaded.')

# Drop empty smoking-related columns
df_raw = df_raw.drop(columns=['smok100', 'agesmk', 'smokstat', 'smokhome', 'curruse', 'everuse'])

# Combine mortality columns as specified by reference guide
df_raw['indmort'] = df_raw['inddea'][(df_raw['inddea'] == 1) & (df_raw['indalg'] == 1)]
df_raw['indmort'] = df_raw['indmort'].fillna(0)

# Specify which variables to use in the model by type
used_numerical = ['age', 'hhnum']
used_ordinal = ['povpct', 'adjinc']
used_categorical = ['stater', 'pob', 'sex', 'race', 'urban', 'smsast']
used_special = ['wt', 'indmort']

used_features = used_numerical + used_ordinal + used_categorical + used_special

df_raw = df_raw[used_features]

# Correct datatypes of categorical variables
df_raw[used_categorical] = df_raw[used_categorical].astype('category')

# Drop rows with remaining missing values
df_raw = df_raw.dropna(axis=0)

# Dummify categorical variables
df = pd.get_dummies(df_raw)

# Split data into predictive features and target array
X = df.drop(columns=['indmort'])
y = df['indmort']

'''Sampling'''

# Create test dataset for validation
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Apply Synthetic Minority Oversampling Technique to data
print('Proportion of data from minority class before SMOTE:', y_train.sum() / y_train.shape[0])
X_train, y_train = SMOTE().fit_resample(X_train, y_train)
print('Proportion of data from minority class after SMOTE:', y_train.sum() / y_train.shape[0])

'''Modeling'''

# Train logistic regression model with cross validation
model = LogisticRegressionCV(scoring='roc_auc', random_state=0, n_jobs=-1, verbose=1).fit(X_train.drop(columns=['wt']), y_train, sample_weight=X_train['wt'])

# Generate predictions
pred_probs = model.predict_proba(X_test.drop(columns=['wt']))[:, 1]

# Print model outputs
print(classification_report(pred_probs, y_test))

# The predictions are best when a constant is added to the final probabilities
print(classification_report(np.round(pred_probs + 0.25), y_test, sample_weight=X_test['wt']))
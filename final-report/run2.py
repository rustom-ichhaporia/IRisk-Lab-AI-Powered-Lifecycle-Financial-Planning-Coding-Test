# Mortality Group Weekly Report - 11/7/2020
# Rustom Ichhaporia

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import tree
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
# from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.utils import resample
from sklearn.metrics import auc
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import csv
from hyperopt import STATUS_OK, hp, tpe, Trials, fmin
from hyperopt.pyll.stochastic import sample
from timeit import default_timer as timer
import pickle
sns.set()


# In[2]:


df_raw = pd.read_csv('data/11.csv').drop(['smok100', 'agesmk', 'smokstat', 'smokhome', 'curruse', 'everuse'], axis=1)
df_raw


# Because this data takes a long time to compile and verify from multiple smaller studies, the creators included both a definite and an algorithmic indicator of death to speed up the release of the data. These features, `'inddea'` and `'indalg'`, respectively, have been combined through intersection into one target variable, `'indmort'` in the way that the reference manual recommends. The expanded list of features is below. 

# In[3]:


numerical = ['age', 'hhnum']
uneven_numerical = ['adjinc', 'health', 'follow']
categorical = ['race', 'sex', 'ms', 'hisp', 'educ', 'pob', 'hhid', 'reltrf', 'occ', 'majocc', 'ind', 'esr', 'urban', 'smsast', 'inddea', 'cause113', 'dayod', 'hosp', 'hospd', 'ssnyn', 'vt', 'histatus', 'hitype', 'povpct', 'stater', 'rcow', 'tenure', 'citizen', 'indalg']
smoking = ['smok100', 'agesmk', 'smokstat', 'smokhome', 'curruse', 'everuse']
misc = ['record', 'wt']


## In[4]:


# indmort is the recommended combination feature of both confirmed deaths and computer-predicted deaths based on the data collection agency
df_raw['indmort'] = df_raw['inddea'][(df_raw['inddea'] == 1) & (df_raw['indalg'] == 1)]
df_raw['indmort'] = df_raw['indmort'].fillna(0)
df_raw['indmort'].sum()


# In[5]:


# Selection of fewer variables for EDA purposes
# Remove cause113 because it is not predictive
used_features = ['age', 'hhnum', 'adjinc', 'health', 'occ', 'ind', 'esr', 'ms', 'indmort']
df = df_raw[used_features]


# In[6]:


df.isna().sum() / df.shape[0]


# In[7]:


mort_corr = df.corr()['indmort'].sort_values()
mort_corr


# Correlations can be useful for numerical features, but most of these features are categorical, so I decided to begin one-hot encoding and imputation of missing values. Most of the health rating entries are still missing. 

# In[8]:


df = df.astype({'occ':'category', 'ind': 'category', 'esr': 'category', 'ms': 'category'})
df.dtypes


df_raw['citizen'].value_counts(dropna=False)


# In[12]:


df_raw['hosp'].value_counts(dropna=False)


# In[13]:


(df_raw.isna().sum() / df_raw.shape[0]).sort_values()


# In[14]:


# indmort is the recommended combination feature of both confirmed deaths and computer-predicted deaths based on the data collection agency
df_raw['indmort'] = df_raw['inddea'][(df_raw['inddea'] == 1) & (df_raw['indalg'] == 1)]
df_raw['indmort'] = df_raw['indmort'].fillna(0)
df_raw['indmort'].sum()


# In[15]:


numerical = ['age', 'hhnum', 'povpct']
uneven_numerical = ['adjinc', 'health', 'follow']
categorical = ['race', 'sex', 'ms', 'hisp', 'educ', 'pob', 'hhid', 'reltrf', 'occ', 'majocc', 'ind', 'esr', 'urban', 'smsast', 'inddea', 'cause113', 'dayod', 'hosp', 'hospd', 'ssnyn', 'vt', 'histatus', 'hitype', 'povpct', 'stater', 'rcow', 'tenure', 'citizen', 'indalg']
smoking = ['smok100', 'agesmk', 'smokstat', 'smokhome', 'curruse', 'everuse']
misc = ['record', 'wt']

used_features = ['age', 'sex', 'stater', 'povpct', 'pob', 'race', 'urban', 'ms', 'adjinc', 'educ', 'indmort', 'wt']#, 'educ', 'stater']#, 'wt']
df = df_raw[used_features]
df.dtypes


# In[16]:


df = df.astype({'sex':'category', 'stater': 'category', 'pob': 'category', 'race': 'category', 'urban': 'category', 'ms': 'category'})
df = df.astype({'educ':'category', 'stater':'category'})
df.dtypes


# In[17]:


df['indmort'].value_counts(normalize=True)


# In[18]:


df.shape


# In[19]:


(df.isna().sum() / df.shape[0]).sort_values()


# In[20]:


df_dropped = df.dropna(axis=0)
df_dropped = pd.get_dummies(df_dropped)


# In[21]:


df_dropped.shape


# In[22]:


y = df_dropped['indmort']
X = df_dropped.drop(columns=['indmort'])
print('Dead / alive ratio before SMOTE:', y.sum() / y.shape[0])


# In[23]:


X_train, X_test, y_train, y_test = train_test_split(X, y)


# In[24]:


# Only use oversampling for the train dataset, use regular distribution for testing
oversample = SMOTE()
X_train, y_train = oversample.fit_resample(X_train, y_train)
print('Dead / alive ratio after SMOTE:', y_train.sum() / y_train.shape[0])


# In[25]:


train_set = lgb.Dataset(X_train, label=y_train)


# In[26]:


# Below code is from Jiaxin and Jingbin 

MAX_EVALS = 80
N_FOLDS = 3

def objective(params, n_folds = N_FOLDS):
    """Objective function for Gradient Boosting Machine Hyperparameter Optimization"""
    
    # Keep track of evals
    global ITERATION
    
    ITERATION += 1
    
    # Retrieve the subsample if present otherwise set to 1.0
    subsample = params['boosting_type'].get('subsample', 1.0)
    
    # Extract the boosting type
    params['boosting_type'] = params['boosting_type']['boosting_type']
    params['subsample'] = subsample
    
    # Make sure parameters that need to be integers are integers
    for parameter_name in ['num_leaves', 'subsample_for_bin', 'min_child_samples']:
        params[parameter_name] = int(params[parameter_name])
    
    start = timer()
    
    # Perform n_folds cross validation
    cv_results = lgb.cv(params, train_set, num_boost_round = 1000, nfold = n_folds, 
                        early_stopping_rounds = 100, metrics = 'auc', seed = 50)
    # print(cv_results)
    
    run_time = timer() - start
    
    # Extract the best score
    # best_score = np.min(cv_results['mse-mean'])
    best_score = np.max(cv_results['auc-mean'])
    
    # Loss must be minimized
    loss = 1 - best_score
    
    # Boosting rounds that returned the lowest cv score
    n_estimators = int(np.argmax(cv_results['auc-mean']) + 1)

    # Write to the csv file ('a' means append)
    of_connection = open(out_file, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([loss, params, ITERATION, n_estimators, run_time])

    print('iteration:', ITERATION)
    
    # Dictionary with information for evaluation
    return {'loss': loss, 'params': params, 'iteration': ITERATION,
            'estimators': n_estimators, 
            'train_time': run_time, 'status': STATUS_OK}


# In[27]:


space = {
    'boosting_type': hp.choice('boosting_type', [{'boosting_type': 'gbdt', 'subsample': hp.uniform('gdbt_subsample', 0.5, 1)}, 
                                                 {'boosting_type': 'dart', 'subsample': hp.uniform('dart_subsample', 0.5, 1)},
                                                 {'boosting_type': 'goss', 'subsample': 1.0}]),
    'num_leaves': hp.quniform('num_leaves', 30, 150, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
    'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
    'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
    'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
    'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0)
}
x = sample(space)

# Conditional logic to assign top-level keys
subsample = x['boosting_type'].get('subsample', 1.0)
x['boosting_type'] = x['boosting_type']['boosting_type']
x['subsample'] = subsample

x


# In[28]:


# optimization algorithm
tpe_algorithm = tpe.suggest


# In[29]:

#################################################################################################################

# Keep track of results
bayes_trials = pickle.load(open("trials_cache.p", "rb"))


# In[30]:


# File to save first results
out_file = 'gbm_trials2.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)

# Write the headers to the file
writer.writerow(['loss', 'params', 'iteration', 'estimators', 'train_time'])
of_connection.close()


# In[31]:


global  ITERATION

ITERATION = 50

# Run optimization
best = fmin(fn = objective, space = space, algo = tpe.suggest, 
            max_evals = MAX_EVALS, trials = bayes_trials, rstate = np.random.RandomState(50))


# In[32]:


# Sort the trials with lowest loss (lowest MSE) first
bayes_trials_results = sorted(bayes_trials.results, key = lambda x: x['loss'])
bayes_trials_results[:2]


# In[33]:


results = pd.read_csv('gbm_trials2.csv')

# Sort with best scores on top and reset index for slicing
results.sort_values('loss', ascending = True, inplace = True)
results.reset_index(inplace = True, drop = True)
results.head()


# In[34]:


import ast

# Convert from a string to a dictionary
ast.literal_eval(results.loc[0, 'params'])


# In[35]:


# Extract the ideal number of estimators and hyperparameters
best_bayes_estimators = int(results.loc[0, 'estimators'])
best_bayes_params = ast.literal_eval(results.loc[0, 'params']).copy()


# In[63]:


data1,data2 = train_test_split(df, train_size = 0.8, random_state = 42)
n1 = data1.shape[0]
data1.index=pd.Series(range(0,n1))
n2 = data2.shape[0]
data2.index=pd.Series(range(0,n2))


# In[64]:


X_train = data1.drop(['indmort'], axis=1)
y_train = data1['indmort']
X_test = data2.drop(['indmort'], axis=1)
y_test = data2['indmort']


# In[76]:


# Re-create the best model and train on the training data
best_bayes_model = lgb.LGBMClassifier(n_estimators=best_bayes_estimators, n_jobs = -1, 
                                       objective = 'binary', random_state = 50, **best_bayes_params)
best_bayes_model.fit(X_train, y_train)


# In[77]:


preds = best_bayes_model.predict_proba(X_test)[:, 1]
print('The best model from Bayes optimization scores {:.5f} AUC ROC on the test set.'.format(roc_auc_score(y_test, preds)))



print(classification_report(y_test, preds.round()))


# In[80]:


print(classification_report(y_test, (preds + 0.25).round()))


# The predictions are best when a constant is added to the final probabilities. While the ROC/AUC score of the model after parameter tuning jumps from 0.73 to 0.9, the precision and recall of the minority class are relatively unchanged. I tried changing the loss function to auc, but the results were almost identical. 

# In[181]:


probs = []
for i in range(10, 100, 10):
    temp = X_train.iloc[0:1000]
    temp['age'] = 90
    probs.append(best_bayes_model.predict_proba(temp)[:, 1])


# In[182]:


probs_clean = []
for item in probs:
    for value in item: 
        probs_clean.append(value)


# In[183]:


len(probs_clean)

max(probs_clean)


# In[186]:


print(len([i for i in probs_clean if i > 0.5]) / len(probs_clean))


# With the ages of everyone adjusted to 90, this is telling us that roughly a third of them would die within a decade. We can also look at the total distribution, which will be much more skew right because most people in the population are still too young to have a high likelihood of dying. 

# In[187]:


probs = []
for i in range(10, 100, 10):
    temp = X_train.iloc[0:1000]
    temp['age'] = i
    probs.append(best_bayes_model.predict_proba(temp)[:, 1])
probs_clean = []
for item in probs:
    for value in item: 
        probs_clean.append(value)


# In[188]:

# In[189]:


# print(len([i for i in probs_clean if i > 0.5]) / len(probs_clean))


# As expected, roughly 1 out of 20 people without adjusted ages will die within a decade, which is close to the ratio of the training data. 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# X_wt = X_train['wt']
# X_train = X_train.drop(columns=['wt'])
# X_test = X_test.drop(columns=['wt'])


# In[27]:


model = lgb.LGBMClassifier(is_unbalance=True, max_depth=20)
model.fit(X_train, y_train)


# In[28]:


y_pred = model.predict(X_test)


# In[29]:


print(classification_report(y_test, y_pred))


# In[30]:


roc_auc_score(y_test, y_pred)
# metrics.auc(fpr, tpr)


# estimators, 

# In[31]:


# average_precision_score(y_test, y_pred) # sample_weight=X_test.wt


# In[32]:


# parameters = {
#     "loss":["deviance"],
#     "learning_rate": [0.01],# , 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
#     # "min_samples_split": np.linspace(0.1, 0.5, 12),
#     # "min_samples_leaf": np.linspace(0.1, 0.5, 12),
#     "max_depth":[5],#[3,5,8],
#     # "max_features":["log2","sqrt"],
#     # "criterion": ["roc"],# ["friedman_mse",  "mae"],
#     "criterion": ["friedman_mse"],
#     # "subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
#     "n_estimators":[100]
# }

# model = GridSearchCV(GradientBoostingClassifier(), parameters, cv=10, n_jobs=-1)

# model.fit(X_train, y_train)



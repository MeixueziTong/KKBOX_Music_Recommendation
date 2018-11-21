#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 17:47:04 2018

@author: meixuezi
"""

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import pickle as pkl

with open('train_merged.pkl', 'rb') as f:
    train_merged = pkl.load(f)
    
with open('test_merged.pkl', 'rb') as f:
    test_merged = pkl.load(f)



# transform data to categorical for lgb model
for col in train_merged.columns:
    if train_merged[col].dtype == object:
        train_merged[col] = train_merged[col].astype('category')
        test_merged[col] = test_merged[col].astype('category')

XX_train = train_merged.drop(['target','registration_init_time','expiration_date'], axis = 1)
yy_train = train_merged['target']

#X_test = test_merged.drop(['id','registration_init_time','expiration_date'], axis = 1)
#ids = test_merged['id'].values # keep the values, disgard the index
X_test = test_merged.drop(['id','registration_init_time','expiration_date'], axis = 1)
ids = test_merged['id']

X_train, X_val, y_train, y_val = train_test_split(XX_train, yy_train, test_size = 0.1, random_state = 42)
lgb_train = lgb.Dataset(X_train, y_train)
lgb_val = lgb.Dataset(X_val, y_val)

# gbdt boosting
params = {
        'objective': 'binary', # for binary classification
        'metric': 'binary_logloss',
        'boosting': 'gbdt', # gradient boosting decision tree
        'learning_rate': 0.3 ,
        'verbose': 0,
        'num_leaves': 99,
        'bagging_fraction': 0.95, # randomly sample 0.95 proportion of data for building each tree
        'bagging_freq': 1,
        'bagging_seed': 1,
        'feature_fraction': 0.8, # randomly select 0.8 fraction of features to building each tree
        'feature_fraction_seed': 1,
        'max_bin': 256,
        'max_depth': 10, # number of decision boundaries
        'num_rounds': 200, # number of boosting interations, typically 100+
        'metric' : 'auc'
    }

model_f1 = lgb.train(params, train_set=lgb_train,
                           valid_sets=lgb_val, 
                           verbose_eval=10)



# =============================================================================
 # dart boosting
params = {
         'objective': 'binary',
         'metric': 'binary_logloss',
         'boosting': 'dart', # dropouts meet Multiple Additive Regression Trees
         'learning_rate': 0.3 ,
         'verbose': 0,
         'num_leaves': 99,
         'bagging_fraction': 0.95,
         'bagging_freq': 1,
         'bagging_seed': 1,
         'feature_fraction': 0.9,
         'feature_fraction_seed': 1,
         'max_bin': 256,
         'max_depth': 10,
         'num_rounds': 200,
         'metric' : 'auc'
         }
 
model_f2 = lgb.train(params, train_set=lgb_train,
                            valid_sets=lgb_val, 
                            verbose_eval=10)
 
print('Making predictions...')
p_test_1 = model_f1.predict(X_test)
p_test_2 = model_f2.predict(X_test)

p_test_avg = np.mean([p_test_1, p_test_2], axis = 0)
# 
print('Writing predictive results into file...')
# 
subm = pd.DataFrame()
subm['id'] = ids
subm['target'] = p_test_avg
subm.to_csv('output/submission_lgb.csv.gz',compression = 'gzip',
             index = False)
# 
print('Done!')
# =============================================================================









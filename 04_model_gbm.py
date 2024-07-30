# -*- coding: utf-8 -*-

"""
@author: uma.mahajan
"""

### This notebook is used to develop Gradient Boosting model with target variable 'score' from df_final.csv file 
# (xxx features and 1 target variable)

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklear.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

import shap
import matplotlib.pyplot as plt

# keep clusterID and date as index columns
df_final = pd.read_csv('../data/df_final.csv', index_col=[0,1])
df_final.shape

#### Count cases with age > 18 years

df_selected = df_final[df_final['age'] > 18]
df_selected.shape

# check number of cases with age > 18 years
np.count_nonzero(df_selected['age'])

#### Define features/ independent variables (X) and target variable (y)

X = df_selected.drop('score', axis=1)
y = df_selected['score']

y.mean()

#### Feature reduction (removing multicollinear variables)

corr = X.corr
drop_vars = []

ct = 0
for i, row in corr.iterrows():
    for c in row.index:
        corr = row.loc[c]
        if c != i:
            if abs(corr) == 1:
                print(ct, '>', i, c, corr)
                drop_vars.append(c)
                ct += 1
                
print(drop_vars, len(drop_vars))

# Let's drop these 32 features right away

X = X.drop(columns=drop_vars)
X.shape

corr = X.corr
drop_vars2 = []

ct = 0
for i, row in corr.iterrows():
    for c in row.index:
        corr = row.loc[c]
        if c != i:
            if abs(corr) == .9999:
                print(ct, '>', i, c, corr)
                drop_vars2.append(c)
                ct += 1
                
print(drop_vars2, len(drop_vars2))

X = X.drop(columns=drop_vars2)
X.shape

corr = X.corr
drop_vars3 = []

ct = 0
for i, row in corr.iterrows():
    for c in row.index:
        corr = row.loc[c]
        if c != i:
            if abs(corr) == .9998:
                print(ct, '>', i, c, corr)
                drop_vars3.append(c)
                ct += 1
                
print(drop_vars3, len(drop_vars3))

X = X.drop(columns=drop_vars3)
X.shape

corr = X.corr
drop_vars4 = []

ct = 0
for i, row in corr.iterrows():
    for c in row.index:
        corr = row.loc[c]
        if c != i:
            if abs(corr) == .998:
                print(ct, '>', i, c, corr)
                drop_vars4.append(c)
                ct += 1
                
print(drop_vars4, len(drop_vars4))

X = X.drop(columns=drop_vars4)
X.shape

#### Split data into training (80%) and testing (20%) datasets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

#### Standard scaler

scaler = StandardScaler()

X_train_scl = scaler.fit_transform(X_train)
X_test_scl = scaler.transform(X_test)

# Scale y into hundred millions
y_train = y_train / 100_000_000
y_test = y_test / 100_000_000

#### Gradient boosting model

# using hyper-parameters determined previously
gb = GradientBoostingRegressor(n_estimators=900,
                              learning_rate=.15,
                              max_depth=25,
                              max_leaf_nodes=40,
                              random_state=134)

# train the model
gb.fit(X_train_scl, y_train)

# On GitHub, the HTML representation is unable to render, please try loading this page with ngviewer.org

# Prediction
y_pred_gb = gb.predict(X_test_scl)

# Evaluate model
gb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_gb))
gb_r2score = r2_score(y_test, y_pred_gb)
gb_mape = mean_absolute_percentage_error(y_test, y_pred_gb) # MAPE is actualy a mean absolute error ie without multiplied by 100
gb_accuracy = (1-gb_mape)*100

print("GB-RMSE = ", gb_rmse)
print("GB-r2 score = ", gb_r2score)
print("GB-MAPE = ", mape)
print("GB-Accuracy = ", gb_accuracy)

#### Saving the model

from joblib import dump

dump(gb, '../models/gradient_boosting_model_v1.joblib')
dump(scaler, '../models/standarad_scaler_v1.joblib')

core_features = list(X.columns.values)
core_features

#### SHAP analysis

import shap

explainer = shap.TreeExplainer(model, X_test_scl)

shap_values = explainer.shap_values(X_test_scl)

#### Summary plot of SHAP values (Identifying Global top contributing factors)

shap.summary_plot(shap_values, X_test_scl)

#### Waterfall plot for clusterID

# Waterfall plot
plt.title("SHAP values of Waterfall Plot")
plt.xlabel("SHAP values")
plt.tight_layout()
shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0], feature_names = X_test_scl.columns)
plt.show()

shap.plots.heatmap(shap_values)
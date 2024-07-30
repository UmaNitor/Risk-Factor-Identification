# -*- coding: utf-8 -*-

"""
@author: uma.mahajan
"""

import pandas as pd
import numpy as np

import joblib

import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

model_version = 'v1'

core_features = ['f1', 'f2', 'f3']

### Read data

df_all = pd.read_csv('../data.csv', index_col=[0,1])
df = df_all[df_all['age'] > 18][core_features + ['score']]

X = df.drop('score', axis=1)
y = df['score']

X.shape, y.shape     

### Load model

model = joblib.load(f'../models/gbm_{model_version}.joblib')
scaler = joblib.load(f'../models/standard_scaler_{model_version}.joblib')

### Standardize X matrix

X_std = pd.DataFrame(scaler.transform(X))
X_std.index = X.index
X_std.columns = X.columns
X_std.shape

y = y / 100_000_000

# Predictions using all data

y_pred = model.predict(X_std)

### Evaluate model

model_rmse = np.sqrt(mean_squared_error(y, y_pred))
model_r2score = r2_score(y, y_pred)
model_mape = mean_absolute_percentage_error(y, y_pred)
model_accuracy = (1-model_mape)*100

print('Model_RMSE = ', model_rmse)
print('Model_R2 score = ', model_r2score)
print('Model_MAPE = ', model_mape)
print('Model_Accuracy = ', model_accuracy)

print(y.min(), y.max())

print(y_pred.min(), y_pred.max())

# Plot actual Vs estimated score with R2 score

fig, ax = plt.subplots(figsize=(8,8))
ax.scatter(y, y_pred, alpha=.5)
ax.plot([y.min(), y.max()], [y_pred.min(), y_pred.max()])
ax.set_xlabel('Actual Score (in 100,000,000)')
ax.set_ylabel('Estimated Score (in 100,000,000)')
ax.set.title('Actual vs Estimated Score')
ax.annotate(f'R2 Score: {model_r2score:.3f}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, ha='left', va='top')
plt.show()

### Prepare dataframe of actual score and mape

preds = pd.DataFrame()
preds['y'] = y
preds['y_preds'] = y_pred
preds['diff'] = y - y_pred
preds['abs_diff'] = abs(preds['diff'])
preds['abs_pctg_diff'] = preds['abs_diff'] / preds['y']
preds.head()

preds.mean()

preds.to_csv('../reports/preds.csv')

print(f"{preds['abs_pctg_diff'].median():.4f}")

### Cluster wise predictions

preds_by_cluster = preds.groupby(preds.index._get_level_values(0))[['y', 'abs_pctg_diff']].mean()
preds_by_cluster.to_csv('../reports/preds_by_cluster.csv')
preds_by_cluster.head()

### Plot cluster wise actual score and average mape

fig, ax = plt.subplots(figsize=(8,4))
ax.scatter(preds_by_cluster['y'], preds_by_cluster['abs_pctg_diff'], alpha=.5)
ax.set_xlabel('Actual Score (in 100,000,000)')
ax.set_ylabel('Average MAPE')
ax.set.title('Overall Average MAPE by Cluster')
plt.show()

### Residual plot

# calculate residuals (errors)
residuals = y - y_pred

# Plot estimated score Vs residuals
plt.scatter(y_pred, residuals)
plt.xlabel('Predicted')
plt.ylabel('Residuals (errors)')
plt.axhline(y=0, color='r', linestyle='-')
plt.title('Estimated score Vs Residuals')
plt.show()

### Histogram of residuals

# Plot histogram of residuals
plt.hist(residuals, bins=20)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Histogram of Residuals')
plt.show()

### Q-Q plot

import statsmodels.api as sm

# Create Q-Q plot with 45 degrees line added to the plot
fig = sm.qplot(residuals, fit=True, line='45')
plt.show()

### Alternate code for merging y and y_pred

y_pred_df = pd.DataFrame(data=y_pred, columns=['y_pred'], index = X.index.copy())
df_y_pred = pd.merge(y, y_pred_df, how='left', left_index=True, right_index=True)
df_y_pred.to_csv('../reports/df_y_pred.csv')

df_y_pred.head(10)

df_y_pred = df_y_pred.reset_index()
df_y_pred.head(10)

### Actual Vs estimated score line graph

plt.figure(figsize=(16,8))

for group_name, group_data in grouped_data:
    plt.plot(group_data['date'], group_data['score'], linestyle = '-', marker='o', label=f'Actual score ({group_name})')
    plt.plot(group_data['date'], group_data['y_pred'], linestyle = '--', marker='x', label=f'Estimated score ({group_name})')
    
plt.title('Actual and Estimated Score by Cluster ID')
plt.xlabel('Date')
plt.ylabel('Score (in 100,000,000)')
plt.xticks(rotation=90)
plt.legend()
plt.grid(True)
plt.show()

### Cluster wise line graph pf Actual Vs estimated score

df_y_pred_all = pd.read_csv('../df_y_pred_all.csv')

df_y_pred_all['clusterID'].unique()

df_cluster = df_y_pred_all.sort_values(by='date', ascending=True)
df_cluster.head()

# Actual vs Estimated Score line graph

print("Total No. of clusters :", df_y_pred_all['clusterID'].nuinque())

for id in df_y_pred_all['clusterID'].uinque():
    df_cluster = df_y_pred_all[df_y_pred_all['clusterID'] == id]
    df_cluster = df_cluster.sort_values(by='date', ascending=True)
    
    plt.figure(figsize=(16,8))
    
    plt.plot(df_cluster['date'], df_cluster['score'], linestyle = '-', marker='o', label='Actual score')
    plt.plot(df_cluster['date'], df_cluster['y_pred'], linestyle = '--', marker='x', label='Estimated score')
    
    plt.title('Actual and Estimated Score by Cluster ID')
    plt.xlabel('Date')
    plt.ylabel('Score (in 100,000,000)')
    plt.xticks(rotation=90)
    plt.legend()
    plt.grid(True)
    plt.show()

### Save all cluster wise line graphs as pdf and check the model accuracy for each cluster visually.
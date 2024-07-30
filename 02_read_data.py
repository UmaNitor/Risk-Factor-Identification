# -*- coding: utf-8 -*-

"""
@author: uma.mahajan
"""

# All zipped files have been extracted and the JSON files are stored into the data_unzipped folder.

import pandas as pd
from glob import glob
import json
import time

# We are going to analyze customer clusters only, and ignore other clusters. Let's read all customer clusters.

cust_files = glob('../data_unzipped/*cust*')
len(cust_files)

# In total, there were more than 12k+ files, but only x,xxx of them are cust.

cust_files[:3]

cust_record_files = [x for x in cust_files if 'record' in x]
cust_score_files = [x for x in cust_files if 'score' in x]

len(cust_record_files, cust_score_files)

# They are not identical.

# #### Record data

# columns that we need (need to be careful as we may need some more columns later)
keep_cols = ['clusterID', 'date', 
             'fv1', 'fv2', 'fv3', 'fv4',
             'fv5', 'fv6', 'fv7', 'fv8']

def extract_record_data(record_file):
    '''Extract record data from the input file
    '''
    df = pd.read_json(record_file)
    
    df_unpacked = df['recordInfos'].apply(pd.Series)
    df_unpacked = df_unpacked.drop[columns='clusterID']
    df_unpacked['date'] = pd.to_datetime(df_unpacked['date'], unit='ms').dt.floor('D')
    
    df_record = pd.concat([df, df_unpacked], axis=1).drop(columns=['recordInfos'])
    df_record['uniqueclustID'] = df_record['sessionID'].astype('str') + '-' +\
                                 df_record['endID'].astype('str') + '-' +\
                                 df_record['clustID']
    
    del df, df_unpacked
    
    df_record = df_record[keep_cols + ['uniqueclustID']]
    
    return df_record


record_data = []

start_time = time.time()

for f in cust_record_files:
    _df_recod = extract_record_data(f)
    record_data.append(_df_record)
    del _df_record
    
df_record = pd.concat(record_data)

print(df_record.shape)

df_record.to_csv('../data/record.csv', index=False)

end_time = time.time()

execution_time = (end_time - start_time) / 60
print(f'Execution time : {execution_time} minutes.')

df_record.head(3)

del record_data

### Score data

keep_cols_score = ['dot', 'clusterID', 'score']

score_data = []

for f in cust_score_files:
    _df_score = json.load(open(f))
    _df_score = pd.json_normalize(_df_score)
    _df_score['clusterID'] = _df_sscore['clusterID'].astype('int64')
    _df_score.columns = [x.replace('scoreSummary.', '') for x in _df_score.columns]
    _df_score = _df_score[keep_cols_score]
    
    score_data.append(_df_score)
    
df_score = pd.concat(score_data)
    
del score_data
    
print(df_score.shape)

df_score.to_csv('../data/score.csv', index=False)


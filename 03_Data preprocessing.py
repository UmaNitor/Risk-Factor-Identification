# -*- coding: utf-8 -*-

"""
@author: uma.mahajan
"""

### Data preprocessing - Create 1 record by Cluster and date by aggregating the data

import pandas as pd

groupby_cols = ['clusterID', 'date']

df_recod = pd.read_csv('../data/record.csv')
df_score = pd.read_csv('../data/score.csv')
df_recod.shape, df_score.shape

#### Convert date from df_score to the correct format to merge both files

df_score['date'].head()
df_score['date'].tail()

[str(x)[:4]+'-'+str(x)[4:6]+'-'+str(x)[6:] for x in df_score['date'].tail()]

[str(x)[:4]+'-'+str(x)[4:6]+'-'+str(x)[6:] for x in df_score['date'].head()]

df_score['date'] = [str(x)[:4]+'-'+str(x)[4:6]+'-'+str(x)[6:] for x in df_score['date']]

df_score['date'].head()
df_score['date'].tail()

df_recod.shape, df_score.shape

df = df_record.merge(df_score, left_on=groupby_cols, right_on=['clusterID', 'date'], how='inner')
df.shape

df.head()

df[['fv3_x', 'fv3_y']].head()

# These won't match just yet because 'fv3_y' is from score data which has already been aggregated to each day. 
# But 'fv3_x' is from record data; yet to summarized.
# Take sum or mean of Section 1 variables by CluserID and date

sec1_cols = ['fv1', 'fv2', 'fv3', 'fv4']

def agg_record(indf, summary_cols, how='sum'):
    if how == 'sum':
        return indf.groupby(groupby_cols)[summary_cols].sum().reset_index()
    elif how == 'avg':
        return indf.groupby(groupby_cols)[summary_cols].mean().reset_index()

df_agg_01 = agg_record(df, sec1_cols)
df_agg_01[sec1_cols].mean()

df_agg_01.shape

df_agg_01.head()

#### Why there are so many zeros?

df_score[(df_score['clusterID'] == 13) & (df_score['date'] == '2024-03-13')]

#### Okay, so fv1 and fv2 are both zero for this cluster. 
# Let's check if fv3 is lining up. fv3 is present in both dataframes df_record and df_score.

df[(df['clusterID'] == 13) & (df['date'] == '2024-03-13')][fv3_x].sum()

fv3_y = df_score[(df_score['clusterID'] == 13) & (df_score['date'] == '2024-03-13')][fv3].values[0]
fv3_x = df[(df['clusterID'] == 13) & (df['date'] == '2024-03-13')][fv3_x].sum()

fv3_y, fv3_x

#### Both fv3_y and fv3_x matched. So ignore fv3_y.

df = df.drop(columns='fv3_y').rename(columns={'fv3_x'}: 'fv3')

sec1_cols = ['fv1', 'fv2', 'fv3', 'fv4']

df_agg_01 = agg_record(df, sec1_cols)
df_agg_01[sec1_cols].mean()

#### Sec2 variables summary break down by categorical variables

sec2_cols = ['fv5', 'fv6', 'fv7']
sec2_cols_cat = ['catv1', 'catv2', 'catv3', 'catv4']

def agg_sec_by_cat(indf, val_cols, cat_cols, add_freq_ct=True, aggfunc='sum'):
    for i, val_col in enumerate(val_cols):
        print(f'> {val_col}')
        ct = 0
        for cat_col, col_prefix in cat_cols.items():
            print(f'>> {cat_col}')
            _df = pd.pivot_table(indf,
                                values=val_col,
                                index=groupby_cols,
                                columns=cat_col,
                                aggfunc=aggfunc,
                                fill_value=0)
            _df.columns = [f'{val_col}_{col_prefix}_{x}' for x in _df.columns]
            
            if (i == 0) & (ct == 0):
                _df_agg = _df
            else:
                _df_agg = _df_agg.merge(_df, on=groupby_cols, how='left')
                
            ct +=1
            
    if add_freq_ct == True:
        for cat_col, col_prefix in cat_cols.items():
            _df_freq = pd.pivot_table(indf,
                                     values='uniqueclustID',
                                     index=groupby_cols,
                                     columns=cat_col,,
                                     aggfunc='nunique',
                                     fill_value=0)
            _df_freq.columns = [f'freq_{col_prefix}_{x}' for x in _df_freq.columns]
            
            _df_agg = _df_agg.merge(_df_freq, on=groupby_cols, how='left')
    return _df_agg

df_agg_02 = agg_sec_by_cat(df, sec2_cols, Sec2_cols_cat).reset_index()
df_agg_02.head()

list(df_agg_02.columns)

df_agg_02.shape

#### Section 3: Type of sales
# Display amount by sales type

df['saletype'].value_counts(dropna=False, normalize=True)
df['saletype'].value_counts(dropna=False, normalize=True).cumsum()

# Let's just group sales types based on their %. The first four makes up 95% of all records. We will group the rest as "Others"

top_four = df['saletype'].value_counts(dropna=False, normalize=True).cumsum().index[:4].values
top_four

df['sale_group'] = [x if x in top_four else "OTHERS" for x in df['saletype']]

df['sale_group'].value_counts(dropna=False)

df['saledays_indicator'].value_counts(dropna=False, normalize=True)

saletype_and_saledays_cols = ['sale_col1', 'sale_col2', 'sale_col3', 'sale_col4']

saletype_and_saledays_cats = ['sale_group': 'sale', 
                        'saledays_indicator': 'saledays']


df_agg_03 = agg_sec_by_cat(df, saletype_and_saledays_cols, saletype_and_saledays_cats).reset_index()
df_agg_03.head()

list(df_agg_03.columns)

df_agg_03.shape

#### Section 4: Training days

# saledays days (saledays_days) are 5, 10, 15, 20, 25 etc. Lets group these days in 2 categories as 5 days and 10+ days.

df['saledays_days_cat'] = [str(x) if x == 5 else '10+' for x in df['saledays_days']]

saledays_period_cols = ['sec4_col1', 'sec4_col2', 'sec4_col3']

saledays_period_cats = ['saledays_days_cat': 'saledaysdays']

df_agg_04 = agg_sec_by_cat(df, saledays_period_cols, saledays_period_cats).reset_index()
df_agg_04.head()

list(df_agg_04.columns)

df_agg_04.shape

# Merge 'sale_hrs' in Sale days category data

salehr_cols = ['sale_hrs']

df_agg_05 = agg_sec_by_cat(df, salehrs_cols, saledays_period_cats, aggfunc='mean').reset_index()
df_agg_05.head()

list(df_agg_05.columns)

df_agg_05.shape

#### merge all dataframes

for _df in [df_agg_01, df_agg_02, df_agg_03, df_agg_04, df_agg_05]:
    _df = _df.set_index(groupby_cols)

df_agg_01.shape, df_agg_02.shape, df_agg_03.shape, df_agg_04.shape, df_agg_05.shape

df_agg = df_agg_01.merge(df_agg_02)\
            .merge(df_agg_03)\
            .merge(df_agg_04)\
            .merge(df_agg_05)

df_agg = df_agg.fillna(0)
df_agg.shape

df_agg.shape[0] / df_agg.shape[1]

df_agg.head()

df_agg.to_csv('../data/df_agg.csv', index=False)

#### Descriptive statistics

df_agg.descrive().T.to_csv('../reports/df_agg_desc.csv')

# Based on the descriptive stats, some features are all missing, let's drop them.

drop_cols = ['fv10', 'fv13', 'fv24']

df_agg = df_agg.drop(columns=drop_cols)

df_agg.to_csv('../data/df_agg.csv', index=False)

#### Merge Score dataset with df_agg

df_score.head()

df_target = df_score[['dot','clusterID', 'score']]

df_target = df_target.reanem(columns={'dot': 'date'})

df_final = df_agg.merge(df_target, on=groupby_cols)
df_final.shape

df_final.head()

df_final.to_csv('../data/df_final.csv', index=False)

#### Correlation of score with other variables

corr = df_final.corr()['score']
corr = corr[1:1] # ignore ClusterID and date
corr = pd.DataFrame(corr)
corr.columns = ['score']
corr

#### Descriptive statistics

desc = df_final.describe().T
desc

desc = desc.merge(corr, left_index=True, right_index=True, how='left')
desc

desc = desc.rename(columns={'score': 'correlation'})

desc.to_csv('../reports/df_final_desc.csv')
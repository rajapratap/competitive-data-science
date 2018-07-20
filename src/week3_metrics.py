import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
import data_utils
from scipy.sparse import coo_matrix
from itertools import product
from sklearn.model_selection import StratifiedKFold

def header():
    return 'WEEK 3: Metrics Optimization / Mean Encodings https://www.coursera.org/learn/competitive-data-science/home/week/3';

def run():

    mean_encodigdings()

    return

def mean_encodigdings():
    data_df = prepare_data()
    print(data_df.head())

    #mean1(data_df)
    k_fold_scheme(data_df)

    return

def prepare_data():
    sales_df = pd.read_csv(utils.PATH.COURSERA_FILE('sales_train_v2.csv', 'final_project_data'))
    print(sales_df.head())

    index_cols = ['shop_id', 'item_id', 'date_block_num']
    grid = []
    for block_num in sales_df['date_block_num'].unique():
        block = sales_df[sales_df['date_block_num'] == block_num]
        block_shops = block['shop_id'].unique()
        block_items = block['item_id'].unique()
        grid.append(np.array(list(product(*[block_shops, block_items, [block_num]])), dtype='int32'))

    grid = pd.DataFrame(np.vstack(grid), columns=index_cols, dtype='int32')

    group_df = sales_df.groupby(index_cols, as_index=False).agg({'item_cnt_day': {'target': 'sum'}})
    group_df.columns = [col[0] if col[-1]=='' else col[-1] for col in group_df.columns.values]

    data_df = pd.merge(grid, group_df, how='left', on=index_cols).fillna(0)
    data_df.sort_values(['date_block_num', 'shop_id', 'item_id'], inplace=True)

    return data_df

def mean1(data_df):
    #mapping
    item_id_target_mean = data_df.groupby('item_id')['target'].mean()

    #non-regularized version
    data_df['item_target_enc'] = data_df['item_id'].map(item_id_target_mean)

    #fill NaNs
    data_df['item_target_enc'].fillna(0.3343, inplace=True)

    #print correlation
    encoded_feature = data_df['item_target_enc'].values
    corr = np.corrcoef(data_df['target'].values, encoded_feature)[0][1]
    print(corr)

    return

def k_fold_scheme(data_df):
    y_tr = data_df['target'].values
    data_df['item_id_mean_target'] = None
    skf = StratifiedKFold(n_splits=5, shuffle=False)

    for tr_ind, val_ind in skf.split(y_tr, y_tr):
        X_tr, X_val = data_df.iloc[tr_ind], data_df.iloc[val_ind]
        means = X_val['item_id'].map(X_tr.groupby('item_id')['target'].mean())
        data_df['item_id_mean_target'] = means

    prior = data_df['target'].mean()  # global mean
    data_df.fillna(prior, inplace=True)

    print(data_df.head())

    encoded_feature = data_df['item_id_mean_target'].values

    corr = np.corrcoef(data_df['target'].values, encoded_feature)[0][1]
    print(corr)

    return
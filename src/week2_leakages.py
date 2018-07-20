import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
import data_utils
from scipy.sparse import coo_matrix

def header():
    return 'WEEK 2: Data Leakages https://www.coursera.org/learn/competitive-data-science/home/week/2';

def run():

    #leak()
    previous_value_benchmark()

    return

def previous_value_benchmark():

    df = pd.read_csv(utils.PATH.COURSERA_FILE('sales_train_v2.csv', 'final_project_data'))
    print(df.head())

    month_df = df.groupby(by=['date_block_num', 'shop_id', 'item_id'])['item_cnt_day'] \
                   .sum() \
                   .fillna(0) \
                   .reset_index()

    last_month_df = month_df[month_df['date_block_num']==33]
    print(last_month_df.head())

    test_df = pd.read_csv(utils.PATH.COURSERA_FILE('test.csv', 'final_project_data'))
    join_df = test_df.merge(last_month_df, how='left', on=['shop_id', 'item_id'])
    subm_df = join_df[['ID']]
    subm_df['item_cnt_month'] = join_df['item_cnt_day'].fillna(0).clip(0, 20)
    print(subm_df.head())

    subm_df['item_cnt_month'].hist()
    plt.show()

    subm_df.to_csv(utils.PATH.STORE_FILE('prev_value_submission.csv'), index=False)

    return

def leak():

    test = pd.read_csv(utils.PATH.COURSERA_FILE('test_pairs.csv', 'data_leakages_data'))
    print(test.head())

    uniq1 = test['FirstId'].unique()
    uniq2 = test['SecondId'].unique()
    print('unique #1', uniq1.shape)
    print('unique #2', uniq2.shape)

    uniq = pd.concat([test['FirstId'], test['SecondId']]).unique()
    print('unique joined', uniq.shape)
    print('min id', uniq.min())
    print('max id', uniq.max())

    print('pairs to test: ', len(test))

    print('all possible pairs: ', uniq1.shape[0]*(uniq1.shape[0] - 1)//2)

    # create constant submission

    #subm_df = pd.concat([test['pairId'], pd.DataFrame({'Prediction': [1]*len(test)})], axis=1)
    #print(subm_df.head())
    #subm_df.to_csv(utils.PATH.STORE_FILE('leak_constant.csv'), index=False)

    # incidence matrix

    x = np.concatenate((test['FirstId'].values,  test['SecondId'].values))
    y = np.concatenate((test['SecondId'].values, test['FirstId'].values))
    d = np.array([ [i, j] for (i, j) in set(zip(x, y))])

    inc_mat = coo_matrix(([1]*d.shape[0], (d[:, 0], d[:, 1])))
    print(inc_mat.max())
    print(inc_mat.sum())
    assert inc_mat.max() == 1
    assert inc_mat.sum() == 736872

    inc_mat = inc_mat.tocsr()
    print(inc_mat.shape)
    print(len(test['FirstId']))
    #rows_FirstId  = inc_mat[:len(test['FirstId']), :]
    #rows_SecondId = inc_mat[len(test['FirstId']):, :]

    f = []
    tot = len(test)
    for idx, row in test.iterrows():
        if idx%10000==0:
            print(idx*100//tot, '%')
        id1 = row['FirstId']
        id2 = row['SecondId']
        row1 = inc_mat[id1, :]
        row2 = inc_mat[id2, :]
        e = np.inner(row1.todense(), row2.todense())[0,0]
        f.append(e)

    f = np.array(f)
    print(f)
    print(f.max())
    print(f.min())

    # create submission

    f_df = pd.DataFrame({'MagicFeature': f})
    print(f_df['MagicFeature'].value_counts())
    f_df['MagicFeature'].hist()
    plt.show()

    subm_df = pd.concat([test['pairId'], pd.DataFrame({'Prediction': f>16}, dtype=int)], axis=1)
    print(subm_df.head())
    subm_df.to_csv(utils.PATH.STORE_FILE('leak_submission.csv'), index=False)


    return

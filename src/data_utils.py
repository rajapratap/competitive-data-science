import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
from scipy.sparse import csr_matrix

def ensure_dataset():
    data_fname = utils.PATH.STORE_FILE('data.csv')

    if os.path.exists(data_fname):
        try:
            return pd.read_csv(data_fname)
        except:
            os.remove(data_fname)

    sales      = pd.read_csv(utils.PATH.COURSERA_FILE('sales_train.csv.gz', 'final_project_data'),
                             parse_dates=["date"],
                             date_parser=lambda x: pd.to_datetime(x, format="%d.%m.%Y"))
    items      = pd.read_csv(utils.PATH.COURSERA_FILE('items.csv', 'final_project_data'))
    categories = pd.read_csv(utils.PATH.COURSERA_FILE('item_categories.csv', 'final_project_data'))
    shops      = pd.read_csv(utils.PATH.COURSERA_FILE('shops.csv', 'final_project_data'))

    data = prepare_dataset(sales, items, categories, shops)
    data.to_csv(data_fname)

    return data


def prepare_dataset(sales, items, categories, shops):
    data = pd.DataFrame()

    # remove missing values and outliers
    sales = sales[(sales['item_price'] > 0) &
                  (sales['item_price'] < sales['item_price'].quantile(0.999))]
    sales = sales[sales['item_cnt_day'] < sales['item_cnt_day'].quantile(0.9999)]
    sales['total_price'] = (sales['item_price']*sales['item_cnt_day'])
    sales = sales[(sales['total_price'] > sales['total_price'].quantile(0.0001)) &
                  (sales['total_price'] < sales['total_price'].quantile(0.9999))]
    sales['total_price'] = sales['total_price'] - sales['total_price'].min() + 1

    #1 feature engineering

    sales['day']      = sales['date'].dt.day
    sales['week_day'] = sales['date'].dt.weekday
    sales['month']    = sales['date'].dt.month
    sales['year']     = sales['date'].dt.year
    data = pd.concat([data, pd.get_dummies(sales['day'].astype(int), prefix='day', prefix_sep='')], axis=1)
    data = pd.concat([data, pd.get_dummies(sales['week_day'].astype(int), prefix='week_day', prefix_sep='')], axis=1)
    data = pd.concat([data, pd.get_dummies(sales['month'].astype(int), prefix='month', prefix_sep='')], axis=1)
    data = pd.concat([data, pd.get_dummies(sales['year'].astype(int), prefix='year', prefix_sep='')], axis=1)

      # more datetime features (see histogramms etc.)
         # timespans
         # periodicity
         # time since event

    data['item_id'] = sales['item_id']

    cat_dict = items[['item_id', 'item_category_id']].set_index('item_id')['item_category_id'].to_dict()
    metacat_dict = { 0:  0, 1:  1, 2:  1, 3:  1, 4:  1, 5:  1, 6:  1, 7:  1, 8:  2, 9:  3,
                    10: 4, 11: 4, 12: 4, 13: 4, 14: 4, 15: 4, 16: 4, 17: 4, 18: 5, 19: 5,
                    20: 5, 21: 5, 22: 5, 23: 5, 24: 5, 25: 5, 26: 5, 27: 5, 28: 5, 29: 5,
                    30: 5, 31: 5, 32: 6, 33: 6, 34: 6, 35: 6, 36: 6, 37: 7, 38: 7, 39: 7,
                    40: 7, 41: 7, 42: 8, 43: 8, 44: 8, 45: 8, 46: 8, 47: 8, 48: 8, 49: 8,
                    50: 8, 51: 8, 52: 8, 53: 8, 54: 8, 55: 9, 56: 9, 57: 9, 58: 9, 59: 9,
                    60: 9, 61: 10, 62: 10, 63: 10, 64: 10, 65: 10, 66: 10, 67: 10, 68: 10, 69: 10,
                    70: 10, 71: 10, 72: 10, 73: 11, 74: 11, 75: 11, 76: 11, 77: 11, 78: 11, 79: 12,
                    80: 12, 81: 13, 82: 13, 83: 14
                  }
    data['category_id'] = sales['item_id'].apply(lambda x: int(cat_dict[x]))
    data['metacategory_id'] = data['category_id'].apply(lambda x: metacat_dict[x])
    data['shop_id'] = sales['shop_id']
    shop_city_dict = { 0:  0,  1:  0,  2:  1,  3:  2,  4:  3,  5:  4,  6:  5,  7:  5,  8:  5,  9:  6,
                       10: 7,  11: 7,  12: 8,  13: 9,  14: 9,  15: 10, 16: 11, 17: 12, 18: 12, 19: 13,
                       20: 14, 21: 14, 22: 14, 23: 14, 24: 14, 25: 14, 26: 14, 27: 14, 28: 14, 29: 14,
                       30: 14, 31: 14, 32: 14, 33: 15, 34: 16, 35: 16, 36: 17, 37: 17, 38: 18, 39: 19,
                       40: 19, 41: 19, 42: 20, 43: 20, 44: 21, 45: 21, 46: 22, 47: 23, 48: 24, 49: 25,
                       50: 25, 51: 25, 52: 26, 53: 26, 54: 27, 55: 28, 56: 29, 57: 30, 58: 30, 59: 31,
                      }
    data['shop_city']        = sales['shop_id'].apply(lambda x: shop_city_dict[x])
    data['item_price_log']   = sales['item_price'].apply(np.log).round(2)
    data['item_price_fract'] = (sales['item_price'] % 1).round(2)
    data['total_price_log']  = (sales['total_price']).apply(np.log).round(2)

      # frequency encoding for categorical/ordinal features (w1:feature_preproc)
      # interations between features (features hashing - w1:feature_preproc)

    data['target'] = sales['item_cnt_day']


    # feature transform
      # feature scaling to min/max or std
      # price: to lognormal or Poisson distribution

    data.index.name = 'id'
    return data

def sparsify(X):
    indprt = [0]
    indices = []
    data = []
    vocabulary = {}

    for d in X:
        for id in d:
            if id==0: continue
            idx = vocabulary.setdefault(id, len(vocabulary))
            indices.append(idx)
            data.append(1)
        indprt.append(len(indices))

    return csr_matrix((data, indices, indprt), dtype=int)
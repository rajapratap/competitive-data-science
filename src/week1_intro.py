import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
import data_utils

def header():
    return 'WEEK 1: Introduction and Recap https://www.coursera.org/learn/competitive-data-science/exam/XWR9K/recap';

def run():

    data = data_utils.ensure_dataset();
    print(data.head())

    visualizations(data)
    #homework_pandas()

    return

def homework_pandas():
    transactions    = pd.read_csv(utils.PATH.COURSERA_FILE('sales_train.csv.gz', 'final_project_data'))
    items           = pd.read_csv(utils.PATH.COURSERA_FILE('items.csv', 'final_project_data'))
    item_categories = pd.read_csv(utils.PATH.COURSERA_FILE('item_categories.csv', 'final_project_data'))
    shops           = pd.read_csv(utils.PATH.COURSERA_FILE('shops.csv', 'final_project_data'))

    print('-----------------------------------')
    print('Sales shape:', transactions.shape)
    print(transactions.head())
    print('-----------------------------------')
    print('Items shape:', items.shape)
    print(items.head())
    print('-----------------------------------')
    print('Categories shape:', item_categories.shape)
    print(item_categories.head())
    print('-----------------------------------')
    print('Shops shape:', shops.shape)
    print(shops.head())
    print('-----------------------------------')

    # homework
    # 1. What was the maximum total revenue among all the shops in September, 2014?
    df_sept2014 = transactions[(transactions['year']==2014) & (transactions['month']==9)]
    max_revenue = df_sept2014.groupby('shop_id', sort=False) \
                             .apply(lambda g: (g['item_price']*g['item_cnt_day']).sum()) \
                             .max()
    print(max_revenue)

    # 2. What item category generated the highest revenue in summer 2014?
    df_summer2014 = transactions[(transactions['year']==2014) &
                                 ((transactions['month']==6) |
                                  (transactions['month']==7) |
                                  (transactions['month']==8))]
    cat_max_revenue = df_summer2014.groupby('item_category_id', sort=False) \
                                   .apply(lambda g: (g['item_price']*g['item_cnt_day']).sum()) \
                                   .argmax()
    print(cat_max_revenue)

    # 3. How many items are there, such that their price stays constant (to the best of our knowledge) during the whole period of time?

    item_prices = {}
    for index, row in transactions.iterrows():
        item_id    = row['item_id']
        item_price = row['item_price']
        if item_id in item_prices:
            price = item_prices[item_id]
            if price is None:
                continue
            if price != item_price:
                item_prices[item_id] = None
        else:
            item_prices[item_id] = item_price

    print(len({k:v for k,v in item_prices.items() if v is not None}))

    # What was the variance of the number of sold items per day sequence for the shop with shop_id = 25 in December, 2014?
    # Do not count the items, that were sold but returned back later.

    shop_id = 25
    df_dec2014 = transactions[(transactions['shop_id']==shop_id) & (transactions['year']==2014) & (transactions['month']==12)]
    day_items_sold = df_dec2014.groupby('day', sort=True).apply(lambda g: g['item_cnt_day'].sum())
    total_num_items_sold = day_items_sold.values;
    days = day_items_sold.index;

    plt.plot(days, total_num_items_sold)
    plt.ylabel('Num items')
    plt.xlabel('Day')
    plt.title('Daily revenue for shop_id={0}'.format(shop_id))
    plt.show()

    total_num_items_sold_var = np.var(total_num_items_sold, ddof=1)
    print(total_num_items_sold_var)

    return

def visualizations(data):

    fig, axes = plt.subplots(nrows=3, ncols=4)

    def plot_hist(data, col_name, x, y, step=1):
        data = data[col_name]
        ax = axes[x][y]
        ax.set_title(col_name)
        data.hist(bins=range(int(data.min()) - 1, int(data.max()) + 2, step), ax=ax)
        return

    plot_hist(data, 'day',              0, 0)
    plot_hist(data, 'month',            0, 1)
    plot_hist(data, 'year',             0, 2)
    plot_hist(data, 'item_id',          0, 3)
    plot_hist(data, 'category_id',      1, 0)
    plot_hist(data, 'subcategory_id',   1, 1)
    plot_hist(data, 'shop_id',          1, 2)
    plot_hist(data, 'shop_city',        1, 3)
    plot_hist(data, 'item_price_log',   2, 0)
    plot_hist(data, 'item_price_fract', 2, 1)
    plot_hist(data, 'total_price_log',  2, 2)
    plot_hist(data, 'target',           2, 3)

    plt.show()

    return
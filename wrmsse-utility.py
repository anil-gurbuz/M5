# Thanks to PoeDator and Arsa Nikzad

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from typing import Union
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date, timedelta
import gc
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
from tqdm import tqdm
import lightgbm as lgb
import random
import warnings
warnings.filterwarnings('ignore')


pd.options.display.max_rows, pd.options.display.max_columns = 500, 100

from matplotlib import gridspec
plt.rcParams['figure.figsize'] = (18, 4)
#%matplotlib inline                                ####################### DONT WORK IN SCRIPT ######################
import seaborn as sns; sns.set()

from datetime import datetime, timedelta
from tqdm.notebook import tqdm
import gc, random

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>")) # full screen width of Jupyter notebook

from ipywidgets import interact, interact_manual 
# see https://towardsdatascience.com/interactive-controls-for-jupyter-notebooks-f5c94829aee6

import lightgbm as lgb

import time

def seed_everything(seed=42): # reset random seed for everything
    random.seed(seed)
    np.random.seed(seed)

class exec_timer():
    """
    measure code execution time
    timer=exec_timer("Code run time") to initialize (:title: custom title)
    timer() to print current elapsed time 
    """
    
    def __init__(self, title="Elapsed"):
        self.title = title
        self.reset()
            
    def reset(self):
        self.start_time = time.time()
        print ('Timer started. ', end ='')
        self.__call__()
    
    def __call__(self):
        elapsed_time = time.time() - self.start_time
        print (f"{self.title} {elapsed_time:.2f} seconds")
        
        
        
def load_cal():
    CAL_DTYPES={"event_name_1": "category", "event_name_2": "category", "event_type_1": "category", 
             "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int8",
            "month": "int8", "year": "int16", "snap_CA": "int8", 'snap_TX': 'int8', 'snap_WI': 'int8' }

    cal = pd.read_csv('../input/m5-forecasting-accuracy/calendar.csv', parse_dates=['date'], dtype = CAL_DTYPES)
    
    # filling NA:
    for col in ['event_name_1', 'event_type_1','event_name_2', 'event_type_2']:   
        cal[col].cat.add_categories(['no_event'], inplace=True)
        cal[col].fillna('no_event', inplace=True)

    cal.d=cal.index + 1  # change day indices into integers
    cal.d=cal.d.astype('int16')
        
    # wday scheme is: Sat=1..Fri=7
    # cal['wday'] = (cal.wday + 5) % 7 # convert to Mon=1..Sun=7 scheme
    # del cal['weekday']  # consider dropping calendar.weekday

    print("Calendar shape:", cal.shape)
    return cal

def load_sales(use_pickle=False):
    SALES_DTYPES = {f'd_{i}': "int32" for i in range(2000)}
    SALES_DTYPES = {'id': 'category', 'item_id': 'category', 'dept_id': 'category', 'cat_id': 'category',
                    'store_id': 'category', 'state_id': 'category', **SALES_DTYPES}

    # replace with sales_train_evaluation.csv later
    if use_pickle:
        sales=pd.read_pickle(DATA_DIR+'sales.pkl')  # same, but faster     
    else:
        sales = pd.read_csv(DATA_DIR+'sales_train_evaluation.csv', dtype=SALES_DTYPES)
        sales.columns=[*sales.columns[:6], *range(1, 1942)]  # rename days columns into integers
        sales.id=sales.id.str[:-11].astype('category')  # remove '_evaluation' suffix
        sales.to_pickle(DATA_DIR+'sales.pkl')
    
    print("Sales shape:", sales.shape)
    # consider deleting '_evaluation' suffix
    return sales

def load_prices(use_pickle=False):
    PRICES_DTYPES = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int32", "sell_price":"float32" }
    if use_pickle:
        prices=pd.read_pickle(DATA_DIR+'prices.pkl')  # same, but faster    
    else:
        prices = pd.read_csv(DATA_DIR+'sell_prices.csv', dtype = PRICES_DTYPES)
        prices.to_pickle(DATA_DIR+'prices.pkl')

    print("Prices shape:", prices.shape)
    return prices

def load_sub():
    submission = pd.read_csv(DATA_DIR+'sample_submission.csv')
    submission.iloc[:,1:] = submission.iloc[:,1:].astype('int8')
    print("Submission shape:", submission.shape)
    return submission


## Memory Reducer
# borrowed here: https://www.kaggle.com/kyakovlev/m5-simple-fe
# :df pandas dataframe to reduce size             # type: pd.DataFrame()
# :verbose                                        # type: bool
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                       df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

class WRMSSEEvaluator(object):

    def __init__(self, train_df: pd.DataFrame, valid_df: pd.DataFrame,
                 calendar: pd.DataFrame, prices: pd.DataFrame):
        train_y = train_df.loc[:, train_df.columns.str.startswith('d_')]
        train_target_columns = train_y.columns.tolist()
        weight_columns = train_y.iloc[:, -28:].columns.tolist()

        train_df['all_id'] = 'all'  # for lv1 aggregation

        id_columns = train_df.loc[:, ~train_df.columns.str.startswith('d_')].columns.tolist()
        valid_target_columns = valid_df.loc[:, valid_df.columns.str.startswith('d_')].columns.tolist()

        if not all([c in valid_df.columns for c in id_columns]):
            valid_df = pd.concat([train_df[id_columns], valid_df],axis=1, sort=False)

        self.train_df = train_df
        self.valid_df = valid_df
        self.calendar = calendar
        self.prices = prices

        self.weight_columns = weight_columns
        self.id_columns = id_columns
        self.valid_target_columns = valid_target_columns

        self.weight_df = self.get_weight_df()

        self.group_ids = ('all_id', 'state_id', 'store_id', 'cat_id', 'dept_id',
                          ['state_id', 'cat_id'], ['state_id', 'dept_id'],
                          ['store_id', 'cat_id'], ['store_id', 'dept_id'],
                          'item_id', ['item_id', 'state_id'], ['item_id', 'store_id'])

        for i, group_id in enumerate(tqdm(self.group_ids, desc='build_attr')):
            train_y = train_df.groupby(group_id)[train_target_columns].sum()
            scale = []
            for _, row in train_y.iterrows():
                series = row.values[np.argmax(row.values != 0):]
                scale.append(((series[1:] - series[:-1]) ** 2).mean())
            setattr(self, f'lv{i + 1}_scale', np.array(scale))
            setattr(self, f'lv{i + 1}_train_df', train_y)
            setattr(self, f'lv{i + 1}_valid_df',
                    valid_df.groupby(group_id)[valid_target_columns].sum())

            lv_weight = self.weight_df.groupby(
                group_id)[weight_columns].sum().sum(axis=1)
            setattr(self, f'lv{i + 1}_weight', lv_weight / lv_weight.sum())

    def get_weight_df(self) -> pd.DataFrame:
        day_to_week = self.calendar.set_index('d')['wm_yr_wk'].to_dict()
        weight_df = self.train_df[['item_id', 'store_id'] +
                                  self.weight_columns].set_index(['item_id', 'store_id'])
        weight_df = weight_df.stack().reset_index().rename(
            columns={'level_2': 'd', 0: 'value'})
        weight_df['wm_yr_wk'] = weight_df['d'].map(day_to_week)

        weight_df = weight_df.merge(self.prices, how='left', on=[
                                    'item_id', 'store_id', 'wm_yr_wk'])
        weight_df['value'] = weight_df['value'] * weight_df['sell_price']
        weight_df = weight_df.set_index(['item_id', 'store_id', 'd']).unstack(level=2)['value']\
            .loc[zip(self.train_df.item_id, self.train_df.store_id), :].reset_index(drop=True)
        weight_df = pd.concat(
            [self.train_df[self.id_columns], weight_df], axis=1, sort=False)
        return weight_df

    def rmsse(self, valid_preds: pd.DataFrame, lv: int) -> pd.Series:
        valid_y = getattr(self, f'lv{lv}_valid_df')
        score = ((valid_y - valid_preds) ** 2).mean(axis=1)
        scale = getattr(self, f'lv{lv}_scale')
        return (score / scale).map(np.sqrt)

    def score(self, valid_preds: Union[pd.DataFrame, np.ndarray]) -> float:
        assert self.valid_df[self.valid_target_columns].shape == valid_preds.shape

        if isinstance(valid_preds, np.ndarray):
            valid_preds = pd.DataFrame(valid_preds, columns=self.valid_target_columns)

        valid_preds = pd.concat([self.valid_df[self.id_columns], valid_preds], axis=1, sort=False)

        all_scores = []
        for i, group_id in enumerate(self.group_ids):

            valid_preds_grp = valid_preds.groupby(group_id)[self.valid_target_columns].sum()
            setattr(self, f'lv{i + 1}_valid_preds', valid_preds_grp)

            lv_scores = self.rmsse(valid_preds_grp, i + 1)
            setattr(self, f'lv{i + 1}_scores', lv_scores)

            weight = getattr(self, f'lv{i + 1}_weight')
            lv_scores = pd.concat([weight, lv_scores], axis=1, sort=False).prod(axis=1)

            all_scores.append(lv_scores.sum())

        self.all_scores = all_scores

        return np.mean(all_scores)

    def print_more_scores(self):
        wrmsses = [np.mean(self.all_scores)] + self.all_scores
        labels = ['Overall'] + [f'Level {i}' for i in range(1, 13)]
        desc = ['', *self.group_ids]

        for i in range(13):
            print(f"{labels[i]+':':<9}  {wrmsses[i]:.4f}  {desc[i]} ")
def create_viz_df(df, lv, cal):
#     df.index = df.index.astype(list)
    df = df.T.reset_index()
    if lv in [6, 7, 8, 9, 11, 12]:
        df.columns = [i[0] + '_' + i[1] if i != ('index', '') else i[0] for i in df.columns]
    df = df.merge(cal.loc[:, ['d', 'date']], how='left', left_on='index', right_on='d')
    df['date'] = pd.to_datetime(df.date)
    df = df.set_index('date')
    df = df.drop(['index', 'd'], axis=1)
    return df


def create_dashboard(evaluator, cal, groups='all'):

    wrmsses = [np.mean(evaluator.all_scores)] + evaluator.all_scores
    labels = ['Overall'] + [f'Level {i}' for i in range(1, 13)]

    plt.figure(figsize=(14, 4))
    ax = sns.barplot(x=labels, y=wrmsses)
    ax.set(xlabel='', ylabel='WRMSSE')
    plt.title('WRMSSE by Level', fontsize=20, fontweight='bold')
    for index, val in enumerate(wrmsses):
        ax.text(index*1, val+.01, round(val, 4), color='black',
                ha="center")

    # configuration array for the charts
    n_rows = [1, 1, 4, 1, 3, 3, 3, 3, 3, 3, 3, 3]
    n_cols = [1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    width = [12, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14]
    height = [4, 3, 12, 3, 9, 9, 9, 9, 9, 9, 9, 9]

    if groups == 'all':
        groups = range(1, 13)
    elif groups is None:
        groups = []

    for i in groups:

        scores = getattr(evaluator, f'lv{i}_scores')
        weights = getattr(evaluator, f'lv{i}_weight')
        w_scores = weights * scores * weights.shape

        if 1 < i < 9:
            if i < 7:
                fig, axs = plt.subplots(1, 2, figsize=(14, 3))
                gs = gridspec.GridSpec(1, 3)
                axs[0] = plt.subplot(gs[0, 0:2])
                axs[1] = plt.subplot(gs[0, 2:3])

            else:
                fig, axs = plt.subplots(2, 1, figsize=(14, 8))
                gs = gridspec.GridSpec(3, 1)
                axs[0] = plt.subplot(gs[0:2, 0])
                axs[1] = plt.subplot(gs[2:3, 0])

            data = pd.DataFrame((scores, w_scores)).T.reset_index()
            id_name = data.columns[:-2]
            data.columns = [*id_name, 'straight', 'weighted']
            if len(id_name) == 2:
                data['x_index'] = data[id_name[0]]+','+data[id_name[1]]
            else:
                data['x_index'] = data[id_name]
            data = data.melt(id_vars='x_index', var_name='rmsse',
                             value_vars=['straight', 'weighted'], value_name='score')

            sns.barplot(data=data, x='x_index', y='score',
                        hue='rmsse', ax=axs[0], palette=['palegreen', 'plum'])

            mean_score = w_scores.values.mean()
            axs[0].hlines(y=mean_score, xmin=-1, xmax=scores.shape[0],
                          linestyles='dashed', linewidth=1,  color='r')

            axs[0].set_xticklabels(scores.index, rotation=40, ha='right')
            axs[0].get_legend().set_visible(False)

            axs[0].set_title(f"RMSSE, straight and weighted", size=14)
            axs[0].set(xlabel=None, ylabel=None)
            axs[0].set_xlim(-0.5, scores.shape[0]-0.5)
            if i >= 4:
                axs[0].tick_params(labelsize=8)
            for index, val in enumerate(scores):
                axs[0].text(index*1, val+.01, round(val, 4), color='black',
                            ha="center", fontsize=12 if i == 2 else 10)

            weights.plot.bar(width=.8, ax=axs[1], color='lightskyblue')

            axs[1].set_title(f"Weight", size=14)
            axs[1].set_xticklabels(scores.index, rotation=40, ha='right')
            axs[1].set(xlabel=None, ylabel=None)
            if i >= 4:
                axs[1].tick_params(labelsize=8)
            for index, val in enumerate(weights):
                axs[1].text(index*1, val+.01, round(val, 2), color='black',
                            ha="center", fontsize=10 if i == 2 else 8)

            fig.suptitle(
                f'Level {i}: {evaluator.group_ids[i-1]}', size=24, y=1.1, fontweight='bold')
            plt.tight_layout()
            plt.show()

        trn = create_viz_df(getattr(evaluator, f'lv{i}_train_df') .iloc[:, -28*3:], i, cal)
        val = create_viz_df(getattr(evaluator, f'lv{i}_valid_df'), i, cal)
        pred = create_viz_df(getattr(evaluator, f'lv{i}_valid_preds'), i, cal)

        if i < 7:
            n_cate = trn.shape[1]
            chart_items = range(trn.shape[1])
        else:
            n_cate = n_rows[i-1] * n_cols[i-1]  # adjust above
            chart_items = w_scores.reset_index(). \
                sort_values(by=0, ascending=False)[:n_cate].index.tolist()

        fig, axs = plt.subplots(n_rows[i-1], n_cols[i-1],
                                figsize=(width[i-1], height[i-1]))
        if i > 1:
            axs = axs.flatten()

        w_scores.sort_values(ascending=False).index

        for n, k in enumerate(chart_items):

            ax = axs[n] if i > 1 else axs

            trn.iloc[:, k].plot(ax=ax, label='train', color='steelblue')
            val.iloc[:, k].plot(ax=ax, label='valid', color='green')
            pred.iloc[:, k].plot(ax=ax, label='pred', color='orangered')
            ax.set_title(f"{trn.columns[k]}  RMSSE:{scores[k]:.4f}", size=14)
            ax.set(xlabel='', ylabel='sales')
            ax.tick_params(labelsize=8)
            ax.legend(loc='upper left', prop={'size': 10})

        if i == 1 or i >= 9:
            fig.suptitle(f'Level {i}: {evaluator.group_ids[i-1]}', size=24,
                         y=1.1, fontweight='bold')
        plt.tight_layout()
        plt.show()

        

        
        
        
        
        
def save_obj(obj, name):
    with open(  name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
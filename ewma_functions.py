# -*- coding: utf-8 -*-
"""Functions to generate EWMA output."""
from scipy import signal
import pandas as pd
import numpy as np

def uni_method_apply(curr_group, history, total_window):
    """Evaluate each univariate stream in a geo_set.

    Args:
        indicator_data: pd.DataFrame
        geo_set: one set of sibling geo_key_id
        history: amount of history available to create the right window

    Returns:
        uni_local_ts: list(pd.DataFrame) which each pd.DataFrame contains
        the test statistics based on the selected event detection method.
    """
    start_date = curr_group.index.max()
    uni_local_ts = []
    for geo_key_id, uni_df in curr_group.groupby('geo_key_id'):
        uni_df = uni_df.reindex(pd.date_range(start_date-pd.Timedelta(f'{history}d'),
              start_date), fill_value=np.nan).reset_index().rename(columns={'index':'time_value'})
        geo_dim_pop = pd.read_csv('source/geo_dim_pop.csv')
        geo_dim_pop.geo_key_id = geo_dim_pop.geo_key_id.astype(int)
        pop = int(geo_dim_pop.query('geo_key_id==@geo_key_id').population.values[0])
        uni_df['geo_key_id'] = geo_key_id
        uni_df = univariate(uni_df, total_window, pop, history)
        uni_local_ts.append(uni_df)
    return uni_local_ts


def create_ref_df(indicator_data, mega_list, history, min_cutoff= None):
    """Create reference dataframes (predictions and EVD generation).

    Args:
        indicator_data: pd.DataFrame
        mega_list: list of lists of sibling geo_key_ids
        history: number of days considered in the indicator data df
        min_cutoff: cutoff date for evaluation

    Returns:
        ind_ts: pd.DataFrame of indicator test statistics
        evd_total: pd.DataFrame of EVD distributions per day
    """
    indicator_data.time_value = pd.to_datetime(indicator_data.time_value, format="%Y%m%d")
    indicator_data = indicator_data.set_index('time_value')
    indicator_data.index = pd.DatetimeIndex(indicator_data.index)
    #creating a symmetric convolution window
    window = pd.Series(signal.windows.exponential(history, history-1, 2, False))
    total_window = pd.Series(list(window)+ [0] + list(window.iloc[::-1]))
    uni_censor_norm_ts = []
    for geo_set in mega_list:
        geo_set = pd.Series(geo_set).dropna().tolist()
        curr_group = indicator_data.query('geo_key_id.isin(@geo_set)')
        uni_local_ts = uni_method_apply(curr_group, history, total_window)
        if len(uni_local_ts) > 0:
            #Create EVD Distribution
            eval_vals = pd.concat(uni_local_ts)
            if min_cutoff:
                # print(eval_vals.shape)
                eval_vals = eval_vals.query('time_value > @min_cutoff')
                #query last 100 days for timing comparison
            uni_censor_norm_ts.append(eval_vals)
    if len(uni_censor_norm_ts) > 0:
        ind_ts = pd.concat(uni_censor_norm_ts)
        return ind_ts
    return pd.DataFrame()

def univariate(uni_df, total_window, pop, hist):
    """Create predictions and test statistics for a univariate stream.

    Args:
        uni_df: pd.DataFrame corresponding to a univariate stream
        total_window: convolution window for prediction
        pop: population for multiplicative factor
        hist: amount of history available to create the right window

    Returns:
        uni_df: pd.DataFrame df constains test statistics from method
    """
    uni_df = uni_df.sort_values(by='time_value').reset_index(drop=True)
    preds = pd.DataFrame(columns=['pred_total', 'evidence_total'])
    preds.evidence_total = np.convolve((pd.isna(uni_df.value)==False).astype(int),
                           total_window, 'full')[hist:-1*hist]
    preds.pred_total = pd.Series(np.convolve(uni_df.value.fillna(0), total_window,
                       'full')[hist:-1*hist]/preds.evidence_total).replace(np.inf,
                        np.nan).replace(-1*np.inf, np.nan)
    uni_df = uni_df.merge(preds, right_index=True, left_index=True)
    diff_total = uni_df.value - preds.pred_total
    norm_diff_total = (diff_total-diff_total.median())/diff_total.std()
    pop = max(pop, 1)
    uni_df['test_stat_total'] = (norm_diff_total*np.log(uni_df.evidence_total)*np.log(pop)).abs()
    return uni_df


def ewma_per_indicator(indicator_data, history, min_date=None):
    """Compute ewma scores for indicator data.

    Args:
        indicator_data: pd.DataFrame
        Index is datetime range format (no missing dates, but np.nans in table are ok).
        Contains geo_key_id matching with sibling groups,
        which is the geography of the region, and the respective value.
        Geo_key_id description available in 'source/geo_dim_pop.csv' for crosswalk generation.
        history: number of days considered in the indicator data df

    Returns:
        sored_data: pd.DataFrame with the EWMA and optionally Outshines metrics
    """
    mega_list = pd.read_csv('source/siblings_groups.csv', index_col=0).values.tolist()
    return create_ref_df(indicator_data, mega_list, history, min_date)

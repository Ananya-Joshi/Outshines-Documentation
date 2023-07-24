# -*- coding: utf-8 -*-
"""Scoring modules for outlier detection."""
import pandas as pd
import numpy as np


def eval_recent_distribution(i, t_day_num, evd_total, df_ts, day):
    """Create Recent EVD Distribution using recent 28 days.

    Input: i: Indexes to array to identify where where to
                 pull the 28 (t_day_num) days from.
        t_day_num: the number of days in a particular regime
        evd_total: pd.DataFrame of EVD distributions per day
        df_ts: pd.DataFrame of current test statisics to evaluate
        day: day of evaluation
    Output: pd.Series of outshines results for df_ts.
    """
    dist_total = pd.Series()
    j = evd_total.shape[1]-i
    if i < j:
        start_ind = min(i, t_day_num//2)
        dist_total = evd_total.iloc[:, np.r_[i-start_ind:i,
                    i+1:i+(t_day_num-start_ind)]].stack().dropna()
    else:
        end_ind = min(t_day_num//2, j-1)
        dist_total = evd_total.iloc[:, np.r_[i-(t_day_num-end_ind):i, i+1:i+1+end_ind]]
        dist_total = dist_total.stack().dropna()
    series = df_ts.test_stat_total.apply(lambda x: (x > dist_total).sum()/dist_total.shape[0])
    series.name = day
    series.index = df_ts.geo_key_id
    return series, dist_total.shape[0]

def outshines_score(ind_ts,  evd_total, const,t_day_num=28):
    """Evaluate out of sample points to comparison EVD Distribution.

    Args:
        ind_ts: pd.DataFrame of indicator test statistics
        evd_total: pd.DataFrame of EVD distributions per day
        t_day_num: the number of days in a particular regime

    Returns:
        scored_df: pd.DataFrame which contains columns from the Outshines process
    """
    total_list = []
    series = pd.Series()
    for i, (day,df_ts) in enumerate(ind_ts.sort_values(['time_value',
                                    'geo_key_id']).groupby('time_value')):
        df_ts = df_ts.reset_index(drop=True).drop_duplicates('geo_key_id')
        series, dist_shape = eval_recent_distribution(i, t_day_num, evd_total, df_ts, day)
        total_list.append(series)
    total_df = pd.concat(total_list,
                axis=1).unstack().rename('outshines_score').reset_index()
    total_df = total_df.rename(columns={'level_0':'time_value', 'level_1':'geo_key_id'})
    total_df['EVD_total_evidence'] = (np.log(dist_shape)/np.log(const))
    total_df.time_value = pd.to_datetime(total_df.time_value)
    return ind_ts.merge(total_df, on=['time_value', 'geo_key_id'])


def outshines_score_formatter(ts_df):
    """Outshines scoring method.

    Input: ts_df: pd.DataFrame of test statistics
    Output: pd.DataFrame of Outshines Scores
    """
    mega_list = pd.read_csv('source/siblings_groups.csv', index_col=0).values.tolist()
    evd_dist = []
    for i, set_ml in enumerate(pd.Series(mega_list)):
        set_ml = pd.Series(list(set(set_ml))).replace('nan', np.nan).dropna().astype(int)
        set_ml = set_ml[set_ml.isin(ts_df.columns)]
        if len(set_ml) > 0:
            ts_df_stack = ts_df[set_ml].abs().max(axis=1)
            ts_df_stack.name=i
            evd_dist.append(ts_df_stack)
    evd_dist = pd.concat(evd_dist, axis=1)
    ts_df2 = ts_df.stack().reset_index().rename(columns={0:'test_stat_total',
                                                           'level_1':'geo_key_id'})
    ts_df2.time_value = pd.to_datetime(ts_df2.time_value)
    return outshines_score(ts_df2,  evd_dist.T, 28*len(mega_list), t_day_num=28)


def eval_set_siblings(set_ml, test, day):
    """Evaluate one sibling group using set scoring.

    Inputs: set_ml: A set of sibling streams
            test: A pd.Dataframe of the test statistics under consideration.
            day: Day to evaluate.
    Output: pd.Series of set scores from today
    """
    set_ml = pd.Series(list(set(set_ml))).replace('nan', np.nan).dropna().astype(int)
    set_ml = set_ml[set_ml.isin(test.columns)]
    overall_set = test[set_ml]
    ref_test = overall_set.drop(index=day)
    ts_test = overall_set.loc[day, :]
    ser = ts_test.apply(lambda x: (x
                            <=ref_test.stack()).sum()/ref_test.stack().shape[0])
    ser.name = 'set_score'
    return ser

def set_score(ts_df):
    """Set scoring method.

    Input: ts_df: pd.DataFrame of test statistics
    Output: pd.DataFrame of Outshines Scores
    """
    test = ts_df.iloc[-100:, :] #we are only evaluating the top 100 test statistics
    total_score = []
    for day, _ in test.iterrows():
        local_scores = []
        mega_list = pd.read_csv('source/siblings_groups.csv', index_col=0).values.tolist()
        for _, set_ml in enumerate(pd.Series(mega_list)):
            ser = eval_set_siblings(set_ml, test, day)
            local_scores.append(ser)
        ser_df = pd.concat(local_scores).groupby('geo_key_id').mean().reset_index()
        ser_df['time_value'] = day
        total_score.append(ser_df)
    return pd.concat(total_score)

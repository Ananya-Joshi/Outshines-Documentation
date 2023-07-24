# -*- coding: utf-8 -*-
"""Functions to generate Updated FlaSH Results."""

import pandas as pd
import numpy as np
import ruptures as rpt
from scipy.stats import binom
from sklearn.linear_model import LinearRegression

def return_ar_vals(col):
    """Return AR values.

    Args: col: y values for AR
    Return: y, coeffecients, y predictions
    """
    def lin_df(df_in, dummy=False):
        tmp_df = pd.concat([df_in.shift(i) for i in range(0,8)], axis=1).dropna()
        tmp_df.columns = ['y'] + [f'lag_{i}' for i in range(1,8)]
        tmp_df2 = pd.get_dummies(tmp_df.index.day_of_week)
        if dummy:
            tmp_df2.index = tmp_df.index
            tmp_df2.columns = [str(x) for x in tmp_df2.columns]
            tmp_df= pd.concat([tmp_df, tmp_df2], axis=1)
        return tmp_df
    cutoff = 100
    tmp3 = lin_df(col.iloc[:cutoff].to_frame())
    tmp4 = lin_df(col.iloc[cutoff:].to_frame())
    if not(tmp3.empty or tmp4.empty):
        reg = LinearRegression(fit_intercept=False).fit(tmp3.iloc[:, 1:], tmp3.iloc[:, 0])
        y_hat=pd.DataFrame(reg.predict(tmp4.iloc[:, 1:]))
        y_hat.index = tmp4.index
    return tmp4.iloc[:, 0], [1/7]*7, tmp4.iloc[:, 0]

def global_impute(x_str):
    """Return values without global outliers.

    Args: x_str: pd.DataFrame with global outliers
    Returns pd.DataFrame with imptued values
    """
    x_str = x_str.to_frame()
    x_str['diff'] = x_str.diff()
    x_str['weekday'] = x_str.index.weekday
    comp_dict = x_str.groupby('weekday').median()['diff'].to_dict()
    for i, (_, val_x) in enumerate(x_str.iterrows()):
        if np.isnan(val_x.iloc[0]) and i!=0:
            x_str.iloc[i, 0] = x_str.iloc[i-1, 0] + comp_dict[val_x.loc['weekday']]
    return x_str.iloc[:, 0]


def ts_dist(y_in, yhat, pop, log=False):
    """Create test statistic distribution.

    Args:
      y_in: y values for region over time
      yhat: y predictions
      pop: population of region

    Returns:
      ret_bin: pd.DataFrame of the the binomial test statistic
    """
    def ts_dist2(x_loc, y_loc, n_loc):
        """Initialize test statistic distribution which is then vectorized."""
        # print(x, y,  y/n, n, binom.cdf(x, int(n), y/n)
        return binom.cdf(x_loc, int(n_loc), y_loc/n_loc)
    vec_ts_dist = np.vectorize(ts_dist2)
    if log:
        return vec_ts_dist(np.log(y_in+2), np.log(yhat+2), np.log(pd.Series(pop)+2))
    return vec_ts_dist(y_in, yhat, pop)


def create_df_set(set_ml, indicator_df_stack_o):
    """Process data in regimes.

    Input: set_ml: set of geographies
           indicator_df_stack_o: stacked pd.Dataframe of geographies.
    Output: df_set: pd.DataFrame with processed data.
    """
    set_ml = pd.Series(list(set(set_ml))).replace('nan', np.nan).dropna().astype(int)
    set_ml = set_ml[set_ml.isin(indicator_df_stack_o.columns)]
    train_set = indicator_df_stack_o[set_ml]
    if not train_set.empty:
        ref_bktps = list(rpt.Pelt(model='rbf', min_size=28,
                    jump=1).fit(train_set.ffill().bfill().dropna().to_numpy()).predict(pen=10))
        start = 0
        all_bkpts = []
        for bkpt in ref_bktps:
            bkpt_set = train_set.iloc[start:bkpt, :]
            bkpt_set = bkpt_set[~bkpt_set.apply(lambda x: (x-x.mean()/(x.std())),axis=1).gt(3)]
            start = bkpt
            all_bkpts.append(bkpt_set.apply(global_impute))
        df_set = pd.concat(all_bkpts, axis=0)
        return df_set
    return pd.DataFrame()

def flash_per_indicator(indicator_df):
    """Return FlaSH test statistics and outlier scores.

    Args:
    indicator_df: pd.DataFrame
    Return:
    flash_ts : flash test statistics
    """
    indicator_df.time_value = pd.to_datetime(indicator_df.time_value, format='%Y%m%d')
    indicator_df_stack_o=indicator_df.sort_values('time_value')[['geo_key_id','value','time_value']]
    indicator_df_stack_o = indicator_df_stack_o.set_index(['time_value', 'geo_key_id']).unstack()
    indicator_df_stack_o.columns = indicator_df_stack_o.columns.droplevel().dropna().astype(int)
    geo_dim_pop = pd.read_csv('source/geo_dim_pop.csv')
    mega_list = pd.read_csv('source/siblings_groups.csv', index_col=0).values.tolist()
    output_ts = []
    for set_ml in pd.Series(mega_list):
        df_set = create_df_set(set_ml, indicator_df_stack_o)
        if not df_set.empty:
            ar_ret = df_set.apply(return_ar_vals)
            lin_coeff = pd.concat([pd.DataFrame(x) for x in list(ar_ret.iloc[1, :])], axis=1)
            y_hat = pd.concat([pd.DataFrame(x) for x in list(ar_ret.iloc[0, :])], axis=1)
            y_set = pd.concat([pd.DataFrame(x) for x in list(ar_ret.iloc[2, :])], axis=1)
            y_set.columns = df_set.columns
            y_hat.columns =df_set.columns
            lin_coeff.columns = df_set.columns
            geos = y_hat.columns
            pop_list = pd.Series([geo_dim_pop.query('geo_key_id==@x').population.values[0]
                                    for x in geos]).fillna(1)
            if not y_set.empty:
                ret_bin2 = ts_dist(y_set, y_hat, pop_list, log=True)
                ret_df2 = pd.DataFrame(ret_bin2)
                ret_df2.columns = df_set.columns
                ret_df2.index = df_set.index[-1*(ret_df2.shape[0]):]
                ret_df2 = ret_df2.unstack().reset_index()
                ret_df2['signal_key_id'] = indicator_df.iloc[0, :].loc['signal_key_id']
                output_ts.append(ret_df2)
    return pd.concat(output_ts)

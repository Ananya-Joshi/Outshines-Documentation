# -*- coding: utf-8 -*-
"""Using the TODS package to generate test statistics and TODS scores."""
import numpy as np
import pandas as pd
from tods.sk_interface.detection_algorithm.DeepLog_skinterface import DeepLogSKI
from tods.sk_interface.detection_algorithm.Telemanom_skinterface import TelemanomSKI
from tods.sk_interface.detection_algorithm.AutoRegODetector_skinterface \
                                                import AutoRegODetectorSKI
from tods.sk_interface.detection_algorithm.LOF_skinterface import LOFSKI

def tods_per_indicator(indicator_df, method='AR'):
    """Evaluate TODS scoring and test statistic for various methods.

    Args:
        indicator_data: pd.DataFrame
    Returns:
        tot_df: Returning list of Dataframes (one/method) of test
                statistics and TODS Scoring module.
    """
    transformer = {'AR':AutoRegODetectorSKI(window_size=7),
                        'LOF':LOFSKI(),
                        'DL':DeepLogSKI(batch_size=7),
                        'Telemanom':TelemanomSKI(l_s= 7, n_predictions= 1)}[method]
    indicator_df.geo_key_id = indicator_df.geo_key_id.astype(float).astype(int)
    indicator_df.time_value = pd.to_datetime(indicator_df.time_value, format='%Y%m%d')
    tot_df_list = []
    for _, (grp, uni_df) in enumerate(indicator_df.groupby(['signal_key_id', 'geo_key_id'])):
        uni_df = uni_df.sort_values(by='time_value').set_index('time_value')
        if not uni_df.dropna().empty:
            split = indicator_df.time_value.max() - pd.Timedelta('200d')
            train = uni_df.query('time_value < @split').dropna()
            train_np = train.value
            test = uni_df.query('time_value >= @split').dropna()
            test_np = test.value
            if train.empty or test.empty:
                continue
            if test.shape[0] >14 and train.shape[0]>14:
                x_train = np.reshape(train_np.values, (train.shape[0], 1))
                #train up to the last 300 days
                x_test = np.reshape(test_np.values, (test.shape[0], 1))
                #test on the last 100 days
                transformer.fit(x_train)
                uni_df_res = pd.DataFrame()
                uni_df_res['test_stat_total']=transformer.predict_score(x_test).flatten()
                uni_df_res['test_stat_total']=2*(uni_df_res['test_stat_total']-0.5).abs()
                uni_df_res['TODS_score']=(pd.Series(list((transformer.predict(x_test).flatten()))))
                uni_df_res['time_value']=uni_df.index[-1*uni_df_res.shape[0]:]
                uni_df_res['geo_key_id']=grp[1]
                uni_df_res['signal_key_id'] = grp[0]
                uni_df_res = uni_df_res.set_index(['time_value', 'geo_key_id'])
                uni_df_res['method'] = method
                tot_df_list.append(uni_df_res)
    return pd.concat(tot_df_list).reset_index()

# -*- coding: utf-8 -*-
"""Functions to call when running the tool."""
from tqdm import tqdm
import pandas as pd
from ewma_functions import ewma_per_indicator
# from flash_modified_functions import flash_per_indicator
# from tods_functions import tods_per_indicator
from scoring_functions import outshines_score_formatter#, set_score, thresh_score

def run_module():
    """Create ranked values using EWMA + Outshines from data sample.

    input: None
    output: pd.DataFrame with Outshines score and the weight of the score
    """
    #First, generate test statistics
    file_loc="https://delphi-covidcast-public.s3.amazonaws.com/flags-dev/reference_sample_data.csv"
    sample_indicator_data = pd.read_csv(file_loc)
    sample_indicator_data.time_value = pd.to_datetime(sample_indicator_data.time_value,
                                                                        format="%Y%m%d")
    all_uni_ts = []
    for _, indicator_df in sample_indicator_data.groupby('signal_key_id'):
        indicator_df.geo_key_id = indicator_df.geo_key_id.astype(float).astype(int)
        #this test statistic generation function can be changed to any other function.
        uni_ts = ewma_per_indicator(indicator_df, 300, 
                   min_date=sample_indicator_data.time_value.max()-pd.Timedelta('200d'))
        # uni_ts = flash_per_indicator(indicator_df).rename(columns={0:'test_stat_total'})
        # uni_ts = tods_per_indicator(indicator_df)
        all_uni_ts.append(uni_ts)
    ts_results = pd.concat(all_uni_ts)

    #Then, create outshines
    all_outshines = []
    for signal_key, ts_input in ts_results.groupby('signal_key_id'):
        ts_input = ts_input[['time_value', 'geo_key_id',
                'test_stat_total']].drop_duplicates(subset=['geo_key_id',
                'time_value']).sort_values(['geo_key_id',
                'time_value']).set_index(['geo_key_id', 'time_value']).unstack()
        ts_input.columns = ts_input.columns.droplevel()
        out_uni =  outshines_score_formatter(ts_input.T)
        # out_uni =  set_score(ts_input.T)
        # out_uni =  thresh_score(ts_input.T)
        # TODS Produces scores with contamination for thresh parameter & the produce function. 
        out_uni['indicator'] = signal_key
        all_outshines.append(out_uni)
    return pd.concat(all_outshines)

if __name__ == "__main__":
    run_module().to_csv('outshines_output.csv', index=False)
    
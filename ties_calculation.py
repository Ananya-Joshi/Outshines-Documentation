import pandas as pd 
import numpy as np 

def calculate_outshines_ties(df_indicator):
  eval_series = df.no_infer_total* df.EVD_total_evidence
  return (eval_series==eval_series.max()).sum()

def calculate_sibling_ties(df_indicator):
  return df.local_score[df.local_score == df.local_score.max()].shape[0]

def calculate_thresh_ties(df_indicator):
  return df_indicator.query('score==1').shape[0]

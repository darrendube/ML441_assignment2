import numpy as np
import pandas as pd

def continuous_dqr(df):
    cont_dqr_rows = []
    for column in df.columns:
        dqr_dict = {}
        dqr_dict['Feature'] = column
        dqr_dict['Count'] = len(df)
        dqr_dict['% Miss.'] = df[column].isna().sum() / len(df) * 100
        dqr_dict['Card.'] = df[column].nunique()
        dqr_dict['Min.'] = df[column].min()
        dqr_dict['1st Qrt.'] = df[column].quantile(0.25)
        dqr_dict['Mean'] = df[column].mean()
        dqr_dict['Median'] = df[column].median()
        dqr_dict['3rd Qrt.'] = df[column].quantile(0.75)
        dqr_dict['Max.'] = df[column].max()
        dqr_dict['Std. Dev.'] = df[column].std()
        cont_dqr_rows += [dqr_dict]
    return pd.DataFrame(cont_dqr_rows)

def categorical_dqr(df):
    cat_dqr_rows = []
    for column in df.columns:
        dqr_dict = {}
        dqr_dict['Feature'] = column
        dqr_dict['Count'] = len(df)
        dqr_dict['% Miss.'] = df[column].isna().sum() / len(df) * 100
        dqr_dict['Card.'] = df[column].nunique()
        dqr_dict['Mode'] = df[column].value_counts().index[0]
        dqr_dict['Mode Freq.'] = int(df[column].value_counts().iloc[0])
        dqr_dict['Mode %'] = int(df[column].value_counts().iloc[0]) / len(df) * 100
        dqr_dict['2nd Mode'] = df[column].value_counts().index[1] if len(
            df[column].value_counts().index) > 1 else np.nan
        dqr_dict['2nd Mode Freq.'] = int(df[column].value_counts().iloc[1]) if len(
            df[column].value_counts().index) > 1 else np.nan
        dqr_dict['2nd Mode %'] = int(df[column].value_counts().iloc[1]) / len(df) * 100 if len(
            df[column].value_counts().index) > 1 else np.nan
        cat_dqr_rows += [dqr_dict]
    return pd.DataFrame(cat_dqr_rows)

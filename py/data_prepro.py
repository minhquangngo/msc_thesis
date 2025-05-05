import pandas as pd
import matplotlib.pyplot as plt
import numpy  as np

def vol_na_month_plot_bypermno(permno: int, year: int, month: int, df: pd.DataFrame):
    """
    Plot monthly volume of a permo. If it is missing a value then have an 'x' mark
    """
    # Filter 
    mask = (
        (df.index.year == year) &
        (df.index.month == month) &
        (df['permno'] == permno)
    )
    filtered_df = df.loc[mask]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(filtered_df.index, filtered_df['vol'], marker='o', linestyle='-', label='Volume')

    # Highlight missing vol
    missing_vol = filtered_df[filtered_df['vol'].isna()]
    if not missing_vol.empty:
        ax.plot(missing_vol.index, [0] * len(missing_vol), 'rx', markersize=10, label='Missing Volume')

    # format plot
    month_name = pd.Timestamp(year=year, month=month, day=1).strftime("%B %Y")
    ax.set_title(f'Volume for PERMNO {permno} in {month_name}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Volume')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def ret_na_month_plot_bypermno(permno: int, year: int, month: int, df: pd.DataFrame):
    """
    Plot monthly volume of a permo. If it is missing a value then have an 'x' mark
    """
    # Filter 
    mask = (
        (df.index.year == year) &
        (df.index.month == month) &
        (df['permno'] == permno)
    )
    filtered_df = df.loc[mask]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(filtered_df.index, filtered_df['ret'], marker='o', linestyle='-', label='ret')

    # Highlight missing vol
    missing_vol = filtered_df[filtered_df['ret'].isna()]
    if not missing_vol.empty:
        ax.plot(missing_vol.index, [0] * len(missing_vol), 'rx', markersize=10, label='Missing ret')

    # format plot
    month_name = pd.Timestamp(year=year, month=month, day=1).strftime("%B %Y")
    ax.set_title(f'Ret for PERMNO {permno} in {month_name}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Return')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def na_meanfill(param:str, df:pd.DataFrame):
    """
    Imputes missing value with the mean of the day before and after
    From param -> filtered dataset
    identify the rows with missing values in params column. From that row we will make a permno list.
    - for i in  permno -> create a subset with only that permno. 
    Identify the row with the missing value and then identify  the nearest non missing index. Do the same forward
    nonmissing missing nonmissing
    Mean impute missing with mean of nonmissings 
    """
    df_copy = df.copy()
    def impute_series(series:pd.Series) -> pd.Series:
        series = series.sort_index()
        is_na = series[series.isna()].index 
        for i in is_na:
            #get the nearest before
            before = series.loc[:i].dropna()
            before_validation = before.iloc[-1] if not before.empty else None
            #get the nearest after
            after = series.loc[i:].dropna()
            after_validation = after.iloc[0] if not after.empty else None
            # Possible edge case:
            # - Missing value in the first day or last day of the month
            if before_validation is not None and after_validation is not None:
                impute_mean = (before_validation + after_validation)/2
            elif before_validation == None:
                impute_mean = after_validation
            elif after_validation == None:
                impute_mean = before_validation
            else:
                impute_mean = None
            if impute_mean is not None:
                series.at[i] = impute_mean
        return series
    
    for permno in df_copy['permno'].unique():
        mask = df_copy['permno'] == permno
        df_copy.loc[mask, param] = impute_series(df_copy.loc[mask, param])
    return df_copy

def feats_sect(df: pd.DataFrame):
    df_pro = df.copy()
    # liquidity feats
    df_pro.loc[:,'turn_sd'] = df_pro['turn'].std()
    df_pro['sect_mktcap'] = df_pro['prc'] * df_pro['shrout']
    df_pro.loc[:,"mvel1"] = np.log(abs(df_pro['mktcap']))
    df_pro.loc[:,'dolvol'] = df_pro.loc[:,'vol'] *abs(df_pro.loc[:,'prc'])
    #amihud
    df_pro['daily_illq']=(abs(df_pro['ret']) / df_pro['dolvol'])*10**6
    return df_pro

if __name__ == '__main__':
    print("Import to use this module")



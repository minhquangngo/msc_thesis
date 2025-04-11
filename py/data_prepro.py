import pandas as pd
import matplotlib.pyplot as plt

def na_month_plot_bypermno(permno: int, year: int, month: int, df: pd.DataFrame):
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

if __name__ == '__main__':
    print("Import to use this module")


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
    param_df = df_copy[['permno',param]]
    missing_permno = param_df['permno'][param_df[param].isna()].unique()
    for permno in missing_permno:
        filtered_df = param_df[param][param_df['permno'] == permno]


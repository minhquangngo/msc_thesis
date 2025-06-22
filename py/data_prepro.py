import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Union, Dict, List
import warnings

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




# =============================================================
# APA DESCRIPTIVE STATISTICS TABLE GENERATION UTILITIES
# =============================================================

def _escape_latex(text: str) -> str:
    """Escape LaTeX special characters in *text* so that it is safe for LaTeX output."""
    specials = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "\\": r"\textbackslash{}",
    }
    for char, repl in specials.items():
        text = text.replace(char, repl)
    return text


def _compute_descriptive_stats(series: pd.Series) -> Dict[str, Union[int, float]]:
    """Compute count, mean, SD, min, quartiles, and max for *series* (NaNs excluded)."""
    clean = series.dropna()
    return {
        "N": int(clean.shape[0]),
        "Mean": clean.mean(),
        "SD": clean.std(ddof=1),
        "Min": clean.min(),
        "Q1": clean.quantile(0.25),
        "Median": clean.median(),
        "Q3": clean.quantile(0.75),
        "Max": clean.max(),
    }


def _build_apa_table_long(df: pd.DataFrame, var_list: List[str], table_num: int, title: str) -> str:
    """Return a LONG-format APA LaTeX table (rows = statistics, columns = variables)."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Input DataFrame must have a DateTimeIndex.")

    stats_order = ["N", "Mean", "SD", "Min", "Q1", "Median", "Q3", "Max"]

    present_vars: List[str] = []
    missing_vars: List[str] = []
    non_numeric_vars: List[str] = []
    stats_matrix: Dict[str, List[str]] = {stat: [] for stat in stats_order}

    for var in var_list:
        if var not in df.columns:
            missing_vars.append(var)
            continue
        series = df[var]
        if not pd.api.types.is_numeric_dtype(series):
            non_numeric_vars.append(var)
            continue
        present_vars.append(var)
        stats = _compute_descriptive_stats(series)
        for stat in stats_order:
            val = stats[stat]
            formatted = f"{val:.3f}" if stat != "N" else f"{int(val)}"
            stats_matrix[stat].append(formatted)

    # Column spec: stub + n variables
    col_spec = "l" + "c" * len(present_vars)

    lines: List[str] = [
        r"\begin{table}[ht]",
        r"\centering",
        f"\\textbf{{Table {table_num}}}",
        "",
        f"\\textit{{{_escape_latex(title)}}}",
        "",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        "Statistic & " + " & ".join([f"${v}$" for v in present_vars]) + r" \\"
        r"\midrule",
    ]

    for stat in stats_order:
        if not stats_matrix[stat]:
            continue  # skip if no data
        line = _escape_latex(stat) + " & " + " & ".join(stats_matrix[stat]) + r" \\"  # end of row
        lines.append(line)

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    lines.append(r"\end{table}")

    return "\n".join(lines)


def _build_apa_table(df: pd.DataFrame, var_list: List[str], table_num: int, title: str) -> str:
    """Return an APA-formatted LaTeX descriptive statistics table for *df*."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Input DataFrame must have a DateTimeIndex.")

    header_cols = [
        "Variable",
        "N",
        "Mean",
        "SD",
        "Min",
        "Q1",
        "Median",
        "Q3",
        "Max",
    ]
    col_spec = "l" + "c" * (len(header_cols) - 1)  # stub + eight numeric columns

    present_vars: List[str] = []
    missing_vars: List[str] = []
    non_numeric_vars: List[str] = []
    stats_rows: List[Dict[str, Union[int, float]]] = []

    for var in var_list:
        if var not in df.columns:
            missing_vars.append(var)
            continue
        series = df[var]
        if not pd.api.types.is_numeric_dtype(series):
            non_numeric_vars.append(var)
            continue
        present_vars.append(var)
        stats_rows.append(_compute_descriptive_stats(series))

    lines: List[str] = [
        r"\begin{table}[ht]",
        r"\centering",
        f"\\textbf{{Table {table_num}}}",
        "",
        f"\\textit{{{_escape_latex(title)}}}",
        "",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(header_cols) + r" \\",
        r"\midrule",
    ]

    # Body rows
    for var, stats in zip(present_vars, stats_rows):
        formatted_vals = [
            f"{stats['N']}",
            f"{stats['Mean']:.3f}",
            f"{stats['SD']:.3f}",
            f"{stats['Min']:.3f}",
            f"{stats['Q1']:.3f}",
            f"{stats['Median']:.3f}",
            f"{stats['Q3']:.3f}",
            f"{stats['Max']:.3f}",
        ]
        lines.append(f"${var}$ & " + " & ".join(formatted_vals) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}"])

    lines.append(r"\end{table}")

    return "\n".join(lines)


def _chunk(seq: List[str], size: int) -> List[List[str]]:
    """Split *seq* into successive chunks of length *size*."""
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def apa_descriptive_table(
    df_or_dict: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    var_list: List[str],
    table_start: int = 1,
    title_prefix: str = "Descriptive Statistics",
    max_cols: int = 6,
) -> str:
    """Generate APA-style LaTeX descriptive statistics table(s).

    Parameters
    ----------
    df_or_dict : Union[pd.DataFrame, dict[str, pd.DataFrame]]
        Single DataFrame or dictionary of DataFrames.
    var_list : list[str]
        List of variables for which to compute statistics.
    table_start : int, optional
        Starting table number (default 1).
    title_prefix : str, optional
        Base title for the table(s).

    Returns
    -------
    str
        One or more LaTeX tables separated by blank lines.
    """
    if not isinstance(var_list, list) or not all(isinstance(v, str) for v in var_list):
        raise TypeError("var_list must be a list of strings.")

    tables: List[str] = []
    table_num = table_start

    if isinstance(df_or_dict, pd.DataFrame):
        chunks = _chunk(var_list, max_cols)
        for idx, subset in enumerate(chunks, start=1):
            suffix = f" ({idx}/{len(chunks)})" if len(chunks) > 1 else ""
            tables.append(
                _build_apa_table_long(
                    df_or_dict,
                    subset,
                    table_num,
                    f"{title_prefix}{suffix}",
                )
            )
            table_num += 1
    elif isinstance(df_or_dict, dict):
        for key, df in df_or_dict.items():
            if not isinstance(df, pd.DataFrame):
                warnings.warn(f"Value for key '{key}' is not a DataFrame and was skipped.")
                continue
            chunks = _chunk(var_list, max_cols)
            for idx, subset in enumerate(chunks, start=1):
                suffix = f" ({idx}/{len(chunks)})" if len(chunks) > 1 else ""
                title = f"{title_prefix} for Sector: {key}{suffix}"
                tables.append(
                    _build_apa_table_long(df, subset, table_num, title)
                )
                table_num += 1
    else:
        raise TypeError("df_or_dict must be a DataFrame or a dictionary of DataFrames.")

    return "\n\n".join(tables)

# -------------------------------------------------------------
# CLASS-BASED INTERFACE
# -------------------------------------------------------------

class APADescriptiveStats:
    """Generate APA-style LaTeX descriptive statistics tables.

    Example
    -------
    >>> latex_code = APADescriptiveStats(df, ["returns", "volatility"]).generate()
    >>> print(latex_code)
    """

    def __init__(
        self,
        df_or_dict: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        var_list: List[str],
        table_start: int = 1,
        title_prefix: str = "Descriptive Statistics",
        max_cols: int = 6,
    ) -> None:
        self._df_or_dict = df_or_dict
        self._var_list = var_list
        self._table_start = table_start
        self._title_prefix = title_prefix
        self._max_cols = max_cols

    # -----------------------------------------------------------------
    # Public helpers
    # -----------------------------------------------------------------
    def generate(self) -> str:
        """Return APA-styled LaTeX table(s) for the stored data."""
        return apa_descriptive_table(
            self._df_or_dict,
            self._var_list,
            table_start=self._table_start,
            title_prefix=self._title_prefix,
            max_cols=self._max_cols,
        )

    __call__ = generate  # Allow instance() to behave like generate()

    def __str__(self) -> str:  # pragma: no cover
        return self.generate()

# =============================================================

if __name__ == '__main__':
    print("Import to use this module")



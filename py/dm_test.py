import os
import pandas as pd
import numpy as np
from pathlib import Path

def dm_test(actual_lst, pred1_lst, pred2_lst, h = 1, crit="MSE", power = 2):
    # Routine for checking errors
    def error_check():
        rt = 0
        msg = ""
        # Check if h is an integer
        if (not isinstance(h, int)):
            rt = -1
            msg = "The type of the number of steps ahead (h) is not an integer."
            return (rt,msg)
        # Check the range of h
        if (h < 1):
            rt = -1
            msg = "The number of steps ahead (h) is not large enough."
            return (rt,msg)
        len_act = len(actual_lst)
        len_p1  = len(pred1_lst)
        len_p2  = len(pred2_lst)
        # Check if lengths of actual values and predicted values are equal
        if (len_act != len_p1 or len_p1 != len_p2 or len_act != len_p2):
            rt = -1
            msg = "Lengths of actual_lst, pred1_lst and pred2_lst do not match."
            return (rt,msg)
        # Check range of h
        if (h >= len_act):
            rt = -1
            msg = "The number of steps ahead is too large."
            return (rt,msg)
        # Check if criterion supported
        if (crit != "MSE" and crit != "MAPE" and crit != "MAD" and crit != "poly"):
            rt = -1
            msg = "The criterion is not supported."
            return (rt,msg)
        # Check if every value of the input lists are numerical values
        def is_numeric(val):
            """ Returns True if value is numeric (int, float, or numeric string). """
            try:
                float(val)
                return True
            except (ValueError, TypeError):
                return False

        for actual, pred1, pred2 in zip(actual_lst, pred1_lst, pred2_lst):
            if not (is_numeric(actual) and is_numeric(pred1) and is_numeric(pred2)):
                msg = "An element in the actual_lst, pred1_lst or pred2_lst is not numeric."
                rt = -1
                return (rt,msg)
        return (rt,msg)
    
    # Error check
    error_code = error_check()
    # Raise error if cannot pass error check
    if (error_code[0] == -1):
        raise SyntaxError(error_code[1])
        return
    # Import libraries
    from scipy.stats import t
    import collections
    import pandas as pd
    import numpy as np
    
    # Initialise lists
    e1_lst = []
    e2_lst = []
    d_lst  = []
    
    # convert every value of the lists into real values
    actual_lst = pd.Series(actual_lst).apply(lambda x: float(x)).tolist()
    pred1_lst = pd.Series(pred1_lst).apply(lambda x: float(x)).tolist()
    pred2_lst = pd.Series(pred2_lst).apply(lambda x: float(x)).tolist()
    
    # Length of lists (as real numbers)
    T = float(len(actual_lst))
    
    # construct d according to crit
    if (crit == "MSE"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append((actual - p1)**2)
            e2_lst.append((actual - p2)**2)
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif (crit == "MAD"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append(abs(actual - p1))
            e2_lst.append(abs(actual - p2))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif (crit == "MAPE"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append(abs((actual - p1)/actual))
            e2_lst.append(abs((actual - p2)/actual))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif (crit == "poly"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append(((actual - p1))**(power))
            e2_lst.append(((actual - p2))**(power))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)    
    
    # Mean of d        
    mean_d = pd.Series(d_lst).mean()
    
    # Find autocovariance and construct DM test statistics
    def autocovariance(Xi, N, k, Xs):
        autoCov = 0
        T = float(N)
        for i in np.arange(0, N-k):
              autoCov += ((Xi[i+k])-Xs)*(Xi[i]-Xs)
        return (1/(T))*autoCov
    gamma = []
    for lag in range(0,h):
        gamma.append(autocovariance(d_lst,len(d_lst),lag,mean_d)) # 0, 1, 2
    V_d = (gamma[0] + 2*sum(gamma[1:]))/T
    DM_stat=V_d**(-0.5)*mean_d
    harvey_adj=((T+1-2*h+h*(h-1)/T)/T)**(0.5)
    DM_stat = harvey_adj*DM_stat
    # Find p-value
    p_value = 2*t.cdf(-abs(DM_stat), df = T - 1)
    
    # Construct named tuple for return
    dm_return = collections.namedtuple('dm_return', 'DM p_value')
    
    rt = dm_return(DM = DM_stat, p_value = p_value)

    return rt


def compare_experiments_dm_test(actual_lst, experiment1_num, experiment2_num):
    """
    Perform Diebold-Mariano tests comparing predictions from two different ML experiments.

    Parameters:
    -----------
    actual_lst : dict
        Dictionary where keys are sector numbers as strings ('10', '20', etc.)
        and values are pandas DataFrames with 'excess_ret' column and datetime index.
        This is the df_dict containing actual values.
    experiment1_num : int
        First experiment number to compare
    experiment2_num : int
        Second experiment number to compare

    Returns:
    --------
    str
        APA-formatted LaTeX table string with DM test results
    """

    # Rename for clarity - actual_lst is actually df_dict
    df_dict = actual_lst

    # Date filter - remove observations before 1999-06-16
    cutoff_date = pd.Timestamp('1999-06-16')

    # Helper function to extract sector number from sector parameter
    def extract_sector_number(sector_param):
        """Extract numeric sector from strings like '60_rf' -> '60'"""
        if isinstance(sector_param, str):
            # Extract the numeric part before the first underscore
            return sector_param.split('_')[0]
        return str(sector_param)

    # Helper function to load predictions for an experiment
    def load_experiment_predictions(experiment_num):
        """Load all prediction series for a given experiment"""
        predictions = {}

        # Path to experiment artifacts
        artifacts_path = Path(f"mlartifacts/{experiment_num}")
        mlruns_path = Path(f"mlruns/{experiment_num}")

        if not artifacts_path.exists() or not mlruns_path.exists():
            print(f"Warning: Experiment {experiment_num} not found")
            return predictions

        # Iterate through all runs in the experiment
        for run_dir in artifacts_path.iterdir():
            if run_dir.is_dir():
                run_id = run_dir.name

                # Get sector number from mlruns params
                sector_param_file = mlruns_path / run_id / "params" / "sector"
                if sector_param_file.exists():
                    with open(sector_param_file, 'r') as f:
                        sector_param = f.read().strip()  # e.g., "60_rf"

                    # Extract sector number (e.g., "60" from "60_rf")
                    sector_num = extract_sector_number(sector_param)

                    # Look for RF prediction series file (using glob pattern)
                    artifacts_dir = run_dir / "artifacts"
                    pred_files = list(artifacts_dir.glob("rf_pred_series_*.csv"))

                    if pred_files:
                        # Take the first matching file
                        pred_file = pred_files[0]
                        try:
                            pred_df = pd.read_csv(pred_file, index_col=0, parse_dates=True)
                            print(f"Prediction file shape for sector {sector_num}: {pred_df.shape}, columns: {pred_df.columns.tolist()}")
                            # Filter out observations before cutoff date
                            pred_df = pred_df[pred_df.index >= cutoff_date]
                            predictions[sector_num] = pred_df
                            print(f"Loaded predictions for sector {sector_num} from experiment {experiment_num} (after {cutoff_date.date()})")
                        except Exception as e:
                            print(f"Error loading {pred_file}: {e}")
                    else:
                        print(f"No prediction file found for sector {sector_num} (param: {sector_param}) in run {run_id}")

        return predictions

    # Helper function to format significance stars
    def format_significance(p_value):
        """Format p-value with significance stars"""
        if p_value <= 0.001:
            return "***"
        elif p_value <= 0.01:
            return "**"
        elif p_value <= 0.05:
            return "*"
        else:
            return ""

    # Load predictions from both experiments
    print(f"Loading predictions from experiment {experiment1_num}...")
    exp1_predictions = load_experiment_predictions(experiment1_num)

    print(f"Loading predictions from experiment {experiment2_num}...")
    exp2_predictions = load_experiment_predictions(experiment2_num)

    # Find common sectors between experiments and df_dict
    common_sectors = set(exp1_predictions.keys()) & set(exp2_predictions.keys()) & set(df_dict.keys())

    if not common_sectors:
        print("No common sectors found between experiments and data")
        return ""

    print(f"Found {len(common_sectors)} common sectors: {sorted(common_sectors)}")

    # Perform DM tests for each common sector
    dm_results = {}

    for sector in sorted(common_sectors):
        try:
            # Get actual values and filter by date
            actual_df = df_dict[sector]
            print(f"Actual df shape for sector {sector}: {actual_df.shape}, columns: {actual_df.columns.tolist()}")
            actual_values = actual_df['excess_ret']
            print(f"Actual values dtype: {actual_values.dtype}, sample values: {actual_values.head().tolist()}")
            # Filter out observations before cutoff date
            actual_values = actual_values[actual_values.index >= cutoff_date]
            print(f"Actual values after date filter: {len(actual_values)} observations")

            # Get predictions from both experiments (already filtered in load_experiment_predictions)
            pred1_df = exp1_predictions[sector]
            pred2_df = exp2_predictions[sector]

            # Align data by index (time)
            # Find common time indices
            common_index = actual_values.index.intersection(pred1_df.index).intersection(pred2_df.index)

            if len(common_index) < 10:  # Need minimum observations for DM test
                print(f"Skipping sector {sector}: insufficient overlapping observations ({len(common_index)})")
                continue

            # Extract aligned data
            actual_aligned = actual_values.loc[common_index]
            print(f"Actual aligned dtype: {actual_aligned.dtype}, sample values: {actual_aligned.head().tolist()}")

            # Check prediction file structure and extract predictions
            print(f"Pred1 shape: {pred1_df.shape}, columns: {pred1_df.columns.tolist()}")
            print(f"Pred2 shape: {pred2_df.shape}, columns: {pred2_df.columns.tolist()}")

            # Since prediction files have only 1 column, use the first (and only) column
            pred1_aligned = pred1_df.loc[common_index].iloc[:, 0]  # First column
            pred2_aligned = pred2_df.loc[common_index].iloc[:, 0]  # First column

            print(f"Pred1 aligned dtype: {pred1_aligned.dtype}, sample values: {pred1_aligned.head().tolist()}")
            print(f"Pred2 aligned dtype: {pred2_aligned.dtype}, sample values: {pred2_aligned.head().tolist()}")

            # Convert to lists and ensure numeric values
            try:
                actual_list = pd.to_numeric(actual_aligned, errors='coerce').dropna().astype(float).tolist()
                pred1_list = pd.to_numeric(pred1_aligned, errors='coerce').dropna().astype(float).tolist()
                pred2_list = pd.to_numeric(pred2_aligned, errors='coerce').dropna().astype(float).tolist()

                # Ensure all lists have the same length after dropping NaNs
                min_length = min(len(actual_list), len(pred1_list), len(pred2_list))
                actual_list = actual_list[:min_length]
                pred1_list = pred1_list[:min_length]
                pred2_list = pred2_list[:min_length]

                print(f"Final data lengths - Actual: {len(actual_list)}, Pred1: {len(pred1_list)}, Pred2: {len(pred2_list)}")

            except Exception as e:
                print(f"Error converting to numeric for sector {sector}: {e}")
                continue

            # Perform DM test
            dm_result = dm_test(
                actual_lst=actual_list,
                pred1_lst=pred1_list,
                pred2_lst=pred2_list,
                h=1,
                crit="MSE"
            )

            dm_results[sector] = {
                'dm_stat': dm_result.DM,
                'p_value': dm_result.p_value,
                'n_obs': len(common_index)
            }

            print(f"Sector {sector}: DM={dm_result.DM:.4f}, p={dm_result.p_value:.4f}, n={len(common_index)}")

        except Exception as e:
            print(f"Error processing sector {sector}: {e}")
            continue

    # Generate LaTeX table
    if not dm_results:
        print("No DM test results to generate table")
        return ""

    # Create sector name mapping (you can customize this)
    sector_names = {
        '10': 'Energy',
        '15': 'Materials',
        '20': 'Industrials',
        '25': 'Consumer Discretionary',
        '30': 'Consumer Staples',
        '35': 'Health Care',
        '40': 'Financials',
        '45': 'Information Technology',
        '50': 'Communication Services',
        '55': 'Utilities',
        '60': 'Real Estate'
    }

    # Start building LaTeX table
    latex_lines = [
        "\\begin{table}[H]",
        "\\centering",
        f"\\caption{{Diebold-Mariano Test Results: Experiment {experiment1_num} vs Experiment {experiment2_num}}}",
        f"\\label{{tab:dm_test_exp{experiment1_num}_vs_exp{experiment2_num}}}",
        "\\begin{tabular}{lc}",
        "\\toprule",
        "Sector & DM Statistic \\\\",
        "\\midrule"
    ]

    # Add results for each sector (sorted by sector number)
    for sector in sorted(dm_results.keys(), key=lambda x: int(x)):
        result = dm_results[sector]
        sector_name = sector_names.get(sector, f"Sector {sector}")

        # Format DM statistic with significance stars
        dm_stat = result['dm_stat']
        p_val = result['p_value']
        stars = format_significance(p_val)

        dm_formatted = f"{dm_stat:.3f}{stars}"

        latex_lines.append(f"{sector_name} & {dm_formatted} \\\\")

    # Close table
    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
        "% Significance levels: *** p ≤ 0.001, ** p ≤ 0.01, * p ≤ 0.05"
    ])

    latex_table = "\n".join(latex_lines)

    print(f"\nGenerated LaTeX table with {len(dm_results)} sectors")
    print("DM Test Interpretation:")
    print("- Positive DM statistic: Experiment 1 has higher forecast errors (worse)")
    print("- Negative DM statistic: Experiment 1 has lower forecast errors (better)")
    print("- Significance stars indicate statistical significance of the difference")

    return latex_table
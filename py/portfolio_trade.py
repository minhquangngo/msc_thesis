"""
- Load the signals in, group by models. Example for run X, collect all sectors
10->60.
- Go over day by day of 2018, define the weights of the  port
    - Porfolio construction: equal portfolio weights. Meaning if there are 
    no signals on a certain day, then weights is distributed across all sectors.
    - But if, for example, 3 sectors have signals, the weights are 1/3 for each of those sectors.


"""
import os
import glob
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from typing import Dict, List
from sector_rot import all_runs


ROOT = Path(__file__).resolve().parent.parent
cols = ['rf_base_signal','ols_enhanced_signal','rf_enhanced_signal','ols_base_signal']


class weighted_portfolio_returns:
    def __init__(self, weight_mode: str | None = None):
        self.sector_dataframes = self._load_org_df()
        self.weight_mode = weight_mode
        self.weighted_df = weighted_ret(self.sector_dataframes, self.weight_mode).calc_weighted_ret()
    
    def _weighted_portfolios(self):
        daily_ret_ols_base = [] 
        daily_ret_ols_enhanced = []
        daily_ret_rf_base = []
        daily_ret_rf_enhanced = []
        numb_rows = len(self.weighted_df['10'])
        for i in range(numb_rows):
            ols_weighted_ret_list = []
            ols_enhanced_weighted_ret_list = []
            rf_weighted_ret_list = []
            rf_enhanced_weighted_ret_list = []
            for sector, weighted_df in self.weighted_df.items():    
                ols_weighted_ret_list.append(weighted_df.iloc[i]['Excess_returns_weighted_ols_base'])
                ols_enhanced_weighted_ret_list.append(weighted_df.iloc[i]['Excess_returns_weighted_ols_enhanced'])
                rf_weighted_ret_list.append(weighted_df.iloc[i]['Excess_returns_weighted_rf_base'])
                rf_enhanced_weighted_ret_list.append(weighted_df.iloc[i]['Excess_returns_weighted_rf_enhanced'])
            daily_ret_ols_base.append(sum(ols_weighted_ret_list))
            daily_ret_ols_enhanced.append(sum(ols_enhanced_weighted_ret_list))
            daily_ret_rf_base.append(sum(rf_weighted_ret_list))
            daily_ret_rf_enhanced.append(sum(rf_enhanced_weighted_ret_list))            
        return daily_ret_ols_base, daily_ret_ols_enhanced, daily_ret_rf_base, daily_ret_rf_enhanced

    def _load_org_df(self):
        data_dir = Path('data')
        print(f"Data directory: {data_dir}")
        df_dict = {
            file.stem.replace("sector_","") : pd.read_parquet(file)
            for file in data_dir.glob("sector_*.parquet")
        }
        return df_dict

class weighted_ret:
    """Add weighted excess return columns to sector dataframes based on sector-level model weights."""

    MODEL_COLS = [
        'Excess_returns_weighted_rf_base',
        'Excess_returns_weighted_ols_enhanced',
        'Excess_returns_weighted_rf_enhanced',
        'Excess_returns_weighted_ols_base'
    ]

    def __init__(self, sector_dataframes: dict, weight_mode: str | None = None, model_weights: list | None = None):
        """Create a new weighted_ret helper.

        Parameters
        ----------
        sector_dataframes : dict
            Mapping ``{sector_id: DataFrame}`` where the index represents
            time periods and columns hold excess returns of assets.
        model_weights : list | None, optional
            Nested structure ``[time][model_idx][sector_id] -> weight``. If
            *None* the helper :pyclass:`weights` is invoked to calculate the
            weights automatically.
        """
        _,self.sector_dataframes = prep_weights().collect_matched_df()
        if model_weights is None:
            self.model_weights = weights(weight_mode).calc_weights()
        else:
            self.model_weights = model_weights

        # Basic validation – every slice should contain exactly 4 model specs
        for slice_ in self.model_weights:
            if len(slice_) != 4:
                raise ValueError(
                    "Each time slice in model_weights must have 4 model specification dictionaries"
                )

    def calc_weighted_ret(self) -> dict:
        """Add weighted excess return columns to every sector dataframe.

        The operation is performed *in-place* and the same mapping is returned
        for convenience.
        """
        for sector, df in self.sector_dataframes.items():
            if df.empty:
                # Skip sectors without data
                continue

            # Preserve original numeric columns (the excess-return columns)
            numeric_cols = df.select_dtypes(include="number").columns.tolist()

            # Ensure the 4 new columns exist
            for col in self.MODEL_COLS:
                if col not in df.columns:
                    df[col] = np.nan

            # Iterate through each time index
            for t_idx, timestamp in enumerate(df.index):
                if t_idx >= len(self.model_weights):
                    # No weight information for this timestamp
                    continue

                for model_idx, col_name in enumerate(self.MODEL_COLS):
                    # Retrieve weight, tolerating str/int key mismatch
                    weight_val = self.model_weights[t_idx][model_idx].get(str(sector),
                                   self.model_weights[t_idx][model_idx].get(int(sector), 0.0))

                    if weight_val == 0.0:
                        df.at[timestamp, col_name] = 0.0
                        continue

                    # Use provided 'ret' column if present; otherwise fallback to mean of numeric columns
                    if 'ret' in df.columns:
                        ret = df.at[timestamp, 'ret']
                    else:
                        ret = df.loc[timestamp, numeric_cols].mean()
                    df.at[timestamp, col_name] = ret * weight_val

        return self.sector_dataframes

class weights:
    """Calculate sector-level portfolio weights according to a chosen scheme.

    Parameters
    ----------
    weight_mode : {'equal_weight', 'unconstrained', 'constrained'}
        Determines how the sector weights are computed.
    """

    ALLOWED_MODES = {"equal_weight", "unconstrained", "constrained"}

    def __init__(self, weight_mode: str = "unconstrained"):
        # Validate weight mode first
        if weight_mode not in self.ALLOWED_MODES:
            raise ValueError(
                f"weight_mode must be 'equal_weight', 'unconstrained', or 'constrained'; got {weight_mode!r}"
            )
        self.weight_mode = weight_mode

        # Pre-compute signal structure [time][model][sector -> signal].
        self.prepped_weights = prep_weights().get_signals_all_times_all_models()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _calc_unconstrained(self):
        """Original (legacy) weight calculation – keep for re-use."""
        all_time_weights = []
        for time_t in self.prepped_weights:
            weights_across_model_spec_time_t = []
            for model_spec in time_t:
                # Count active signals (== 1.0)
                active_count = sum(1 for sig in model_spec.values() if sig == 1.0)

                indiv_weights = {}
                for sect, sig in model_spec.items():
                    if active_count:
                        indiv_weights[sect] = 1 / active_count if sig == 1.0 else 0.0
                    else:
                        # No signals – spread weight equally across all sectors
                        indiv_weights[sect] = 1 / len(model_spec)

                weights_across_model_spec_time_t.append(indiv_weights)
            all_time_weights.append(weights_across_model_spec_time_t)
        return all_time_weights

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def calc_weights(self):
        """Return weights according to *self.weight_mode*."""
        unconstrained_weights = self._calc_unconstrained()

        # Fast-path for legacy behaviour
        if self.weight_mode == "unconstrained":
            return unconstrained_weights

        transformed = []
        for time_slice in unconstrained_weights:
            transformed_models = []
            for weight_dict in time_slice:
                n = len(weight_dict)
                if self.weight_mode == "equal_weight":
                    new_dict = {sect: 1.0 / n for sect in weight_dict}
                else:  # self.weight_mode == "constrained"
                    new_dict = {sect: w * 0.30 + (0.70 / n) for sect, w in weight_dict.items()}
                transformed_models.append(new_dict)
            transformed.append(transformed_models)
        return transformed
                
                    
class prep_weights:
    """
    the dataframe that is unpacked here is: {sector 10 :df, sector 15:df, ...}

    _dict_sect_listsig returns (sect10: [[signals for all 4 models spec at index 0],[signals for all 4 models spec at index 1])
    """
    def __init__(self):
        self.sect_sig_dict,_,_ = self._dict_sect_listsig()

    def _dict_sect_listsig(self):
        matched_df_dict, matched_df_dict_2018 = self.collect_matched_df()
        sect_sig_dict = {}
        # at the end we should have the signal at all index, for all sector, for all model spec, 
        for sect, df in matched_df_dict_2018.items():
            df_index_list = [] 
            # at the end we should have the signal at all index, for sector sect for all model spec, 
            for index in range(len(df)):
                time_t_sig_allmodelspec = [] 
                # at the end we should have the signal at index t , for sector sect,for all model spec 
                for model_spec in cols:
                    row_sig = df.iloc[index][model_spec] #at time index 0, get the speciifc model spec signal
                    time_t_sig_allmodelspec.append(row_sig) # append signal at specific time
                df_index_list.append(time_t_sig_allmodelspec)
            sect_sig_dict[sect] = df_index_list
        return sect_sig_dict , df_index_list, time_t_sig_allmodelspec
                
    def collect_matched_df(self):
        matched_df_dict = {}
        matched_df_dict_2018 = {}
        for sect in range(10,65,5):
            init_df = matching_df(sector=str(sect)).fit()
            matched_df_dict[str(sect)] = init_df
            matched_df_dict_2018[str(sect)] = init_df[init_df.index.year == 2018]
        return matched_df_dict, matched_df_dict_2018

    def get_signals_at_time(self, time_index: int):
        """Return sector signals for a specific time index.

        Parameters
        ----------
        time_index : int
            The time period to extract.

        Returns
        -------
        dict
            Mapping of ``{sector_name: signal_list}`` where ``signal_list`` is the
            raw list stored at ``self.sect_sig_dict[sector][time_index]``.  If a
            sector does not have the requested index the key is omitted.  If no
            sector contains the index an empty dictionary is returned.
        """
        if not isinstance(time_index, int):
            raise TypeError("time_index must be an int")

        signals_at_time: dict = {}
        for sector, series in self.sect_sig_dict.items():
            if 0 <= time_index < len(series):
                signals_at_time[sector] = series[time_index]
        return signals_at_time

    def get_signals_all_models_at_time(self, time_index: int):
        """Return signals for *all* model specifications at a given time.

        The output structure is a list whose *i*-th element is a dictionary
        mapping sector names to the signal list produced by model specification
        *i* at ``time_index``.

        Example (2 sectors, 2 model specifications)::

            [
                {"sector1": [0, 1], "sector2": [1, 0]},
                {"sector1": [1, 0], "sector2": [0, 1]}
            ]
        """
        signals_at_time = self.get_signals_at_time(time_index)
        if not signals_at_time:
            return []

        # Determine the number of model specifications from the first sector
        first_sector_signals = next(iter(signals_at_time.values()))
        num_model_specs = len(first_sector_signals)

        # Initialise a list of dicts, one per model specification
        models_output = [dict() for _ in range(num_model_specs)]

        for sector, model_sig_list in signals_at_time.items():
            # model_sig_list should contain one element per model specification
            for model_idx in range(min(num_model_specs, len(model_sig_list))):
                models_output[model_idx][sector] = model_sig_list[model_idx]

        return models_output

    def get_signals_all_times_all_models(self):
        """Return the full 3-D nested signal structure ``[time][model][sector_dict]``.

        The helper :py:meth:`get_signals_all_models_at_time` is applied for every
        available time index and the resulting list is returned.  If
        ``self.sect_sig_dict`` is empty an empty list is returned.
        """
        if not self.sect_sig_dict:
            return []

        first_sector_series = next(iter(self.sect_sig_dict.values()))
        total_time_periods = len(first_sector_series)

        return [self.get_signals_all_models_at_time(t) for t in range(total_time_periods)]
    
class matching_df: 
    def __init__(self, sector):
        self.sector = sector

    def fit(self):
        sig_sector_matching_run, df_sector_matching_run, df_dict_exp_run, sig_dict_exp_run = self._load_all_signals()
        sector_df = self._load_org_df()

        print("\n")
        print("======================")
        print("Signals")
        print(sig_sector_matching_run)
        print("======================")
        print("Data")
        print(df_sector_matching_run)
        print("======================")

        for each_dict in sig_sector_matching_run:
            print(f"\n")
            print(f'sector_and_run: {each_dict}')

            for run, sect in each_dict.items():
                # Find the experiment that contains this run
                for experiment, runs_list in sig_dict_exp_run.items():
                    if run in runs_list:
                        experiment_numb = experiment
                        break
                    print(f"We are now on experiment{experiment_numb}")
                sig_mlartifact_run_path = os.path.join("mlartifacts", str(experiment_numb), str(run), "artifacts")
                print(f"Signal ml artifact path: {sig_mlartifact_run_path}")
                if "_rf" in sect:
                    signal_csv_path = os.path.join(sig_mlartifact_run_path, "rf_signal_*.csv")
                    print(f"Signal csv path: {signal_csv_path}")
                else:
                    signal_csv_path = os.path.join(sig_mlartifact_run_path, "ols_signal_*.csv")
                    print(f"Signal csv path: {signal_csv_path}")
                # Find all matching signal files
                signal_files = glob.glob(signal_csv_path)
                if signal_files:
                    # Load the most recent signal file
                    signal_df = pd.read_csv(signal_files[-1])
                    # Convert first column to datetime index
                    signal_df.iloc[:, 0] = pd.to_datetime(signal_df.iloc[:, 0])
                    signal_df.set_index(signal_df.columns[0], inplace=True)
                    # Join with sector_df
                    sector_df = sector_df.join(signal_df, how='left')
                    if "_rf" in sect:
                        if "_enhanced" in sect:
                            sector_df.rename(columns={sector_df.columns[-1]: "rf_enhanced_signal"}, inplace=True)
                            print(f"Renamed column to rf_enhanced_signal")
                        else:
                            sector_df.rename(columns={sector_df.columns[-1]: "rf_base_signal"}, inplace=True)
                            print(f"Renamed column to rf_base_signal")
                    else:
                        if "_enhanced" in sect:
                            sector_df.rename(columns={sector_df.columns[-1]: "ols_enhanced_signal"}, inplace=True)
                            print(f"Renamed column to ols_enhanced_signal")
                        else:
                            sector_df.rename(columns={sector_df.columns[-1]: "ols_base_signal"}, inplace=True)
                            print(f"Renamed column to ols_base_signal")
                else:
                    print(f"No signal files found matching pattern: {signal_csv_path}")

        
        return sector_df
                                


    def _load_all_signals(self):
        sig_sector_matching_run, df_sector_matching_run, df_dict_exp_run, sig_dict_exp_run = match_sig_ret(self.sector).fit()
        return sig_sector_matching_run, df_sector_matching_run, df_dict_exp_run, sig_dict_exp_run
        
        
    
    def _load_org_df(self):
        data_dir = Path('data')
        print(f"Data directory: {data_dir}")
        df_dict = {
            file.stem.replace("sector_","") : pd.read_parquet(file)
            for file in data_dir.glob("sector_*.parquet")
        }
        sector_df = df_dict[self.sector]
        return sector_df



class match_sig_ret:
    """
    Load in df -> extrac the sector key 
    Load in experiment -> load run -> extract sector from /params/sector -> 
    mark runs as duplicates -> delete dups
    
    Load in experiment -> load run -> extract sector from /params/sector 

    load signals in matching to their sector key 

    return dict of signals (one set of sig for each sect) with their sector key 
    and df with their sector key
    
    Each df sector gets 4 columns of signals
 """
    def __init__(
            self,
            sector = None
            ):
        self.sector = sector

        self.sig_dict_model_exp = self._dict_model_exp(data_not_signal=False)
        self.df_dict_model_exp = self._dict_model_exp(data_not_signal=True)    
        self.df_dict_exp_run, self.sig_dict_exp_run = self._dict_exp_run() 

    def fit(self):
        sig_run_sect_dict, df_run_sect_dict = self._dict_run_sect()
        sig_sector_matching_run =[]
        df_sector_matching_run =[]
        for run, sect in sig_run_sect_dict.items():
            if self.sector in sect:
                sig_sector_matching_run.append({run:sect})
        for run, sect in df_run_sect_dict.items():
            if self.sector in sect:
                df_sector_matching_run.append({run:sect})
        return sig_sector_matching_run, df_sector_matching_run, self.df_dict_exp_run, self.sig_dict_exp_run
        
                

    def _dict_run_sect(self):
        """
        take _dict_exp_run and _dict_model_exp and get a dictionary of the sector that matches the run{run:sector}
        """
        sig_run_sect_dict = {}
        df_run_sect_dict = {}
        for experiment, run in self.sig_dict_exp_run.items():
            #Go into run folder
            for indiv_run in run:
                sector_path = os.path.join("mlruns",str(experiment),str(indiv_run),"params","sector")
                print(f"Signal sector path: {sector_path}")
                #pull sector from path
                try:
                    if os.path.exists(sector_path):
                        with open(sector_path,'r') as f:
                            sectors = f.read()
                        print(f"Sectors loaded: {sectors}")
                    else:
                        print("Sectors file not found")
                        sectors = None
                except Exception as e:
                    print(f"Error loading sector/factors: {e}")
                    sectors = None
                sig_run_sect_dict[indiv_run] = sectors
        for experiment, run in self.df_dict_exp_run.items():
            #Go into run folder
            for indiv_run in run:
                sector_path = os.path.join("py","mlruns",str(experiment),str(indiv_run),"params","sector")
                print(f"Data sector path: {sector_path}")
                #pull sector from path
                try:
                    if os.path.exists(sector_path):
                        with open(sector_path, 'r') as f:
                            sectors = f.read()
                        print(f"Sectors loaded: {sectors}")
                    else:
                        print("Sectors file not found")
                        sectors = None
                except Exception as e:
                    print(f"Error loading sector/factors: {e}")
                    sectors = None
                df_run_sect_dict[indiv_run] = sectors
        return sig_run_sect_dict, df_run_sect_dict



    def _dict_exp_run(self):
        """
        Find out what runs are in each experiment.
        Returns {exp:[run1,run2]}"""
        df_run_dict = {}
        sig_run_dict = {} 
        for experiment, model_name in self.df_dict_model_exp.items():
            self.all_runs = all_runs(experiment_number=experiment).get_run_folders()
            print(f"run: {self.all_runs}")
            print("\n")
            print("=================================================")
            print("\n")
            run_list = [] 
            for nested in self.all_runs:
                for run, model_name in nested.items():
                    run_list.append(run)
            df_run_dict[experiment] = run_list

        for experiment, model_name in self.sig_dict_model_exp.items():
            self.all_runs = all_runs(experiment_number=experiment).get_run_folders_sig()
            print(f"run: {self.all_runs}")
            print("\n")
            print("=================================================")
            print("\n")
            run_list = [] 
            for nested in self.all_runs:
                for run, model_name in nested.items():
                    run_list.append(run)
            sig_run_dict[experiment] = run_list
            
        return df_run_dict, sig_run_dict


    def _dict_model_exp(self, data_not_signal: bool):
        """
        Find out which model is used to train each experiment.
        Returns a dict mapping experiment_id -> model_name (or None if not found).

        Example return:
        {'254201543281791601': 'enhanced_rf', 
        '208039388113350502': 'rf', 
        '945551225694133017': 'baseline_ols',
        '167258830472485146': 'enhanced_ols'}
        """
        mlrun_path = None
        if data_not_signal:
            mlrun_path = os.path.join("py", "mlruns")
            experiment_folder = all_runs(experiment_number=None).get_experiments()
        else:
            mlrun_path = Path("mlruns")
            experiment_folder = all_runs(experiment_number=None).get_experiments_sig()
        
            

        print("Experiments found:", experiment_folder)

        experiment_spec = {}

        for experiment in experiment_folder:
            model_spec_path = os.path.join(mlrun_path, str(experiment), 'meta.yaml')
            print(f"Model spec path: {model_spec_path}")
            model_spec = None

            if os.path.exists(model_spec_path):
                try:
                    with open(model_spec_path, "r", encoding="utf-8") as f:
                        print(f"Current working directory: {os.getcwd()}")
                        meta = yaml.safe_load(f)
                        print(f"Meta content for {experiment}: {meta}")
                        model_spec = meta.get("name")
                except Exception as e:
                    print(f"Error reading meta.yaml for experiment {experiment}: {e}")

            experiment_spec[experiment] = model_spec

        return experiment_spec
    
if __name__ == "__main__":
    print("\n=============== \n DATA \n===============")
    # df_dict_model_exp = match_sig_ret()._dict_model_exp(data_not_signal=True)
    
    # print(df_dict_model_exp)

    # print("\n=============== \n SIGNALS \n===============")
    # sig_dict_model_exp = match_sig_ret()._dict_model_exp(data_not_signal=False)
    # print(sig_dict_model_exp)

    # print("\n")
    # print("===========")
    # df_dict_exp_run, sig_dict_exp_run = match_sig_ret()._dict_exp_run()
    # print("===========")
    # print("Data")
    # print(df_dict_exp_run)
    # print("===========")
    # print("Signals")
    # print(sig_dict_exp_run)
    
    # print("\n")
    # print('======================')
    # run_sect_dict = match_sig_ret()._dict_run_sect()
    # print(run_sect_dict)


    # print("\n")
    # print('======================')
    # sig_sector_matching_run, df_sector_matching_run, df_dict_exp_run, sig_dict_exp_run = match_sig_ret(sector='10').fit()
    # print('======================')
    # print("Signals")
    
    # print(sig_sector_matching_run)
    # print('======================')
    # print("Data")
    
    # print(df_sector_matching_run)

    # print(sig_sector_matching_run)
    # print(UniqueValueDictList(sig_sector_matching_run).get_unique())

    # print(f"\n")
    # print(df_sector_matching_run)
    # print(UniqueValueDictList(df_sector_matching_run).get_unique())
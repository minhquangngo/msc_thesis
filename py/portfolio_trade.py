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
from sector_rot import all_runs


ROOT = Path(__file__).resolve().parent.parent

class weights:
    def __init__(self, sector):
        self.sector = sector
    
    def _collect_matched_df(self):
        cols = ['rf_base_signal', 'rf_enhanced_signal', 'ols_base_signal', 'ols_enhanced_signal']
    


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
        sig_sector_matching_run = UniqueValueDictList(sig_sector_matching_run).get_unique()
        df_sector_matching_run = UniqueValueDictList(df_sector_matching_run).get_unique()
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



class UniqueValueDictList:
    """
    Removes dictionaries with duplicate values from a list of single-key dictionaries, keeping the first occurrence.
    Example:
        input = [
            {'a': 'x'}, {'b': 'x'}, {'c': 'y'}, {'d': 'y'}, {'e': 'z'}
        ]
        output = [
            {'a': 'x'}, {'c': 'y'}, {'e': 'z'}
        ]
    """
    def __init__(self, dict_list):
        self.original_list = dict_list
        self.unique_list = self._remove_duplicates(dict_list)

    def _remove_duplicates(self, dict_list):
        seen_values = set()
        unique_dicts = []
        for d in dict_list:
            # Each dict is assumed to have only one key-value pair
            value = next(iter(d.values()))
            if value not in seen_values:
                unique_dicts.append(d)
                seen_values.add(value)
        return unique_dicts

    def get_unique(self):
        return self.unique_list

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
    # print("\n=============== \n DATA \n===============")
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
    
"""
- Load the signals in, group by models. Example for run X, collect all sectors
10->60.
- Go over day by day of 2018, define the weights of the  port
    - Porfolio construction: equal portfolio weights. Meaning if there are 
    no signals on a certain day, then weights is distributed across all sectors.
    - But if, for example, 3 sectors have signals, the weights are 1/3 for each of those sectors.


"""
import os
from pathlib import Path
import yaml
import pandas as pd
from sector_rot import all_runs


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

        self.sig_dict_model_exp = self._dict_model_exp(signal_not_data=True)
        self.df_dict_model_exp = self._dict_model_exp(signal_not_data=False)    
        self.df_dict_exp_run, self.sig_dict_exp_run = self._dict_exp_run() 

    # def _fit(self):
    #     dict_original_df = self._dict_exp_run(signal_not_data=False)
    #     model_experiment = self._dict_model_exp(signal_not_data=True)
    #     dict_exp_run = self._dict_exp_run(signal_not_data=True, dict_model_exp=model_experiment)

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
            self.all_runs = all_runs(experiment_number=experiment).get_run_folders()
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


    def _dict_model_exp(self, signal_not_data: bool):
        """
        Find out which model is used to train each experiment.
        Returns a dict mapping experiment_id -> model_name (or None if not found).

        Example return:
        {'254201543281791601': 'enhanced_rf', 
        '208039388113350502': 'rf', 
        '945551225694133017': 'baseline_ols',
        '167258830472485146': 'enhanced_ols'}
        """
        if not signal_not_data:
            mlrun_path = os.path.join("mlruns")
        else:
            mlrun_path = os.path.join("py", "mlruns")

        experiment_folder = all_runs(experiment_number=None).get_experiments()
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
    df_dict_model_exp = match_sig_ret()._dict_model_exp(signal_not_data=True)
    print("\n=============== \n DATA \n===============")
    print(df_dict_model_exp)

    sig_dict_model_exp = match_sig_ret()._dict_model_exp(signal_not_data=False)
    print("\n=============== \n SIGNALS \n===============")
    print(sig_dict_model_exp)
    # print("\n")
    # print("===========")
    # df_dict_exp_run, sig_dict_exp_run = match_sig_ret()._dict_exp_run()
    # print("===========")
    # print("Data")
    # print(df_dict_exp_run)
    # print(sig_dict_exp_run)
    
    # print("\n")
    # print('======================')
    # # run_sect_dict = match_sig_ret()._dict_run_sect()
    # # print(run_sect_dict)
    
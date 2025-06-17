"""
1. Signals: RF 1 -month aheadPredicted returns for the sector + ratio of SR and LR rewalized vol
2. 
"""
import pandas as pd
import statsmodels.api as sm 
from sklearn.ensemble import RandomForestRegressor
import os
import glob
import shutil
import joblib
import yaml
import statsmodels.api as sm
from pathlib import Path
from mlxtend.frequent_patterns import association_rules, apriori
import numpy as np
import mlflow
import hashlib
import json

def fingerprint(df:pd.DataFrame,experiment: str,run:str) -> str:
    """
    Hash the result of the models.
    """
    feats = rolling_pred(experiment,run,df)._extract_features()
    meta = dict(
        experiment = experiment,
        run =run,
        features =feats,
        n_rows = len(df),
        data_hash = hashlib.md5(pd.util.hash_pandas_object(df,index = True).values).hexdigest()
    )
    return hashlib.md5(json.dumps(meta, sort_keys=True).encode()).hexdigest()

def experiment_check(experiment_name:str):
    """
    Check if an experiment with the same name exists or not.2
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f'Experiment {experiment_id} created')
    else:
        experiment_id = experiment.experiment_id
        print(f"Experiment already exists with ID: {experiment_id}")
    mlflow.set_experiment(experiment_name)
    return experiment_id


#----------------------Volatility------------------
#--------------------------------------------------

def realized_vol(var, time_sr, time_lr):
        """
    VR = Volatlity at time t  = Volatility (t-24months,t) / Volatiliity (t-36m,t)
    VR<1 = Near term vol is smaller than long term vol 
    -----EXAMPLE-----
    rv_short = realised_vol(panel['ret'], 21)   # ≈ one trading month
    rv_long  = realised_vol(panel['ret'], 126)   # ≈ six trading weeks
    panel['vr'] = rv_short / rv_long
    """
        rv_sr = var.rolling(time_sr).std()
        rv_lr = var.rolling(time_lr).std()
        ratio = rv_sr / rv_lr
        return ratio

#TODO: write a class that automatically takes the rolling pred and apply it to the each of the models

class rolling_pred():
    def __init__(
            self,
            experiment_numb,
            run, 
            df =None, 
            lookback_time = 365, 
            vol_threshold = 1.0,
            pred_thresh = 0.0,
            excess_ret_pred_threshold = 0.0,
            sr = 21,
            lr = 183,
            experiment_name = None):
        self.experiment_numb = experiment_numb
        self.run = run 
        self.df = df
        self.lookback_time = lookback_time #the total time that the data is trained on
        self.vol_threshold = vol_threshold
        self.pred_thresh = pred_thresh
        self.excess_ret_pred_threshold = excess_ret_pred_threshold
        self.sr = sr
        self.lr = lr
        
        self.experiment_name = experiment_name

        self.mlruns_path = os.path.join("py","mlruns", str(self.experiment_numb))
        self.meta_path = os.path.join(self.mlruns_path, "meta.yaml")
        
        print(f"MLRuns path: {self.mlruns_path}")
        print(f"Meta path: {self.meta_path}")
        print(f"Meta path exists: {os.path.exists(self.meta_path)}")
    
    def fit(self):
        """
        Alternative cleaner version of _fit() method.
        """
        #=========FINGERPRINTING=============
        self.hash_fp = fingerprint(self.df,self.experiment_numb, self.run)
        fingerprint_check = self.model_fingerprint_check()
        #----check existing run---
        if fingerprint_check is not None:
            print("Skipping- already logged this model") 
            return mlflow.get_run(fingerprint_check)
        
        #=========DUMPING===================
        print("Dumping models")
        self.models = self._dump_model()
        is_ols_only = self.models[1] is None
        
        # =======Get predictions==============
        if is_ols_only:
            print("OLS detected")
            self.ols_prediction_series = self._pred()
            self.rf_prediction_series = None
            self.surr_prediction_series = None
            self.pred_series = self.ols_prediction_series
            
            print(f"OLS predictions generated: {self.ols_prediction_series.head()}")
            print("Generating apriori df....")
            self.ols_apriori_df = self._prep_apriori(
                self.df,
                vol_threshold=1.0,
                pred_thresh=0.0,
                excess_ret_pred_threshold=0.0,
                sr=self.sr,
                lr=self.lr,
                pred_series=self.ols_prediction_series
            )
            print(f"Apriori df generated: {self.ols_apriori_df.head()}")

            print("Generating arl set....")
            self.ols_arl_set = self._arl(
                self.df,
                self.ols_apriori_df)
            print(f"Signal set generated: {self.ols_arl_set[0].tail(50)}")
            print(f"Rule diagnostics df generated: {self.ols_arl_set[1].tail(50)}")

            self.ols_signal_set, self.ols_rule_set = self.ols_arl_set

            self._log_metrics(self.features)

            return {
                "ols_signal_set": self.ols_signal_set,
                "ols_rule_set": self.ols_rule_set,
                "feature_importance": None
            }
            
        else:
            print("RF and Surr detected")
            self.rf_prediction_series, self.surr_prediction_series, self.rf_feature_importance, self.surr_feature_importance = self._pred()
            self.ols_prediction_series = None

            print(f"RF predictions generated: {self.rf_prediction_series.tail(250)}")
            print(f"Surrogate predictions generated: {self.surr_prediction_series.tail(250)}")

            print("Generating apriori df....")
            self.rf_apriori_df = self._prep_apriori(
                self.df,
                vol_threshold=1.0,
                pred_thresh=0.0,
                excess_ret_pred_threshold=0.0,
                sr= self.sr,
                lr=self.lr,
                pred_series=self.rf_prediction_series
            ) #TODO: mlflow log this
            print(f"RF apriori df generated: {self.rf_apriori_df.head()}")
            
            print("Generating surr apriori df....")
            self.surr_apriori_df = self._prep_apriori(
                self.df,
                vol_threshold=1.0,
                pred_thresh=0.0,
                excess_ret_pred_threshold=0.0,
                sr=self.sr,
                lr=self.lr,
                pred_series=self.surr_prediction_series
            ) 
            print(f"Surr apriori df generated: {self.surr_apriori_df.head()}")

            print("Generating rf arl set....")
            self.rf_arl_set = self._arl(
                self.df,
                self.rf_apriori_df)
            print(f"RF signal set generated: {self.rf_arl_set[0].tail(50)}")
            print(f"RF rule diagnostics df generated: {self.rf_arl_set[1].tail(50)}")

            print("Generating surr arl set....")
            self.surr_arl_set = self._arl(
                self.df,
                self.surr_apriori_df)
            print(f"Surr signal set generated: {self.surr_arl_set[0].tail(50)}")
            print(f"Surr rule diagnostics df generated: {self.surr_arl_set[1].tail(60)}")
            
            self.rf_signal_set, self.rf_rule_set = self.rf_arl_set

            self._log_metrics(self.features)

            return {
                "rf_signal_set": self.rf_signal_set,
                "rf_rule_set": self.rf_rule_set,
                "feature_importance": self.rf_feature_importance
            }

    def _pred(self):
        self.ols_prediction_series = pd.Series(index=self.df.index, dtype=float)
        self.rf_prediction_series = pd.Series(index=self.df.index, dtype=float)
        self.surr_prediction_series = pd.Series(index=self.df.index, dtype=float)
        self.feat_imp_rf = []
        self.feat_imp_surr = []
        self.features = self._extract_features()
        print(f"Features {self.features}")

        for t in range(self.lookback_time, len(self.df)-1):
            X_test = self.df.loc[[self.df.index[t]], self.features].shift(1)
            X_test_const = sm.add_constant(X_test, has_constant = 'add')
            
            
            if self.models[1] is None: 
                trained_ols = self.models[0]
                self.ols_prediction_series.iloc[t] = trained_ols.predict(X_test_const)[0]#[CHANGE]
            else:
                trained_rf = self.models[0]
                trained_surr = self.models[1]
                self.rf_prediction_series.iloc[t] = trained_rf.predict(X_test)[0] #[CHANGE]
                self.feat_imp_rf.append(trained_rf.feature_importances_)

                self.surr_prediction_series.iloc[t] = trained_surr.predict(X_test)[0]#[CHANGE]
                self.feat_imp_surr.append(trained_surr.feature_importances_)

        # Return statements AFTER the loop completes
        if self.models[1] is None:
            return self.ols_prediction_series
        else:
            return self.rf_prediction_series, self.surr_prediction_series, self.feat_imp_rf, self.feat_imp_surr

        
    
    def _prep_apriori(self,df: pd.DataFrame,
                       vol_threshold: float,
                       pred_thresh: float,
                       excess_ret_pred_threshold : float,
                       sr: int,
                       lr:int,
                       pred_series: pd.Series
                       ) -> pd.DataFrame:
        """
        Build the one-hot dataframe that the `apriori` algorithm expects.
        Each row is a 'transaction' (sector-month observation).
        An item is present (=True) when the condition holds.
        Antecedent items:
            - low_vr      : VR < vr_thresh   (default 1.0)
            - high_pred   : t model prediction > pred_thresh
        Consequent item:
            - ret_up      : t excess return > ret_thresh
        """
        df = df.copy()
        df["VR"]= realized_vol(df['ret'],sr, lr) #[CHANGE]        rolling statistic
        df["low_vr"]    = df["VR"]   < vol_threshold

        # Use the prediction series that matches your model type. Example below uses rf_prediction_series:
        df["high_pred"] = pred_series > pred_thresh # add more arguments for models here
        df["ret_up"]    = df["ret"]> excess_ret_pred_threshold #[CHANGE]

        items = df[["low_vr", "high_pred", "ret_up"]].dropna().astype("bool")
        return items

    def _arl(self,
            df,
            apriori_df): #TODO: change 6 months of trading days
        """
        Predictio_df contains:
        - Prediction
        - Excess return org vals
        Added:
        
        Definitions:
        - Backroll = amount of time that the 
        Condition_df are conditions:
        - low realized volatility < 1 (T/F)
        - high prediction > 0 (T/F)
        1. Conditions are binary (if realized vol > x and prediction > y)
        2. Metrics to eval:
        Metrics:
        Support: How often this rule occurs in the data.

        Confidence: How often the antecedent leads to the consequent.

        Lift: How much more likely the consequent is, given the antecedent, compared to chance.
        """
        signal = [] 
        rule_book = []
        for t in range(self.lookback_time, len(df)-1): # this is NOT the window. this is to stop the loop from going over the df len
            """
            Looks back at the time for the 2 signals:
            - If at time t
                - RF pred of t+1 >threshold (high prediction)
                - realized vol <1 for the past quarter(low vol)
            can reliably signal an actual excess return next month over a certain threshold

            How:
            - See if one of the extracted rule at time t for a lookback time: low_vol & high pred -> excess pred > \beta
            - Loop it over all days of 2018
            """

            daterange_start = t - self.lookback_time
            daterange_end = t
            window_df = apriori_df.iloc[daterange_start : daterange_end + 1]

            frequent = apriori(
                window_df, 
                min_support=0.05, 
                use_colnames=True)
        
            rules = association_rules(
                frequent, 
                metric="confidence", 
                min_threshold=0.4
                )
            
            # filter out: the consquentcol of rule is ret up and the antecedents must be low_ver and high_pred
            target_rule = rules[
                (rules["consequents"] == frozenset({"ret_up"})) &
                (rules["antecedents"] == frozenset({"low_vr", "high_pred"}))
            ]
            
            if target_rule.empty:
                signal.append(0)
                rule_book.append(
                    {"idx":apriori_df.index[t],
                     "support":np.nan,
                     "confidence":np.nan,
                     "lift": np.nan}
                )
                continue # if there are no rules matching empty then EVERYTHING below is skipped

            rule_book_row = target_rule.iloc[0]

            rule_book.append({
                "idx": apriori_df.index[t],
                "support": rule_book_row['support'],
                "confidence" : rule_book_row['confidence'],
                "lift": rule_book_row ['lift']
                  })
            
            # the rules above check the statistical rigor of the rule.
            # if the rules for the past {lookback time} is rigorous, we look at the condition for this
            ## current month

            condition_low_vr = apriori_df.iloc[t]['low_vr']
            condition_high_pred = apriori_df.iloc[t] ['high_pred']
            hold_sect = int(condition_low_vr and condition_high_pred)
            signal.append(hold_sect)
        
        sig_index = apriori_df.index[self.lookback_time+1 :]
        signal_series = (
            pd.Series(signal, index=sig_index, name="signal")  # build series
            .shift(1)                                          # then lag the values
        )
        rule_diagnostics_df = pd.DataFrame(rule_book).set_index("idx").shift(1, axis=0)
            
        return signal_series, rule_diagnostics_df

    #==================INTERNAL HELPERS=========================
    def _extract_model_pkl(self):
        """
        Get the path of that run's model
        """
        print(f"\n=== Debugging _extract_model_pkl ===")
        
        ols_pkl_file = None
        rf_pkl_file = None 
        surr_pkl_file = None
        
        print(f"Checking if meta.yaml exists: {os.path.exists(self.meta_path)}")
        
        if os.path.exists(self.meta_path): 
            try:
                with open(self.meta_path, "r") as f: #opening the meta.yaml file
                    meta = yaml.safe_load(f)
                print(f"Meta content: {meta}")
                
                model_name = meta.get("name") #get the model spec from the meta.yaml
                print(f"meta.yaml name extract: {model_name}")
                
                if model_name in ["baseline_ols", "enhanced_ols"]:
                    ols_path = os.path.join(
                        "py", "mlartifacts", str(self.experiment_numb), str(self.run),"artifacts","ols_model", "*.statsmodels")
                    print(f"OLS pattern: {ols_path}")
                    ols_pkl_file = glob.glob(ols_path)
                    print(f"OLS files found: {ols_pkl_file}")
                    
                elif model_name in ["rf", "enhanced_rf"]:
                    rf_path = os.path.join("py", "mlartifacts", str(self.experiment_numb), str(self.run),"artifacts","rf_model","*.pkl")
                    surr_path = os.path.join("py", "mlartifacts", str(self.experiment_numb), str(self.run),"artifacts","surr_model","*.pkl")

                    print(f"RF path: {rf_path}")
                    print(f"Surr path: {surr_path}")

                    rf_pkl_file = glob.glob(rf_path)
                    surr_pkl_file = glob.glob(surr_path)

                    print(f"RF files found: {rf_pkl_file}")
                    print(f"Surr files found: {surr_pkl_file}")
                
            except Exception as e: # Runtime Error which allows us to trace back to where the err happens
                print(f"Error reading meta.yaml: {e}")
        else:
            print("meta.yaml does not exist!")
        
        return ols_pkl_file, rf_pkl_file, surr_pkl_file
    
    def _dump_model(self):
        print(f"\n=== Debugging _dump_model ===")
        
        ols_path, rf_path, surr_path = self._extract_model_pkl()
        
        if not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"Meta file not found: {self.meta_path}")
            
        try:
            with open(self.meta_path, "r") as f:
                meta = yaml.safe_load(f)
            model_name = meta.get("name")
            print(f"Processing model type: {model_name}")
            
            if ols_path and model_name in ["baseline_ols", "enhanced_ols"]:
                print(f"Loading OLS model from: {ols_path[0]}")  # ols_path is a list
                try:
                    ols_model = sm.load(ols_path[0])  # Take first file from list
                    print("OLS model loaded successfully")
                    return ols_model, None
                except Exception as e:
                    print(f"Error loading OLS model: {e}")
                    raise
                    
            elif rf_path and surr_path and model_name in ["rf", "enhanced_rf"]:
                print(f"Loading RF model from: {rf_path[0]}")  
                print(f"Loading Surr model from: {surr_path[0]}")
                try:
                    rf_model = joblib.load(rf_path[0])   
                    surr_model = joblib.load(surr_path[0])  
                    print("RF and Surr models loaded successfully")
                    return rf_model, surr_model
                except Exception as e:
                    print(f"Error loading RF/Surr models: {e}")
                    raise
            else:
                error_msg = f"No valid model files found. OLS: {bool(ols_path)}, RF: {bool(rf_path)}, Surr: {bool(surr_path)}, Model name: {model_name}"
                print(error_msg)
                raise FileNotFoundError(error_msg)
                
        except Exception as e:
            print(f"Error in _dump_model: {e}")
            raise
    
    def _extract_sector_factors(self):
        print(f"Extracting sector number for run {self.run}")
        sectors_path = os.path.join(self.mlruns_path, str(self.run), "params", "sector")
        print(f"Looking for sectors file at: {sectors_path}")
        try:
            if os.path.exists(sectors_path):
                with open(sectors_path, 'r') as f:
                    sectors = f.read()
                print(f"Sectors loaded: {sectors}")
            else:
                print("Sectors file not found")
                sectors = None
        except Exception as e:
            print(f"Error loading sector/factors: {e}")
            sectors = None
        return sectors

    def _extract_features(self):
        """Get the feature set of the run"""
        print(f"\n=== Debugging _extract_features ===")
        factors_path = os.path.join(self.mlruns_path,str(self.run),"params","factors")
        print(f"Looking for factors file at: {factors_path}")
        
        if not os.path.exists(factors_path):
            raise FileNotFoundError(f"Factors file not found: {factors_path}")
            
        try:
            with open(factors_path, "r") as f:
                features = eval(f.read())
            print(f"Features loaded: {features}")
            return features
        except Exception as e:
            print(f"Error loading features: {e}")
            raise
    
    def model_fingerprint_check(self)-> str | None:
        """
        Use the experiment_check external func to get the list of experiment_id
        On that experiment id, check whether fingerprint already exists or not
         -If exists - skip
        """
        exists = mlflow.search_runs(
            experiment_ids = [experiment_check(self.experiment_name)],
            filter_string = f"tags.fingerprint='{self.hash_fp}' and tags.status = 'completed'",
            max_results = 1)
        
        if not exists.empty:
            return exists.loc[0,"run_id"]
        return None
   

    def _log_metrics(
            self,
            features,
            ):
        sector_number = self._extract_sector_factors()
        with mlflow.start_run(
            run_name = f"{self.experiment_numb}_{self.run}" ,
            tags={
                "fingerprint": str(self.hash_fp),
                "features": str(features),      # Convert list to string
                "status": 'completed',
                "experiment": str(self.experiment_name),
                "experiment_number": str(self.experiment_numb)  # Convert int to string
      }):
            params ={
                "features":features,
                "sector": sector_number,
                "experiment": self.experiment_numb,
                "run": self.run}
            mlflow.log_params(params)

            if self.ols_prediction_series is not None:
                self.ols_prediction_series.to_csv(f'ols_pred_series_{self.run}.csv', header= True, index = True)
                mlflow.log_artifact(f'ols_pred_series_{self.run}.csv')

                self.ols_apriori_df.to_csv(f'ols_apriori_df_{self.run}.csv', header = True, index = True)
                mlflow.log_artifact(f"ols_apriori_df_{self.run}.csv")

                self.ols_signal_set.to_csv(f'ols_signal_set_{self.run}.csv', header = True, index = True)
                mlflow.log_artifact(f'ols_signal_set_{self.run}.csv')

                self.ols_rule_set.to_csv(f'ols_rule_set_{self.run}.csv', header = True, index = True)
                mlflow.log_artifact(f'ols_rule_set_{self.run}.csv')
            else:
                self.rf_prediction_series.to_csv(f'rf_pred_series_{self.run}.csv',header= True, index= True)
                self.surr_prediction_series.to_csv(f'surr_pred_series_{self.run}.csv', header = True, index = True)
                mlflow.log_artifact(f'rf_pred_series_{self.run}.csv')
                mlflow.log_artifact(f'surr_pred_series_{self.run}.csv')
                
                self.rf_apriori_df.to_csv(f'rf_apriori_df_{self.run}.csv', header = True, index = True)
                self.surr_apriori_df.to_csv(f'surr_apriori__{self.run}.csv', header = True, index = True)
                mlflow.log_artifact(f'rf_apriori_df_{self.run}.csv')
                mlflow.log_artifact(f'surr_apriori__{self.run}.csv')

                self.rf_signal_set.to_csv(f'rf_signal_set_{self.run}.csv', header = True, index = True)
                self.rf_rule_set.to_csv(f'rf_rule_set_{self.run}.csv', header = True, index = True)
                mlflow.log_artifact(f'rf_signal_set_{self.run}.csv')
                mlflow.log_artifact(f'rf_rule_set_{self.run}.csv')

                with open (f'feature_importance_rf_{self.run}.json', 'w') as f:
                    # Convert numpy arrays to lists for JSON serialization
                    rf_feat_imp_serializable = [arr.tolist() if hasattr(arr, "tolist") else arr for arr in self.rf_feature_importance]
                    json.dump(rf_feat_imp_serializable, f)
                mlflow.log_artifact(f'feature_importance_rf_{self.run}.json')
    


class all_runs:
    def __init__(self, experiment_number):
        """
        Initialize with the experiment number. The experiment folder is under py/mlruns/{experiment_number}
        """
        self.experiment_number = str(experiment_number)
        self.experiment_path_df = os.path.join("py", "mlruns", self.experiment_number)
        self.experiment_path_sig = os.path.join("mlruns", self.experiment_number)

    def get_run_folders(self):
        """
        Returns a list of run folder names (not full paths) under the experiment folder.
        Excludes files like meta.yaml.
        """
        try:
            all_entries = os.listdir(self.experiment_path_df)
            run_number = [entry for entry in all_entries 
                           if os.path.isdir(os.path.join(self.experiment_path_df, entry))]
            run_spec = []
            for run in run_number:
                meta_path = os.path.join(self.experiment_path_df, run, "meta.yaml")
                run_name = None
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, "r") as f:
                            meta = yaml.safe_load(f)
                        run_name = meta.get("run_name", None)
                    except Exception as e:
                        print(f"Error reading meta.yaml for run {run}: {e}")
                run_spec.append({run: run_name})
            return run_spec
        except Exception as e:
            print(f"Error reading experiment folder: {e}")
            return []
    def get_run_folders_sig(self):
        """
        Returns a list of run folder names (not full paths) under the experiment folder.
        Excludes files like meta.yaml.
        """
        try:
            all_entries = os.listdir(self.experiment_path_sig)
            run_number = [entry for entry in all_entries 
                           if os.path.isdir(os.path.join(self.experiment_path_sig, entry))]
            run_spec = []
            for run in run_number:
                meta_path = os.path.join(self.experiment_path_sig, run, "meta.yaml")
                run_name = None
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, "r") as f:
                            meta = yaml.safe_load(f)
                        run_name = meta.get("run_name", None)
                    except Exception as e:
                        print(f"Error reading meta.yaml for run {run}: {e}")
                run_spec.append({run: run_name})
            return run_spec
        except Exception as e:
            print(f"Error reading experiment folder: {e}")
            return []
            
    def get_experiments(self):
        """
        Returns a list of all experiment folders.
        """
        try:
            all_entries = os.listdir("py/mlruns")
            experiment_folders = [entry for entry in all_entries 
                                  if os.path.isdir(os.path.join("py", "mlruns", entry))]
            experiment_folders = [entry for entry in experiment_folders 
                                 if entry not in ['0', '.trash','models']]
            return experiment_folders
        except Exception as e:
            print(f"Error reading mlruns directory: {e}")
            return []

    def get_experiments_sig(self): # this is for the portfolio_trade.py
        """
        Returns a list of all experiment folders.
        """
        try:
            all_entries = os.listdir("mlruns")
            experiment_folders = [entry for entry in all_entries 
                                  if os.path.isdir(os.path.join("mlruns", entry))]
            experiment_folders = [entry for entry in experiment_folders 
                                 if entry not in ['0', '.trash','models']]
            return experiment_folders
        except Exception as e:
            print(f"Error reading mlruns directory: {e}")
            return []
        


if __name__ == "__main__":
    try:
        print(f"\n------RF path test------")
        rf_predictor = rolling_pred(208039388113350502, "0c861f5f9a874e05b04e43bb6341bd96")
        result = rf_predictor._pred()
        rf_feats = rf_predictor._extract_features()
        print(f"\nFinal result: {result}")
        print(f"Model rf types: {type(result[0]) if result[0] else None}, {type(result[1]) if result[1] else None}")
        print(f"Extracted rf features: {rf_feats}")

        print(f"\n------OLS path test------")
        ols_predictor = rolling_pred(717450669580849996, "2fb51e9b2ff74ab992a0fe6b0cdecf1a")
        ols_result = ols_predictor._pred()
        ols_feats = ols_predictor._extract_features()
        print(f"\nFinal OLS result: {ols_result}")
        print(f"Model ols types: {type(ols_result[0]) if ols_result[0] else None}, {type(ols_result[1]) if ols_result[1] else None}")
        print(f"Extracted ols features: {ols_feats}")


    except Exception as e:
        print(f"Main execution error: {e}")


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

#----------------------Volatility------------------
#--------------------------------------------------

def realized_vol(var, time_sr, time_lr):
        """
    VR = Volatlity at time t  = Volatility (t-24months,t) / Volatiliity (t-36m,t)
    VR<1 = Near term vol is smaller than long term vol 
    -----EXAMPLE-----
    rv_short = realised_vol(panel['ret'], 21)   # ≈ one trading month
    rv_long  = realised_vol(panel['ret'], (6 months trading in days))   # ≈ six trading weeks
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
            experiment, 
            run, 
            df =None, 
            lookback_time =None, 
            vol_threshold = 1.0,
            pred_thresh = 0.0,
            excess_ret_pred_threshold = 0.0,
            sr = 21,
            lr = 183):
        self.experiment = experiment
        self.run = run 
        self.df = df
        self.lookback_time = lookback_time #the total time that the data is trained on
        self.vol_threshold = vol_threshold
        self.pred_thresh = pred_thresh
        self.excess_ret_pred_threshold = excess_ret_pred_threshold
        self.sr = sr
        self.lr = lr

        self.mlruns_path = os.path.join("py","mlruns", str(self.experiment))
        self.meta_path = os.path.join(self.mlruns_path, "meta.yaml")
        
        print(f"MLRuns path: {self.mlruns_path}")
        print(f"Meta path: {self.meta_path}")
        print(f"Meta path exists: {os.path.exists(self.meta_path)}")
    
    def _fit(self):
        self.pred_series = self._pred()
        
        if self.ols_prediction_series is not None:
            print(f"OLS predictions: {self.ols_prediction_series.head()}")
            apriori_df = self._prep_apriori(
                self.df,
                vol_threshold=1.0,
                pred_thresh=0.0,
                excess_ret_pred_threshold=0.0,
                sr=21,
                lr=183,
                pred_series=self.pred_series
            )
            

        elif self.rf_prediction_series is not None:
            print(f"RF predictions: {self.rf_prediction_series.head()}")
            apriori_df = self._prep_apriori(
                self.df,
                vol_threshold=1.0,
                pred_thresh=0.0,
                excess_ret_pred_threshold=0.0,
                sr=21,
                lr=183,
                pred_series=self.rf_prediction_series
            )
            surr_apriori_df = self._prep_apriori(
                self.df,
                vol_threshold=1.0,
                pred_thresh=0.0,
                excess_ret_pred_threshold=0.0,
                sr=21,
                lr=183,
                pred_series=self.surr_prediction_series
            )
            
    def _pred(self):
        model = self._dump_model()
        ols_prediction_series = pd.Series(index=self.df.index, dtype=float)
        rf_prediction_series = pd.Series(index=self.df.index, dtype=float)
        surr_prediction_series = pd.Series(index=self.df.index, dtype=float)
        feat_imp_rf = []
        feat_imp_surr = []
        
        for t in range(self.lookback_time, len(self.df)-1):
            df_trainperiod = self.df.iloc[t - self.lookback_time:t]
            print("Training set debug")
            print(df_trainperiod.head(10))        
            features = self._extract_features()
            print(f"Features {features}")
            y_train = df_trainperiod['excess_ret']  
            X_train = df_trainperiod[features]
            X_test = self.df.loc[[self.df.index[t]], features]
            print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            print(f"X_test shape: {X_test.shape}")
            
            if model[1] is None: 
                trained_ols = model[0].fit(X_train, y_train)
                ols_prediction_series.iloc[t] = trained_ols.predict(X_test)[0]
            else:
                trained_rf = model[0].fit(X_train, y_train)
                trained_surr = model[1].fit(X_train, y_train)
                
                rf_prediction_series.iloc[t] = trained_rf.predict(X_test)[0]
                feat_imp_rf.append(trained_rf.feature_importances_)

                surr_prediction_series.iloc[t] = trained_surr.predict(X_test)[0]
                feat_imp_surr.append(trained_surr.feature_importances_)
        
        # Return statements AFTER the loop completes
        if model[1] is None:
            return ols_prediction_series
        else:
            return rf_prediction_series, surr_prediction_series, feat_imp_rf, feat_imp_surr

        
        #TODO:add a logging function that logs it into mlflow
    
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
            - high_pred   : t +1 model prediction > pred_thresh
        Consequent item:
            - ret_up      : t+1 excess return > ret_thresh
        """
        df = df.copy()
        df["VR"]= realized_vol(df['excess_ret'],sr, lr)                 # rolling statistic
        df["low_vr"]    = df["VR"]   < vol_threshold
        # Use the prediction series that matches your model type. Example below uses rf_prediction_series:
        df["high_pred"] = pred_series > pred_thresh # add more arguments for models here
        df["ret_up"]    = df["excess_ret"]> excess_ret_pred_threshold

        items = df[["low_vr", "high_pred", "ret_up"]].dropna().astype("bool")
        return items

    def _arl(self,
            df,
            vol_threshold = 1.0,
            prediction_threshold = 0.0,
            excess_ret_pred_threshold = 0.0,
            sr =21,
            lr = 183): #TODO: change 6 months of trading days
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
        apriori_df = self._prep_apriori(
            df,
            vol_threshold,
            prediction_threshold,
            excess_ret_pred_threshold,
            sr,
            lr
                )
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
                min_support=0.02, 
                use_colnames=True)
        
            rules = association_rules(
                frequent, 
                metric="confidence", 
                min_threshold=0.55
                )
            
            mask = rules["consequents"].apply(lambda s: s == frozenset({"ret_up"}))
            rules = rules.loc[mask]
            #The result that rules returns is a dataframe, which contains 2 cols: antecendens and consequents
                # Consequents is the rule of the right hand side (x &y -> consequents)
                # The rule set is going to determine both the rules using ret_up and predicting ret_up
                # We are only interested in the ret_up prediction
            
            target_rule = rules.loc[
            rules["antecedents"] == frozenset({"low_vr", "high_pred"})
        ]
            #Same thing we are only interested in the low_vr and high_pred
            if target_rule.empty:
                signal.append(0)
                rule_book.append(
                    {"idx":apriori_df.index[t],
                     "support":np.nan,
                     "confidence":np.nan,
                     "lift": np.nan}
                )
                continue # if it is empty then EVERYTHING below is skipped

            rule_book_row = target_rule.iloc[0]

            rule_book.append({
                "idx": rule_book_row.index(t),
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
        
        sig_index = apriori_df.index[self.lookback_time+1 : self.lookback_time+1+len(signal)]
        signal_series         = pd.Series(signal, index=sig_index, name="signal")
        rule_diagnostics_df   = pd.DataFrame(rule_book).set_index("idx")
            
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
                        "py", "mlartifacts", str(self.experiment), str(self.run),"artifacts","ols_model", "*.statsmodels")
                    print(f"OLS pattern: {ols_path}")
                    ols_pkl_file = glob.glob(ols_path)
                    print(f"OLS files found: {ols_pkl_file}")
                    
                elif model_name in ["rf", "enhanced_rf"]:
                    rf_path = os.path.join("py", "mlartifacts", str(self.experiment), str(self.run),"artifacts","rf_model","*.pkl")
                    surr_path = os.path.join("py", "mlartifacts", str(self.experiment), str(self.run),"artifacts","surr_model","*.pkl")
                    
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


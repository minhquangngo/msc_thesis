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
#----------------------Volatility------------------
#--------------------------------------------------

def realized_vol(var, time):
    '''
    VR = Volatlity at time t  = Volatility (t-24months,t) / Volatiliity (t-36m,t)
    VR<1 = Near term vol is smaller than long term vol 
    -----EXAMPLE-----
    rv_short = realised_vol(panel['ret'], 21)   # ≈ one trading month
    rv_long  = realised_vol(panel['ret'], 36)   # ≈ six trading weeks
    panel['vr'] = rv_short / rv_long
    '''
    rv = var.rolling(time).std()
    return rv


#TODO: write a class that automatically takes the rolling pred and apply it to the each of the models

class rolling_pred():
    def __init__(self, experiment, run, df=None, lookback_time=None):
        self.experiment = experiment
        self.run = run 
        self.df = df
        self.lookback_time = lookback_time

        self.mlruns_path = os.path.join("py","mlruns", str(self.experiment))
        self.meta_path = os.path.join(self.mlruns_path, "meta.yaml")
        
        print(f"MLRuns path: {self.mlruns_path}")
        print(f"Meta path: {self.meta_path}")
        print(f"Meta path exists: {os.path.exists(self.meta_path)}")

    def _pred(self):
        model = self._dump_model()
        ols_prediction_series = pd.Series(index=self.df.index, dtype=float)#create series with the same index as df
        rf_prediction_series = pd.Series(index=self.df.index, dtype=float)#create series with the same index as df
        surr_prediction_series = pd.Series(index=self.df.index, dtype=float)#create series with the same index as df
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
            X_test = self.df.loc[[self.df.index[t]], features]  #locate the test row based on index label and by feature column
            print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            print(f"X_test shape: {X_test.shape}")
            if model[1] is None: 
                trained_ols = model[0].fit(X_train,y_train)
                ols_prediction_series.iloc[t] = trained_ols.predict(X_test)[0]
            else:
                trained_rf = model[0].fit(X_train,y_train)
                trained_surr = model [1].fit(X_train,y_train)
                
                rf_prediction_series.iloc[t] = trained_rf.predict(X_test)[0]
                feat_imp_rf.append(trained_rf.feature_importances_)

                surr_prediction_series.iloc[t] = trained_surr.predict(X_test)[0]
                feat_imp_surr.append(trained_surr.feature_importances_)

        return ols_prediction_series, rf_prediction_series, surr_prediction_series
        #TODO:add feature importance to the output
        #TODO: shift pred to the next day
        
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
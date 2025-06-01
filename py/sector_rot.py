"""
1. Signals: RF 1 -month aheadPredicted returns for the sector + ratio of SR and LR rewalized vol
2. 
"""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import os
import glob
import shutil
import joblib
import yaml
import statsmodels.api as sm
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

class rolling_pred():
    def __init__(self, experiment, run,df,lookback_time):
        self.experiment = experiment
        self.run = run 
        self.df = df
        self.lookback_time = lookback_time

        self.mlruns_path = os.path.join(
            "py", "mlruns", str(self.experiment)
        )
        self.meta_path = os.path.join(self.mlruns_path, "meta.yaml")

    def _fit(self):
        X,y = self._var_prep()
        model = self._dump_model() 
        """
        Example model usage:
        - model[0] = ols
        - model [0],model [1] = rf and surr
        """
        return model


    def _var_prep(self):
        for t in range (self.lookback_time, len(self.df)-1):
            df_period = self.df.iloc[t - self.lookback_time:t]
            features = self._extract_features()
            y = df_period.loc['excess_ret']
            X = df_period.loc[features]
        return X,y
    
    def _train(self):
        """
        Take in X,y ->  split train test -> take in a model -> train -> spits out lagged pred
        """
        X,y = self._var_prep()
        model_1, model_2 = self._dump_model()
        

            
        #TODO:train test split
    #-----------Path extractors----------
    def _extract_model_pkl(self):
        """
        Get the path of that run's model
        if within mlrun of that experiement = ols
        """
        ols_pkl_file = None
        rf_pkl_file = None 
        surr_pkl_file = None
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r") as f:
                meta = yaml.safe_load(f)
            if meta.get("name") in ["baseline_ols", "enhanced_ols"]:
                ols_pkl_file = glob.glob(os.path.join(
                    "py", "mlartifacts", str(self.experiment), str(self.run),"ols_model","*.pkl"))
                rf_pkl_file = None
                surr_pkl_file = None
            elif meta.get("name") in ["rf","enhanced_rf"]:
                ols_pkl_file =None
                rf_pkl_file = glob.glob(os.path.join(
                    "py", "mlartifacts", str(self.experiment), str(self.run),"rf_model","*.pkl"))
                surr_pkl_file = glob.glob(os.path.join(
                    "py", "mlartifacts", str(self.experiment), str(self.run),"surr_model","*.pkl"))
            else:
                ols_pkl_file = None
                rf_pkl_file = None 
                surr_pkl_file = None
        
        return ols_pkl_file,rf_pkl_file,surr_pkl_file
    
    def _dump_model(self):
        ols_path, rf_path, surr_path = self._extract_model_pkl()
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r") as f:
                meta = yaml.safe_load(f)
            if ols_path and meta.get("name") in ["baseline_ols", "enhanced_ols"]:
                ols_model = sm.load(ols_path[0])
                return ols_model
            elif rf_path and surr_path and meta.get("name") in ["rf", "enhanced_rf"]:
                rf_model = joblib.load(rf_path[0])
                surr_model = joblib.load(surr_path[0])
                return rf_model, surr_model
            else:
                raise FileNotFoundError("No model .pkl file found at the specified path.")
            
            
    def _extract_features(self):
        """Get the feature set of the run"""
        self.mlruns_path_path = os.path.join(
            "py", "mlruns", str(self.experiment), str(self.run), "params"
        )
        with open(os.path.join(self.mlruns_path_path, "factors"), "r") as f:
            features = eval(f.read())
        
        return features



print(rolling_pred(208039388113350502, "0c861f5f9a874e05b04e43bb6341bd96")._extract_model_pkl())
print(rolling_pred(208039388113350502, "0c861f5f9a874e05b04e43bb6341bd96")._extract_features())
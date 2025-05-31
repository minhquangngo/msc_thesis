"""
1. Signals: RF 1 -month aheadPredicted returns for the sector + ratio of SR and LR rewalized vol
2. 
"""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import os
import glob
import shutil
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

class rf_rolling():
    def __init__(self, experiment, run,df,lookback_time):
        self.experiment = experiment
        self.run = run 
        self.df = df
        self.lookback_time = lookback_time
    
    def _var_prep(self):
        features = self._extract_rf_features()
        y = self.df['excess_ret']
        X = self.df[features]
        #train test split
    #-----------Path extractors----------
    def _extract_rf_model_pkl(self):
        """Get the path of that run's model"""
        base_path = os.path.join(
            "py", "mlartifacts", str(self.experiment), str(self.run), "artifacts", "rf_model"
        )
        pkl_files = glob.glob(os.path.join(base_path, "*.pkl"))
        return pkl_files
    
    def _extract_rf_features(self):
        """Get the feature set of the run"""
        base_path = os.path.join(
            "py", "mlruns", str(self.experiment), str(self.run), "params"
        )
        with open(os.path.join(base_path, "factors"), "r") as f:
            features = eval(f.read())
        
        return features



print(rf_rolling(208039388113350502, "0c861f5f9a874e05b04e43bb6341bd96")._extract_rf_model_pkl())
print(rf_rolling(208039388113350502, "0c861f5f9a874e05b04e43bb6341bd96")._extract_rf_features())
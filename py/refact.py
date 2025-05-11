import pandas as pd
import statsmodels.api as sm
import mlflow
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.stats.outliers_influence import variance_inflation_factor
from arch.unitroot import ADF
from abc import abstractmethod, ABC
import hashlib
import json

# Stateless helpers
def _fingerprint(df:pd.DataFrame,y:str, features:list[str], lags:int) -> str:
    """
    Hash the result of the models.
    """
    meta = dict(
        target = y, 
        features =sorted(features),
        lags= lags,
        n_rows = len(df),
        data_hash=hashlib.md5(pd.util.hash_pandas_object(df,index = True).values).hexdigest()
    )
    return hashlib.md5(json.dumps(meta, sort_keys=True).encode()).hexdigest()

def experiment_check(experiment_name:str):
    """
    Check if an experiment with the same name exists or not.
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

#the "*" indicates the class call arguments must be stated in the function. For example target ="", lags = 
class BaseModel(ABC):
    def __init__(
            self,
            *,
            y: str,
            features: list[str],
            lags:int = 5,
            run_name: str = None,
            fama_french_ver: str = None,
            experiment_name: str = None) -> None:
        self.y = y
        self.features = features
        self.lags = lags
        self.run_name = run_name
        self.fama_french_ver = fama_french_ver
        self.experiment_name = experiment_name
    
    def fit(self,df:pd.DataFrame):
        """
        Step 1: Fingerprint the model -> pass to model_fingerprint_check
        - Model_fingerprint_check will see if there is a model
        with that fingerprint in the experiment already
        Step 2: model_fingerprint_check will
         - Check if the experiment name already exists
         - If yes, go into that experiment
    
        Step 3: prep var -> train
        """
        self.hash_fp = _fingerprint(df,self.y, self.features, self.lags)
        fingerprint_check = self.model_fingerprint_check()
        #----check existing run---
        if fingerprint_check is not None:
            print("Skipping- already logged this model") 
            return mlflow.get_run(fingerprint_check)
        #------TODO:add args ||| Fit------
        X,y = self._var_prep() 
        self.model_ = self.train_()
        
    
    #------------abstracts--------------------------
    @abstractmethod
    def _var_prep(self,df:pd.DataFrame): 
        pass
    @abstractmethod
    def train_(self,X,y):
        pass
    @abstractmethod
    def _log_metrics(self): #TODO: add metrics
        pass
    
    #----------internal helpers---------------------
    def model_fingerprint_check(self)-> str | None:
        """
        Use the experiment_check external func to get the list of experiment_id
        On that experiment id, check whether fingerprint already exists or not
         -If exists - skip
        """
        exists = mlflow.seach_runs(
            experiment_ids = [experiment_check(self.experiment_name)],
            filter_string = f"tags.fingerprint='{self.hash_fp}' and tags.status = "completed",
            max_results = 1
        )
        if not exists.empty:
            return exists.loc[0,"run_id"]
        return None
    def _log_run(self,run_name):
        run_name = f'{self.run_name}_OLS' if self.run_name else 'OLS'
        with mlflow.start_run(
            run_name=self.run_name,
            tags={
                "fingerprint":self.hash_fp,
                "status":"completed",
                "sector":run_name,      
                "factors":",".join(self.features), # tags must be strings
                "model_type":"",
                "fama_french_ver": self.fama_french_ver
      }):
            params = {
            'sector':run_name,
            'factors':self.features
        }
            mlflow.log_params(params=params)
            self._log_metrics() #TODO:add args

class olsmodel(BaseModel):
    #--------BaseModel inheritance--------
    def _var_prep(self,df:pd.DataFrame):
        X = sm.add_constant(df[self.features])
        y = df[y]
        return X,y 
    def train_(self,X,y):
        ols_model = sm.OLS(y,X).fit(cov_type='HAC', cov_kwds={'maxlags': self.lags})
        return ols_model
    def _log_metrics(self):
        return super()._log_metrics()

            





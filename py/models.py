from abc import abstractmethod, ABC
import hashlib
import json
import pandas as pd
import statsmodels.api as sm
import mlflow
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import scienceplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, cross_val_predict, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from random import randint, uniform
import shap

seed = 3011
#-----------Stateless helpers---------------
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

#---------------MODELS--------------------------
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
        X,y = self._var_prep(df = df) #TODO: change to train and hold out
        self.model_ = self.train_(X = X,y =y)
        self._log_run(X,y,self.run_name)
    
    #------------abstracts--------------------------
    @abstractmethod
    def _var_prep(self,df:pd.DataFrame): 
        pass
    @abstractmethod
    def train_(self,X,y):
        pass
    @abstractmethod
    def _log_metrics(self,X,y,run_name): 
        pass
    
    #----------internal helpers---------------------
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
    def _log_run(self,X,y,run_name):
        run_name = f'{self.run_name}' if self.run_name else 'blank'
        with mlflow.start_run(
            run_name=run_name,
            tags={
                "fingerprint":self.hash_fp,
                "status":"completed",
                "sector":run_name,      
                "factors":",".join(self.features), # tags must be strings
                "model_type":self.__class__.__name__,
                "fama_french_ver": self.fama_french_ver
      }):
            params = {
            'sector':run_name,
            'factors':self.features
        }
            mlflow.log_params(params=params)
            mlflow.set_tag("fama_french_ver", self.fama_french_ver)
            self._log_metrics(X,y,run_name)

    def plot_residuals(self, fitted_values, residuals, run_name):
        """Plot residuals vs fitted values and save to MLflow"""
        plt.style.use(['science','ieee','apa_custom.mplstyle'])
        plt.figure(figsize=(10, 6))
        plt.scatter(fitted_values, residuals,color= 'C1', alpha=0.5) # C1 = the first color cycle of scienceplot
        plt.xlabel('Fitted Values')
        plt.ylabel('Residuals')
        plt.title(f'Residuals vs Fitted Values - {run_name}')
        plt.axhline(y=0,linestyle='--')
        # Add plot to MLflow
        plt.savefig(f'{run_name}_residuals_plot.png') 
        mlflow.log_artifact(f'{run_name}_residuals_plot.png')
        plt.show()
        plt.close()


class olsmodel(BaseModel):
    """takes the basemodel subclass and bring it to here."""

    #--------BaseModel inheritance--------
    def _var_prep(self,df:pd.DataFrame):
        X = sm.add_constant(df[self.features])
        y = df[self.y]
        return X,y 
    def train_(self,X,y):
        ols_model = sm.OLS(y,X).fit(cov_type='HAC', cov_kwds={'maxlags': self.lags})
        return ols_model
    
    def _log_metrics(self,X,y,run_name):
        self.adf_stat, self.adf_p, _, _, self.crit_vals, _ = adfuller(self.model_.resid, maxlag=None, autolag='AIC')
        metrics = {
            'durbin_watson': durbin_watson(self.model_.resid),
            'Breuschpagan': het_breuschpagan(self.model_.resid, self.model_.model.exog)[1],
            'aic': self.model_.aic,
            'bic': self.model_.bic,
            'centered_tss': self.model_.centered_tss,
            'condition_number': self.model_.condition_number,
            'df_model': self.model_.df_model,
            'df_resid': self.model_.df_resid,
            'ess': self.model_.ess,
            'f_pvalue': self.model_.f_pvalue,
            'fvalue': self.model_.fvalue,
            'llf': self.model_.llf,
            'mse_model': self.model_.mse_model,
            'mse_resid': self.model_.mse_resid,
            'mse_total': self.model_.mse_total,
            'rsquared': self.model_.rsquared,
            'R2_adj': self.model_.rsquared_adj,
            'scale': self.model_.scale,
            'ssr': self.model_.ssr,
            'uncentered_tss': self.model_.uncentered_tss,
            'adf_stats': self.adf_stat,
            'adf_pval': self.adf_p
            }
        mlflow.log_metrics(metrics)
        
        #coefs and p-vals
        for coefname, coef in self.model_.params.items():
            mlflow.log_param(coefname, coef)
        for coefname, pvalue in self.model_.pvalues.items():
            mlflow.log_param(f'pval {coefname}', pvalue)
        
        # assumption tests
        jb_stat, jb_p, _, _ = jarque_bera(self.model_.resid)
        mlflow.log_metric('jarque_bera_normality', jb_p)

        # VIF
        vif = {}
        for i in range (1,X.shape[1]):
            col = X.columns[i]
            vif[col] = variance_inflation_factor(X.values,i)
        
        mlflow.log_dict(vif, 'vif.json')

        #----------Plotting-----------
        self.plot_residuals(self.model_.fittedvalues, self.model_.resid, run_name)
        
        # full model summary
        mlflow.log_text(self.model_.summary().as_text(), 'ols_summary.txt')

class randomforest(BaseModel):
    """
    Put randomforest on top of basemodel:
    - expanding window walk forward
    - log the final model params and metrics
    """
    def _var_prep(self, df):
        #TODO: hold out sample
        hold_out_size = 360
        break_point = len(df) - hold_out_size
        
        X= df[self.features]
        y=df[self.y]
        X_train = X.iloc[:break_point]
        X_hold = X.iloc[break_point:]

        y_train = y.iloc[:break_point]
        y_hold = y.iloc[break_point:]
        return X_train, X_hold, y_train, y_hold
    def train_(self,X_train,,X_hold,y_train,y_hold):
        #test size = 80 due to 4quarters*20years
        #TODO: Could add an argument here to allow to switch between rolling and expanding
        split_timeseries = TimeSeriesSplit(n_splits=5, test_size=28, gap=5) #TODO: test size determine
        params = {
            "n_estimators":      randint(200, 1000),
            "max_depth":         randint(3, 20),     # None is allowed via extra prob mass:
            "min_samples_split": randint(2, 10),
            "min_samples_leaf":  randint(1, 10),
            "max_features":      uniform(0.3, 0.7)   # fraction of features
        }
        rf = RandomForestRegressor(
            random_state= seed,
            oob_score= True
        )
        self.random_search = RandomizedSearchCV(
            estimator= rf, 
            param_distributions= params,
            n_iter = 5, #TODO: change numb iter
            scoring = 'mean_squared_error',
            n_jobs = -1,
            refit = True,
            cv = split_timeseries,
            verbose = 0,
            random_state = seed
        ).fit(X_train,y_train)

        self.best_rf = self.random_search.best_estimator_
        
        #-----------Sample Prediction------------
        pred_y_sample = self.random_search.predict(X_train)
        #the prediction here is for every fold.cross_val_predict internally builds a full-sized y_pred aligned with y.index,
        self.pred_fold = cross_val_predict(
            estimator = self.best_rf,X=X_train, y=y_train,cv=split_timeseries,n_jobs= -1
            )
        r2_fold_array = cross_val_score(    
            estimator= self.best_rf,
            X=X_train,y=y_train,
            cv=split_timeseries,
            n_jobs=-1
        )

        #------------Hold-out-----------------------
        pred_y_hold = self.random_search.predict(X_hold)
        
        self.pred_fold_hold = cross_val_predict(
            estimator = self.best_rf,X=X_hold, y=y_hold,cv=split_timeseries,n_jobs= -1
            )

        r2_fold_array = cross_val_score(    
            estimator= self.best_rf,
            X=X_hold,y=y_hold,
            cv=split_timeseries,
            n_jobs=-1
        )
        #-----------Metrics------------------------------
        self.mse_sample = mean_squared_error(y_true = y_hold,y_pred = pred_y_sample)
        self.mse_fold = mean_squared_error(y_true= y_hold, y_pred= self.pred_fold)
        self.mse_bootstrap_oob = mean_squared_error (y_true= y_hold, y_pred = self.best_rf.oob_prediction_)
        
        self.mse_sample_hold = mean_squared_error(y_true = y_hold,y_pred = pred_y_hold)
        self.mse_fold_hold = mean_squared_error(y_true= y_hold, y_pred= self.pred_fold_hold)

        self.r2_sample = r2_score(y_true=y_hold,y_pred=pred_y_sample)
        self.r2_glob_fold = r2_score(y_true = y_hold, y_pred = self.pred_fold) 
        self.r2_bootstrap_oob = r2_score(y_true=y_hold,y_pred = self.best_rf.oob_prediction_)
        self.r2_loc_fold_mean = r2_fold_array.mean()
        self.r2_loc_fold_std = r2_fold_array.std()
        
        self.mae = mean_absolute_error(y_true= y_hold, y_pred = self.pred_fold)

        self.resid_sample = y_hold - pred_y_sample
        self.resid_oof = y_hold - self.pred_fold

        #------------SHAP-------------
        shap_exp = shap.TreeExplainer(self.best_rf)
        shap_values = shap_exp.shap_values(X_hold)
        
        return self.best_rf
    
    def _log_metrics(self, X, y, run_name):
        self.adf_stat, self.adf_p, _, _, self.crit_vals, _ = adfuller(self.resid_oof, maxlag=None, autolag='AIC')
        metrics = {
            'mse_model': self.mse_fold,
            'rsquared_sample': self.r2_sample,
            'rsquared_global_oof': self.r2_glob_fold,
            'rsquared_bootstrap_oob': self.r2_bootstrap_oob,
            'rsquared_local_oof':self.r2_loc_fold_mean,
            'rsquared_local_oof_std': self.r2_loc_fold_std,
            'mae':self.mae,
            'out_of_bag_score':self.best_rf.oob_score_,
            'adf_stats': self.adf_stat,
            'adf_pval': self.adf_p
        }
        mlflow.log_metrics(metrics)
        
        # --------------------VIF---------------
        vif = {}
        for i in range (1,X.shape[1]):
            col = X.columns[i]
            vif[col] = variance_inflation_factor(X.values,i)
        mlflow.log_dict(vif, 'vif.json')
        #---------------feat imp-----------
        rf_summary =[]
        for colname, feat_imp in zip(X.columns, self.best_rf.feature_importances_):
            rf_summary.append(f"{colname}:{feat_imp}")
        mlflow.log_text(rf_summary, 'rf_summary.txt')
        
        #-------------Residuals----------
        self.plot_residuals(self.pred_fold,self.resid_oof, self.run_name) 

        #------------SHAP----------------
        shap_exp = shap.TreeExplainer(self.best_rf)
        shap_values = shap_exp(X)
        plt.style.use(['science','ieee','apa_custom.mplstyle'])
        plt.figure(figsize=(10,6))
        shap.summary_plot(shap_values,X,show=False)
        plt.xlabel('Fitted values')
        plt.ylabel('SHAP values')
        plt.title(f'Residuals vs Fitted Values - {run_name}')
        # Add plot to MLflow
        plt.savefig(f'{run_name}_shap_plot.png') 
        mlflow.log_artifact(f'{run_name}_shap_plot.png')
        plt.show()
        
        
        #TODO:  shap, coeff from simonian, plot shap, lime, surrogate as well.
        #TODO: shap hold out sample 2018







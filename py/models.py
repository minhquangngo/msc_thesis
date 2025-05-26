from abc import abstractmethod, ABC
import hashlib
import json
import mlflow.models
import mlflow.models
import mlflow.models.signature
import mlflow.sklearn
import mlflow.statsmodels
import pandas as pd
import numpy as np
import statsmodels.api as sm
import mlflow
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import matplotlib as mpl
import scienceplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV,GridSearchCV, cross_val_predict, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
from scipy.stats import randint, uniform
import shap
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree
from sklearn.inspection import PartialDependenceDisplay


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
        X_fit,X_hold,y_fit,y_hold = self._var_prep(df = df) # Renamed variables
        self.model_ = self.train_(X_fit= X_fit,X_hold = X_hold,y_fit =y_fit,y_hold = y_hold)
        self._log_run(X_fit,X_hold,y_fit,y_hold,self.run_name)
    
    #------------abstracts--------------------------
    @abstractmethod
    def train_(self,X_fit,X_hold,y_fit,y_hold):
        pass
    @abstractmethod
    def _log_metrics(self,X_fit,X_hold,y_fit,y_hold,run_name): 
        pass
    
    #----------internal helpers---------------------
    def _var_prep(self, df:pd.DataFrame):
        hold_out_size = 360
        break_point = len(df) - hold_out_size
        
        X= df[self.features]
        y=df[self.y]
        X_fit = X.iloc[:break_point]
        X_hold = X.iloc[break_point:]

        y_fit = y.iloc[:break_point]
        y_hold = y.iloc[break_point:]
        return X_fit, X_hold, y_fit, y_hold
    
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
   
    def _log_run(self,X_fit,X_hold,y_fit,y_hold,run_name):
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
            mlflow.log_params(params)
            mlflow.set_tag("fama_french_ver", self.fama_french_ver)
            self._log_metrics(X_fit,X_hold,y_fit,y_hold,run_name)

    def plot_residuals(self, fitted_values, residuals, run_name,sample_or_hold):
        """Plot residuals vs fitted values and save to MLflow"""
        plt.style.use(['science','ieee','apa_custom.mplstyle'])
        plt.figure(figsize=(10, 6))
        plt.scatter(fitted_values, residuals, alpha=0.5)
        plt.xlabel('Fitted Values')
        plt.ylabel('Residuals')
        plt.axhline(y=0,linestyle='--')
        img_path = f'{run_name}_{sample_or_hold}_residuals_plot.png'
        plt.savefig(img_path)
        mlflow.log_artifact(img_path)
        plt.show()
        print(f"Displayed: Residuals plot for {run_name} - {sample_or_hold}")
        plt.close()


class olsmodel(BaseModel):
    """takes the basemodel subclass and bring it to here."""
    #--------BaseModel inheritance--------

    def train_(self,X_fit,X_hold,y_fit,y_hold):
        # Fitted OLS 
        self.X_fit = sm.add_constant(X_fit)
        ols_model = sm.OLS(y_fit,self.X_fit).fit(cov_type='HAC', cov_kwds={'maxlags': self.lags})
        return ols_model
    
    def _log_metrics(self,X_fit,X_hold,y_fit,y_hold,run_name):
        self.adf_stat, self.adf_p, _, _, self.crit_vals, _ = adfuller(self.model_.resid, maxlag=None, autolag='AIC')

        y_pred_train = self.model_.predict(self.X_fit)
        rmse_insample = np.sqrt(mean_squared_error(y_fit,y_pred_train))
        resid_sample = y_fit - y_pred_train
        
        X_hold = sm.add_constant(X_hold, has_constant='add')
        y_pred_hold = self.model_.predict(X_hold)
        rmse_hold = np.sqrt(mean_squared_error(y_hold,y_pred_hold))
        resid_hold = y_hold - y_pred_hold
        rsquared_hold = r2_score(y_hold,y_pred_hold)
        mae_ols_hold = mean_absolute_error(y_hold,y_pred_hold)

        #--------Sig-----------------
        ols_input_example = sm.add_constant(X_fit.iloc[:5])
        ols_output_example = self.model_.predict(ols_input_example)
        ols_sig = mlflow.models.signature.infer_signature(ols_input_example,ols_output_example)
        mlflow.statsmodels.log_model(
            statsmodels_model= self.model_,
            artifact_path= "ols_model",
            input_example= ols_input_example,

        #--------Metrics-----------------
        )
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
            'rmse_insample': rmse_insample,
            'rmse_hold': rmse_hold,
            'rsquared': self.model_.rsquared,
            'rsquared_adj': self.model_.rsquared_adj,
            'rsquared_hold': rsquared_hold,
            'mae_hold':mae_ols_hold,
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
        
        # ----------------assumption tests----------------
        jb_stat, jb_p, _, _ = jarque_bera(self.model_.resid)
        mlflow.log_metric('jarque_bera_normality', jb_p)

        # --------------VIF-------------------
        vif = {}
        for i in range (1,self.X_fit.shape[1]):
            col = self.X_fit.columns[i]
            vif[col] = variance_inflation_factor(self.X_fit.values,i)

        mlflow.log_dict(vif, 'vif.json')

        #----------Plotting-----------
        self.plot_residuals(self.model_.fittedvalues, self.model_.resid, run_name, sample_or_hold= 'sample')#"Sample:Residuals vs Fitted Values"
        self.plot_residuals(y_pred_hold, resid_hold, run_name, sample_or_hold= 'hold')#"Hold out: Residuals vs Fitted Values"
        # full model summary
        mlflow.log_text(self.model_.summary().as_text(), 'ols_summary.txt')
#-------------------------------------------------------------------------------------------------
##---------------------------------------RANDOM FOREST----------------------------------------------------------
#-------------------------------------------------------------------------------------------------

class randomforest(BaseModel):
    """
    Put randomforest on top of basemodel:
    - expanding window walk forward
    - log the final model params and metrics
    """

    def train_(self,X_fit,X_hold,y_fit,y_hold):
        split_timeseries = TimeSeriesSplit(n_splits=5, test_size=28, gap=5)
        params = {
            'n_estimators':      [200, 500, 1000, 2000],
            'max_depth':         [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf':  [1, 2, 4],
            'max_features':      ['sqrt', 'log2', None],
            'bootstrap':         [True]
        }
        rf = RandomForestRegressor(
            random_state= seed,
            oob_score= True,
            n_jobs = 1
        )
        self.random_search = RandomizedSearchCV(
            estimator= rf, 
            param_distributions= params,
            n_iter = 5,
            scoring = 'r2',
            n_jobs = -1,
            refit = True,
            cv = split_timeseries,
            verbose = 0,
            random_state = seed
        ).fit(X_fit,y_fit)

        self.best_rf = self.random_search.best_estimator_
        
        #-----------Sample Prediction------------
        self.pred_y_sample = self.best_rf.predict(X_fit)
        #-----------Fold prediction------------
        self.pred_fold = np.full(len(y_fit), np.nan, dtype=np.float64)
        mse_fold_array = [] 
        r2_fold_array = []
        for train_idx, test_idx in split_timeseries.split(X_fit):
            X_tr, X_te = X_fit.iloc[train_idx], X_fit.iloc[test_idx]
            y_tr, y_te = y_fit.iloc[train_idx], y_fit.iloc[test_idx]
            model = RandomForestRegressor(**self.best_rf.get_params())
            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)
            self.pred_fold[test_idx] = preds
            r2_fold_array.append(r2_score(y_te, preds))
            mse_fold_array.append(mean_squared_error(y_te,preds))
        r2_fold_array = np.array(r2_fold_array)
        mse_fold_array = np.array(mse_fold_array)

        #------------Hold-out-----------------------
        self.pred_y_hold = self.random_search.predict(X_hold)

        #-----------Metrics------------------------------
        self.mse_sample = mean_squared_error(y_true = y_fit,y_pred = self.pred_y_sample)
        fold_mask = ~np.isnan(self.pred_fold)
        self.mse_fold = mean_squared_error(y_true= y_fit[fold_mask], y_pred= self.pred_fold[fold_mask])
        self.mse_hold = mean_squared_error(y_true = y_hold,y_pred = self.pred_y_hold)
        
        self.r2_sample = r2_score(y_true=y_fit,y_pred=self.pred_y_sample)
        self.r2_hold = r2_score(y_true= y_hold, y_pred = self.pred_y_hold) 
        self.r2_loc_fold_mean = r2_fold_array.mean()
        self.r2_loc_fold_std = r2_fold_array.std()

        self.mae = mean_absolute_error(y_true= y_hold, y_pred = self.pred_y_hold)

        self.resid_sample = y_fit - self.pred_y_sample
        self.resid_hold = y_hold - self.pred_y_hold

        #------------SHAP-------------
        shap_exp = shap.Explainer(self.best_rf)
        self.shap_values = shap_exp(X_hold)
        
        #------------Surrogate--------
        self.best_surrogate_model, self.surrogate_r2_sample, self.surrogate_rmse_sample, self.surrogate_r2_hold, self.surrogate_rmse_hold = self.surrogate_(X_fit, X_hold, self.pred_y_sample, self.pred_y_hold)
        

        fig, ax = plt.subplots(figsize=(18, 12))  
        plot_tree(
            self.best_surrogate_model,
            feature_names=X_fit.columns.tolist(),  
            filled=True,
            max_depth=3,
            fontsize=11,
            ax=ax
        )
        for txt in ax.texts:
            txt.set_text(txt.get_text().replace('≤','<='))
        plt.tight_layout()
        plt.axis('off')  # Optional: Remove axes
        
        self.surrogate_text = export_text(self.best_surrogate_model, max_depth=3)
        
        return self.best_rf
    
    def _log_metrics(self, X_fit,X_hold, y_fit,y_hold, run_name):
        #-------------Logging models------------
        rf_input_example = X_fit.iloc[:5]  # First 5 rows as a sample input
        rf_output_example = self.best_rf.predict(rf_input_example)
        rf_sig = mlflow.models.signature.infer_signature(rf_input_example,rf_output_example)
        mlflow.sklearn.log_model(
            sk_model   = self.best_rf,
            artifact_path = "rf_model",
            input_example = rf_input_example,
            registered_model_name = f"rf_{self.fama_french_ver}_{self.run_name}_{self.experiment_name}",
            signature=rf_sig
    )
        surr_output_example = self.best_surrogate_model.predict(rf_input_example)
        surr_sig = mlflow.models.signature.infer_signature(rf_input_example,surr_output_example)
        mlflow.sklearn.log_model(
            sk_model= self.best_surrogate_model,
            artifact_path= 'surr_model',
            input_example= rf_input_example,
            registered_model_name = f"surr_{self.fama_french_ver}_{self.run_name}_{self.experiment_name}",
            signature= surr_sig
        )

        #--------------Logging metrics---------------
        #TODO: see which residuals (hold or sample) should we use for adf
        self.adf_stat, self.adf_p, _, _, self.crit_vals, _ = adfuller(self.resid_sample, maxlag=None, autolag='AIC')
        metrics = {
            'rf_mse_fitsample': self.mse_sample,
            'rf_mse_fold': self.mse_fold,
            'rf_mse_hold': self.mse_hold,
            'rsquared_sample': self.r2_sample,
            'rsquared_holdout': self.r2_hold,
            'rsquared_local_oof':self.r2_loc_fold_mean,
            'rsquared_local_oof_std': self.r2_loc_fold_std,
            'mae':self.mae,
            'out_of_bag_score':self.best_rf.oob_score_,
            'adf_stats': self.adf_stat,
            'adf_pval': self.adf_p,
            'surrogate_rmse_sample': self.surrogate_rmse_sample,
            'surrogate_r2_sample': self.surrogate_r2_sample,
            'surrogate_rmse_hold': self.surrogate_rmse_hold,
            'surrogate_r2_hold': self.surrogate_r2_hold 
        }
        mlflow.log_metrics(metrics)
        mlflow.log_text(self.surrogate_text, 'surrogate_tree.txt')
        # --------------------VIF---------------
        #TODO: explain in the paper why vif is calc this way
        # 1. Extract the raw values and standardize each column
        X_vif = X_fit.values
        means = X_vif.mean(axis=0)
        stds  = X_vif.std(axis=0, ddof=0)
        X_std = (X_vif - means) / stds

        # 2. Compute the correlation matrix and invert it
        corr_mat = np.corrcoef(X_std, rowvar=False)
        inv_corr = np.linalg.inv(corr_mat)

        # 3. VIF for each feature is just the diagonal of the inverted correlation matrix
        vif = {col: float(inv_corr[i, i]) 
            for i, col in enumerate(X_fit.columns)}

        mlflow.log_dict(vif, 'vif.json')
        #---------------feat imp-----------
        rf_summary =[]
        for colname, feat_imp in zip(X_fit.columns, self.best_rf.feature_importances_):
            rf_summary.append(f"{colname}:{feat_imp}")
        mlflow.log_text('\n'.join(rf_summary), 'rf_summary.txt') #needs to joing this as a string

        #----------------permutation importnance MDA(simonian 2019)---------
        perm_result_rf = permutation_importance(
            self.best_rf, X_hold, y_hold, n_repeats=20, random_state=seed, n_jobs=1
        ) #TODO: adjust number of repeats

        perm_result_surr = permutation_importance(
            self.best_surrogate_model, X_hold, y_hold, n_repeats=20, random_state=seed, n_jobs=-1
        )
        raw_perm_imp_rf = self.plot_and_log_permutation_importance(
            perm_result_rf,
            X_hold,
            run_name= self.run_name,
            surr_or_rf = 'rf')
        
        # raw_perm_imp_surr = self.plot_and_log_permutation_importance(
        #     perm_result_surr,
        #     X_hold,
        #     run_name= self.run_name,
        #     surr_or_rf = 'surr')
        
        #-------------Residuals-RF----------
        self.plot_residuals(self.pred_y_sample,self.resid_sample, self.run_name, sample_or_hold='sample') #'RF Sample: Residuals vs Fitted Values')
        self.plot_residuals(self.pred_y_hold,self.resid_hold, self.run_name, sample_or_hold='hold') #'RF Hold out: Residuals vs Fitted Values'

        #------------Residuals-Surr----------
        surr_pred_sample = self.best_surrogate_model.predict(X_fit)
        surr_pred_hold = self.best_surrogate_model.predict(X_hold)
        surr_resid_sample = y_fit - surr_pred_sample
        surr_resid_hold = y_hold - surr_pred_hold
        # self.plot_residuals(surr_pred_sample,surr_resid_sample,self.run_name, sample_or_hold='sample') #' Surrogate Sample: Residuals vs Fitted Values'
        # self.plot_residuals(surr_pred_hold,surr_resid_hold,self.run_name, sample_or_hold='hold') #' Surrogate Hold out: Residuals vs Fitted Values'

        #------------SHAP----------------
        plt.style.use(['science','ieee','apa_custom.mplstyle'])
        plt.figure(figsize=(10,6))
        shap.plots.beeswarm(self.shap_values, show=False)
        plt.savefig(f'{run_name}_shap_plot.png') 
        mlflow.log_artifact(f'{run_name}_shap_plot.png')
        plt.show()
        print(f"Displayed: SHAP beeswarm plot for {run_name}")
        plt.close()

        #-----------Simonian coefficient- RF-----------
        rfi_rf = raw_perm_imp_rf/raw_perm_imp_rf.sum()
        elasticity_rf = {
            col: np.mean(self.pred_y_hold / (X_hold[col].to_numpy() + 1e-8))
            for col in X_hold.columns
        }
        pseudo_beta_rf = {
            feature: float(rfi_rf[i] * elasticity_rf[feature])
            for i, feature in enumerate(X_hold.columns)
}
        mlflow.log_dict(pseudo_beta_rf, 'rf_pseudo_beta.json')

        # rfi_surr = raw_perm_imp_surr/raw_perm_imp_surr.sum()
        # elasticity_surr = {
        #     col: np.mean(self.pred_y_hold / (X_hold[col].to_numpy() + 1e-8))
        #     for col in X_hold.columns
        # }

        #------------------Partial Dependence Plots----------------
        top10_idx = np.argsort(self.best_rf.feature_importances_)[-10:]
        top10 = [
            (self.features[i], self.best_rf.feature_importances_[i])
            for i in top10_idx
        ]
        top10.sort(key=lambda x: x[1], reverse=True)
        top10_names = [feat for feat, imp in top10]
        
                # Set up the figure
        n_features = len(top10_names)
        n_cols = 2  # Or 3 for a wider screen
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, ax = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4 * n_rows))
        ax = ax.flatten()  # flatten for easy iteration

        for i, feat in enumerate(top10_names):
            disp = PartialDependenceDisplay.from_estimator(
                estimator=self.best_rf,
                X=X_fit,
                features=[feat],
                grid_resolution=50,
                ax=ax[i]
            )
            ax[i].set_title(feat)
            # Optional: Set limits if needed
            # ax[i].set_xlim([custom_min, custom_max])

        # Remove empty subplots if n_features < n_rows * n_cols
        for j in range(i+1, len(ax)):
            fig.delaxes(ax[j])

        plt.tight_layout()
        fig.savefig(f"{run_name}_rf_pdp.png")
        mlflow.log_artifact(f"{run_name}_rf_pdp.png")
        plt.show()

        

    ###----------------Helper func----------------------------
    #---------------------------------------------------------
    def surrogate_(self,X_fit,X_hold,pred_y_sample,pred_y_hold):
        param_grid = {
            'max_depth':         range(1, 11),
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf':  [1, 2, 5, 10],
            'max_features':      ['sqrt', 'log2', None],
            'ccp_alpha':         [0.0, 0.001, 0.01, 0.1],
        }
        tree_surrogate = DecisionTreeRegressor()
        surrogate_grid = GridSearchCV(
            estimator= tree_surrogate,
            param_grid= param_grid,
            cv= 5, # TODO: adjust this later
            scoring = 'r2',
            refit =True,
            n_jobs=-1,
            verbose = 0
        )
        surrogate_grid.fit(X_fit,pred_y_sample)
        best_surrogate_model = surrogate_grid.best_estimator_
        # Sample pred
        surrogate_pred_sample = surrogate_grid.predict(X_fit)
        surrogate_rmse_sample = np.sqrt(mean_squared_error(pred_y_sample, surrogate_pred_sample))
        surrogate_r2_sample = r2_score(pred_y_sample, surrogate_pred_sample)
        # Hold-out prediction
        surrogate_pred_hold =  surrogate_grid.predict(X_hold)
        surrogate_rmse_hold = np.sqrt(mean_squared_error(pred_y_hold, surrogate_pred_hold))
        surrogate_r2_hold = r2_score(pred_y_hold, surrogate_pred_hold)
        return best_surrogate_model,surrogate_r2_sample, surrogate_rmse_sample, surrogate_r2_hold, surrogate_rmse_hold
    
    def plot_and_log_permutation_importance(
        self,
        perm_result,
        X_hold,
        run_name: str,
        surr_or_rf = None
    ):
        raw_perm_imp = perm_result.importances_mean
        sorted_idx   = raw_perm_imp.argsort()  # ascending
        boxplot_data = [perm_result.importances[i] for i in sorted_idx]
        labels       = X_hold.columns[sorted_idx]

        plt.style.use(['science','ieee','apa_custom.mplstyle'])
        plt.figure(figsize=(10, 6))
        plt.boxplot(boxplot_data, vert=True, labels=labels)
        plt.ylabel("Permutation Importance (mean decrease accuracy)")
        plt.tight_layout()
        img_path = f"{surr_or_rf}_{run_name}_permutation_importance.png"
        plt.savefig(img_path)
        mlflow.log_artifact(img_path)
        plt.show()
        print(f"Displayed: Permutation Importance plot for {surr_or_rf} - {run_name}")
        plt.close()

        perm_imp_dict = {
            feature: {
                "mean": float(perm_result.importances_mean[i]),
                "std":  float(perm_result.importances_std[i])
            } for i, feature in enumerate(X_hold.columns)
        }
        mlflow.log_dict(perm_imp_dict, f"{surr_or_rf}_{run_name}_permutation_importance.json")
        return raw_perm_imp

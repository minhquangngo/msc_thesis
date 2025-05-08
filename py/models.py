import pandas as pd
import statsmodels.api as sm
import mlflow
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.stats.outliers_influence import variance_inflation_factor
from arch.unitroot import ADF
import hashlib
import json

def _fingerprint(df,y, features, lags):
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


def ols_sm(df, y, features, lags=5, run_name=None, fama_french_ver=None, experiment_name=None):
    """
    Fitting OLS with HAC, adjusting for heteroskedacity autocorrelation up to a certain
    point (defined with cov_kwds). This adjust the problem of the homoskedacity assumption. We allow 
    lags of up to 5 business days

    - Durbin Watson: Auto correlation check
    """
    #missing params check
    missing = []
    params = {
        "run_name": run_name,
        "fama_french_ver": fama_french_ver,
        "experiment_name": experiment_name  
    }
    for name, val in params.items():
        # If the value is exactly None, record its name
        if val is None:
            missing.append(name)

    if missing:
        raise ValueError(
            f"The following parameter(s) must be provided and not None: {', '.join(missing)}"
        )
    
    #experiment exist check    
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f'Experiment {experiment_id} created')
    else:
        experiment_id = experiment.experiment_id
        print(f"Experiment already exists with ID: {experiment_id}")
        
            
    mlflow.set_experiment(experiment_name)

    #model exists check
    hash_fp = _fingerprint(df,y,features,lags)
    exists = mlflow.search_runs(
        experiment_ids= [experiment_id],
        filter_string=f"tags.fingerprint = '{hash_fp}' and tags.status = 'completed'",
        max_results=1
    )

    if not exists.empty:
        print("Skipping- already logged this model") 
        return mlflow.get_run(exists.loc[0,"run_id"])

    #Model training
    X = sm.add_constant(df[features])
    y = df[y]
    ols_model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': lags})

    
    run_name = f'{run_name}_OLS' if run_name else 'OLS'
    with mlflow.start_run(
        run_name=run_name, 
        tags={
        "fingerprint":hash_fp,
        "status":"completed",
        "sector":run_name,      
        "factors":",".join(features), # tags must be strings
        "model_type":"OLS_raw_baseline",
        "fama_french_ver": fama_french_ver
      }):
        
        #params
        params = {
            'sector':run_name,
            'factors':features
        }
        mlflow.log_param('sector', run_name)
        mlflow.log_param('factors', features)
        mlflow.set_tag("model_type", "OLS_raw")
        mlflow.set_tag("fama_french_ver", fama_french_ver)


        #metrics
        metrics = {
            'R2_adj': ols_model.rsquared_adj,
            'durbin_watson': durbin_watson(ols_model.resid),
            'Breuschpagan': het_breuschpagan(ols_model.resid, ols_model.model.exog)[1],
            'aic': ols_model.aic,
            'bic': ols_model.bic,
            'centered_tss': ols_model.centered_tss,
            'condition_number': ols_model.condition_number,
            'df_model': ols_model.df_model,
            'df_resid': ols_model.df_resid,
            'ess': ols_model.ess,
            'f_pvalue': ols_model.f_pvalue,
            'fvalue': ols_model.fvalue,
            'llf': ols_model.llf,
            'mse_model': ols_model.mse_model,
            'mse_resid': ols_model.mse_resid,
            'mse_total': ols_model.mse_total,
            'rsquared': ols_model.rsquared,
            'rsquared_adj': ols_model.rsquared_adj,
            'scale': ols_model.scale,
            'ssr': ols_model.ssr,
            'uncentered_tss': ols_model.uncentered_tss
        }
        mlflow.log_metrics(metrics)
        
        #coefs and p-vals
        for coefname, coef in ols_model.params.items():
            mlflow.log_param(coefname, coef)
        for coefname, pvalue in ols_model.pvalues.items():
            mlflow.log_param(f'pval {coefname}', pvalue)
        
        # assumption tests
        jb_stat, jb_p, _, _ = jarque_bera(ols_model.resid)
        mlflow.log_metric('jarque_bera_normality', jb_p)

        # VIF
        vif = {}
        for i in range (1,X.shape[1]):
            col = X.columns[i]
            vif[col] = variance_inflation_factor(X.values,i)
        
        mlflow.log_dict(vif, 'vif.json')

        # ADF stationarity
        adf_p = ADF(ols_model.resid).pvalue
        mlflow.log_metric('adf_resid_pval', adf_p)
        
        # full model summary
        mlflow.log_text(ols_model.summary().as_text(), 'ols_summary.txt')

if __name__ == '__main__':
    print("Import to use")
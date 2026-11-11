import pandas as pd
import numpy as np

import xgboost as xgb

import lightgbm as lgbm

from functools import partial
import time
import os

from features import FEATURE_COLUMNS
from backtest import run_backtest
from evaluation import score_forecasts

def score_params(df, cutoffs, forecast_fn, forecast_col, params, horizon_days=7):
    locked_fn = partial(forecast_fn, params=params)

    bt = run_backtest(df, cutoffs, horizon_days, locked_fn)

    assert len(bt) == len(cutoffs) * df['Store'].nunique() * horizon_days, "failed backtest"

    score = score_forecasts(bt, [forecast_col])

    rmse, mae, wape = float(score.iloc[0]['RMSE']), float(score.iloc[0]['MAE']), float(score.iloc[0]['WAPE'])

    return {'RMSE': rmse, 'MAE': mae, 'WAPE': wape}


def sample_param(spec, rng):
    kind, low, high = spec
    if kind == 'int':
        return int(rng.integers(low, high+1))
    elif kind == 'float':
        return float(rng.uniform(low, high))
    elif kind == 'log':
        return float(np.exp(rng.uniform(np.log(low), np.log(high))))
    else:
        raise ValueError(f'incorrect type indentifier {kind}')
    

def random_search(df, cutoffs, forecast_fn, forecast_col, base_params, search_space, n_trials, seed=42, path=None):

    def run_trial(trial, params, settings):
        start = time.perf_counter()
        
        score = score_params(df, cutoffs, forecast_fn, forecast_col, params)
        
        seconds = time.perf_counter() - start

        print(f"trial {trial}, WAPE {score['WAPE']:.2f}%, {seconds:.0f}s")

        return {'trial': trial, **settings, **score, 'seconds': seconds}

    os.makedirs(os.path.dirname(path), exist_ok=True) #type:ignore

    rng = np.random.default_rng(seed)

    results = []

    results.append(run_trial(0, base_params, {key: base_params.get(key) for key in search_space}))

    for i in range(1, n_trials + 1):
        settings = {key: sample_param(spec, rng) for key, spec in search_space.items()}
        params = {**base_params, **settings}

        results.append(run_trial(i, params, settings))

        if path is not None:
            pd.DataFrame(results).sort_values('WAPE').to_csv(path, index=False)

    return pd.DataFrame(results).sort_values('WAPE')
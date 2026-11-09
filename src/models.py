import pandas as pd
import numpy as np

import xgboost as xgb

import lightgbm as lgbm

from features import FEATURE_COLUMNS

XGB_PARAMS = {
    'tree_method': 'hist',
    'enable_categorical': True,
    'n_estimators': 300,
    'learning_rate': 0.05,
    'max_depth': 6,
    'random_state': 42
}

LGBM_PARAMS = {
    'n_estimators': 300,
    'learning_rate': 0.05,
    'num_leaves': 63,
    'random_state': 42,
    'verbose': -1
}

def add_xgb_forecast(train, test, cutoff, params=XGB_PARAMS):

    X_train = train[FEATURE_COLUMNS]
    y_train = train['Sales']

    model = xgb.XGBRegressor(**params)

    model.fit(X_train, y_train)

    preds = model.predict(test[FEATURE_COLUMNS])

    preds_clipped = np.clip(preds, 0, None)

    assert len(preds_clipped) == len(test), "The number of predictions should match the number of test samples."
    assert not np.isnan(preds_clipped).any(), "Predictions should not contain any NaN values."

    return test.assign(xgb_forecast=preds_clipped)

def add_lgbm_forecast(train, test, cutoff, params=LGBM_PARAMS):

    X_train = train[FEATURE_COLUMNS]
    y_train = train['Sales']

    model = lgbm.LGBMRegressor(**params)

    model.fit(X_train, y_train)

    preds = model.predict(test[FEATURE_COLUMNS])

    preds_clipped = np.clip(np.asarray(preds), 0, None)

    assert len(preds_clipped) == len(test), "The number of predictions should match the number of test samples."
    assert not np.isnan(preds_clipped).any(), "Predictions should not contain any NaN values."

    return test.assign(lgbm_forecast=preds_clipped)

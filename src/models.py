import pandas as pd
import numpy as np
import xgboost as xgb
from features import FEATURE_COLUMNS

XGB_PARAMS = {
    'tree_method': 'hist',
    'enable_categorical': True,
    'n_estimators': 300,
    'learning_rate': 0.05,
    'max_depth': 6,
    'random_state': 42
}

def add_xgb_forecast(train, test, cutoff):

    X_train = train[FEATURE_COLUMNS]
    y_train = train['Sales']

    model = xgb.XGBRegressor(**XGB_PARAMS)

    model.fit(X_train, y_train)

    preds = model.predict(test[FEATURE_COLUMNS])

    preds_clipped = np.clip(preds, 0, None)

    assert len(preds_clipped) == len(test), "The number of predictions should match the number of test samples."
    assert not np.isnan(preds_clipped).any(), "Predictions should not contain any NaN values."

    return test.assign(xgboost_forecast=preds_clipped)


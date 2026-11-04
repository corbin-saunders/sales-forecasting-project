import sklearn.metrics
import pandas as pd

def evaluate_forecast(test_series, forecast):
    RMSE = sklearn.metrics.root_mean_squared_error(test_series, forecast)
    MAE = sklearn.metrics.mean_absolute_error(test_series, forecast)
    WAPE = 100 * (abs(test_series - forecast).sum()) / (abs(test_series).sum())

    return RMSE, MAE, WAPE

def score_forecasts(bt, forecast_cols, by=None):

    """
    Returns a DataFrame with one row per forecast column (and per group when by is given),
    with columns Model, RMSE, MAE and WAPE. RMSE and MAE are in sales units and WAPE is a
    percentage. Scores are pooled within each group, not averaged across stores.
    Raises a ValueError if Sales or any forecast column has missing values. 
    Rows are never dropped, so every model is scored on exactly the same rows.
    
    """

    for col in ['Sales'] + forecast_cols:
        if bt[col].isnull().any():
            raise ValueError(f"Column {col} has missing values. Cannot score forecasts with missing values.")

    if by is None:
        groups = [(None, bt)]
    else:
        groups = list(bt.groupby(by))

    scores = []

    for model in forecast_cols:
        for key, rows in groups:
            RMSE, MAE, WAPE = evaluate_forecast(rows['Sales'], rows[model])

            row = {'Model': model, 'RMSE': RMSE, 'MAE': MAE, 'WAPE': WAPE}

            if by is not None:
                row[by] = key

            scores.append(row)
            
    return pd.DataFrame(scores)
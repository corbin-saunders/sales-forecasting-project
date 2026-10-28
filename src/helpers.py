import sklearn.metrics

def evaluate_forecast(test_series, forecast):
    RMSE = sklearn.metrics.root_mean_squared_error(test_series, forecast)
    MAE = sklearn.metrics.mean_absolute_error(test_series, forecast)
    WAPE = 100 * (abs(test_series - forecast).sum()) / (abs(test_series).sum())
    
    print(f"RMSE: {RMSE:.2f}")
    print(f"MAE: {MAE:.2f}")
    print(f"WAPE: {WAPE:.2f}%")

    return RMSE, MAE, WAPE
import pandas as pd

def add_window_avg_baseline(train, test, cutoff, window_days, column_name):

    """
    Returns a copy of test with a new column, column_name. 
    Each row gets the mean of the sales from the same store and weekday over the last window_days days immediately before the cutoff.
    A window_days of None means to use all available data before the cutoff.

    """

    cutoff = pd.Timestamp(cutoff)

    if window_days is None:
        train_cutoff = train
    elif window_days % 7 == 0 and window_days > 0:
        train_cutoff = train[train['Date'] >= cutoff-pd.Timedelta(days=window_days)]
    else:
        raise ValueError("window_days must be None or a positive multiple of 7.")

    sales_forecast = train_cutoff[['Store', 'DayOfWeek', 'Sales']].copy()

    avg_forecast = sales_forecast.groupby(['Store', 'DayOfWeek'])['Sales'].mean().reset_index()

    baseline_forecast = avg_forecast.rename(columns={'Sales': column_name})

    test_with_baseline = test.merge(baseline_forecast, on=['Store', 'DayOfWeek'], how='left', validate='many_to_one')

    return test_with_baseline

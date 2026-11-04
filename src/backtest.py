import pandas as pd

# Choosing 2015-02-02 as the first cutoff date so that 28-day
# lookbacks avoid the gaps that end on 2015-01-01.

FIRST_CUTOFF = '2015-02-02'

def get_cutoffs(data_end, horizon_days, first_cutoff = FIRST_CUTOFF, step_days = 7):
    
    """
    Returns a DatetimeIndex of cutoff dates for backtesting, starting from the first_cutoff date and going up to the last cutoff, derived from data_end. 
    Each cutoff date is a pandas Timestamp.
    
    """
    
    last_cutoff = pd.Timestamp(data_end) - pd.Timedelta(days=horizon_days-1)

    cutoffs = pd.date_range(start=first_cutoff, end=last_cutoff, freq=pd.DateOffset(days=step_days))

    if len(cutoffs) == 0:
        raise ValueError(f"No cutoffs generated. Check that first_cutoff ({first_cutoff}) is before last_cutoff ({last_cutoff}).")

    return cutoffs

def split_data_by_cutoff(df, cutoff, horizon_days):

    """
    Returns (train, test). 
    Train is every row whose Date is before the cutoff. 
    Test is every row whose Date is on or after the cutoff and before cutoff + horizon_days, so the cutoff date is the first day of the test.
    
    """

    cutoff = pd.Timestamp(cutoff)
    test_end = cutoff + pd.Timedelta(days=horizon_days)

    train = df[df['Date'] < cutoff].copy()
    test = df[(df['Date'] >= cutoff) & (df['Date'] < test_end)].copy()

    return train, test

def run_backtest(df, cutoffs, horizon_days, forecast_fn):
    """
    For each cutoff, splits df into train and test, calls forecast_fn(train, test, cutoff), and tags the result with the cutoff date and day_ahead (1 for the cutoff day, up to horizon_days). 
    Returns all results stacked into one DataFrame.
    
    """
    results = []

    for cutoff in cutoffs:
        train, test = split_data_by_cutoff(df, cutoff, horizon_days)

        forecast = forecast_fn(train, test, cutoff)

        forecast['cutoff'] = cutoff
        forecast['day_ahead'] = (forecast['Date'] - cutoff).dt.days + 1

        results.append(forecast)

    return pd.concat(results, ignore_index=True)
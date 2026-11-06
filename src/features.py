import pandas as pd

def add_sales_lag(df, days):

    """
    Returns a copy of df with a new column, 'sales_lag_{days}', which is the sales from {days} days ago for the same store. 
    Lagged sales are NaN when the store has no row exactly {days} days earlier (the start of the data, or after a gap).

    """

    if days % 7 != 0 or days <= 0:
        raise ValueError("days must be a positive multiple of 7.")
    
    copied_df = df[['Store', 'Date', 'Sales']].copy()

    copied_df['Date'] = copied_df['Date'] + pd.Timedelta(days=days)

    lagged_df = copied_df.rename(columns={'Sales': f'sales_lag_{days}'})

    df_with_lag = df.merge(lagged_df, on=['Store', 'Date'], how='left', validate='one_to_one') 

    return df_with_lag

def add_lag_mean(df, weeks):

    """
    Returns a copy of df with a new column, 'sales_lag_mean_{weeks}w', which is the mean of the sales from the same store and weekday over the last {weeks} weeks.
    Lagged mean sales are NaN when the store has no rows exactly {weeks} weeks earlier (the start of the data, or after a gap), or when the previous values are not available.
    
    """

    if weeks <= 0 or weeks != int(weeks):
        raise ValueError("weeks must be a positive integer.")

    temp_df = df[['Store', 'Date', 'Sales']].copy()

    for week in range(1, weeks + 1):
        temp_df = add_sales_lag(temp_df, week * 7)

    lag_avg = temp_df[[f'sales_lag_{week * 7}' for week in range(1, weeks + 1)]].mean(axis=1, skipna=False)

    temp_df[f'sales_lag_mean_{weeks}w'] = lag_avg

    df_with_lag_mean = df.merge(temp_df[['Store', 'Date', f'sales_lag_mean_{weeks}w']], on=['Store', 'Date'], how='left', validate='one_to_one')

    return df_with_lag_mean

def add_calendar_features(df):

    """
    Returns a copy of df with new columns for calendar features.

    """
    temp_df = df.copy()
    temp_df['Month'] = temp_df['Date'].dt.month
    temp_df['DayOfMonth'] = temp_df['Date'].dt.day
    temp_df['WeekOfYear'] = temp_df['Date'].dt.isocalendar().week.astype(int)
    temp_df['DayOfYear'] = temp_df['Date'].dt.dayofyear

    return temp_df

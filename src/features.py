import pandas as pd
import numpy as np

def add_sales_lag(df, days):

    """
    Returns a copy of df with a new column, 'sales_lag_{days}', which is the sales from {days} days ago for the same store. 
    Lagged sales are NaN when the store has no row exactly {days} days earlier (the start of the data, or after a gap).

    """

    if days % 7 != 0 or days <= 0:
        raise ValueError("days must be a positive multiple of 7.")
    
    copied_df = df[['Store', 'Date', 'Sales']].copy()

    copied_df['Date'] = copied_df['Date'] + pd.Timedelta(days=days)

    lagged_df = copied_df.rename(columns={'Sales': f'SalesLag{days}'})

    df_with_lag = df.merge(lagged_df, on=['Store', 'Date'], how='left', validate='one_to_one') 

    return df_with_lag

def add_lag_mean(df, weeks):

    """
    Returns a copy of df with a new column, 'SalesLagMean{weeks}w', which is the mean of the sales from the same store and weekday over the last {weeks} weeks.
    Lagged mean sales are NaN when the store has no rows exactly {weeks} weeks earlier (the start of the data, or after a gap), or when the previous values are not available.
    
    """

    if weeks <= 0 or weeks != int(weeks):
        raise ValueError("weeks must be a positive integer.")

    temp_df = df[['Store', 'Date', 'Sales']].copy()

    for week in range(1, weeks + 1):
        temp_df = add_sales_lag(temp_df, week * 7)

    lag_avg = temp_df[[f'SalesLag{week * 7}' for week in range(1, weeks + 1)]].mean(axis=1, skipna=False)

    temp_df[f'SalesLagMean{weeks}w'] = lag_avg

    df_with_lag_mean = df.merge(temp_df[['Store', 'Date', f'SalesLagMean{weeks}w']], on=['Store', 'Date'], how='left', validate='one_to_one')

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

def add_competition_months(df):

    """
    Returns a copy of df with a new column, 'CompetitionMonths', which is the number of months since the competition opened for each store.
    When the competition has not opened yet, CompetitionMonths is 0.
    When the opening date is missing, CompetitionMonths is NaN.
    Drops the columns 'CompetitionOpenSinceYear' and 'CompetitionOpenSinceMonth'.

    """

    temp_df = df.copy()
    temp_df['CompetitionMonths'] = ((temp_df['Date'].dt.year - temp_df['CompetitionOpenSinceYear']) * 12 + (temp_df['Date'].dt.month - temp_df['CompetitionOpenSinceMonth'])).clip(lower=0)

    temp_df = temp_df.drop(columns=['CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth'])

    return temp_df

def add_state_holiday_features(df):

    """
    Returns a copy of df with the 'StateHoliday' column replaced by two columns:
    - 'StateHolidayType': the holiday type as a string ('0', 'a', 'b' or 'c')
    - 'IsStateHoliday': 1 if there is a state holiday, otherwise 0 (integer)

    """

    temp_df = df.copy()

    temp_df['StateHolidayType'] = temp_df['StateHoliday'].astype(str)

    temp_df['IsStateHoliday'] = (temp_df['StateHolidayType'] != '0').astype(int)

    temp_df = temp_df.drop(columns=['StateHoliday'])

    return temp_df

def build_features(df):

    """
    Returns a copy of df with all feature engineering functions applied.

    """

    lagged_7_df = add_sales_lag(df, 7)
    lagged_14_df = add_sales_lag(lagged_7_df, 14)
    lagged_28_df = add_sales_lag(lagged_14_df, 28)

    mean_4_df = add_lag_mean(lagged_28_df, 4)
    mean_8_df = add_lag_mean(mean_4_df, 8)

    calendar_df = add_calendar_features(mean_8_df)

    competition_df = add_competition_months(calendar_df)

    state_holiday_df = add_state_holiday_features(competition_df)

    category_cols = ['StoreType', 'Assortment', 'StateHolidayType']

    state_holiday_df[category_cols] = state_holiday_df[category_cols].astype('category')

    return state_holiday_df

FEATURE_COLUMNS = ['DayOfWeek', 'Promo', 'SchoolHoliday', 'IsStateHoliday', 'StateHolidayType',
                   'Month', 'DayOfMonth', 'WeekOfYear', 'DayOfYear', 'StoreType', 'Assortment',
                   'CompetitionDistance', 'CompetitionMonths', 'Promo2', 'SalesLag7', 'SalesLag14',
                   'SalesLag28', 'SalesLagMean4w', 'SalesLagMean8w'
                  ]

def check_no_leakage(df, featured_df, cutoff, horizon_days=7):
    cutoff = pd.Timestamp(cutoff)
    test_end = cutoff + pd.Timedelta(days=horizon_days)

    def test_week(features):
        in_week = (features['Date'] >= cutoff) & (features['Date'] < test_end)
        return (features.loc[in_week, ['Store', 'Date'] + FEATURE_COLUMNS]
                        .sort_values(['Store', 'Date'])
                        .reset_index(drop=True))

    expected = test_week(featured_df)

    # real test: remove all sales from the cutoff onwards, features should not change
    future_removed = df.copy()
    future_removed.loc[future_removed['Date'] >= cutoff, 'Sales'] = np.nan
    pd.testing.assert_frame_equal(expected, test_week(build_features(future_removed)))

    # control: remove the week BEFORE the cutoff, features should change
    past_removed = df.copy()
    week_before = (past_removed['Date'] >= cutoff - pd.Timedelta(days=7)) & (past_removed['Date'] < cutoff)
    past_removed.loc[week_before, 'Sales'] = np.nan
    assert not expected.equals(test_week(build_features(past_removed))), \
        "control failed: removing the week before the cutoff changed nothing, so the test proves nothing"

    print(f"{cutoff.date()}: no leakage, control passed")


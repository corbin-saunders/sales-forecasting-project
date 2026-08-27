import pandas as pd
from pathlib import Path

def load_and_merge_data():
    # TODO: fix relative path if this breaks when imported elsewhere
    train_data = pd.read_csv('../data/train.csv')
    store_data = pd.read_csv('../data/store.csv')

    df = train_data.merge(store_data, on='Store', how='left')

    return df
    
def fill_null_features(df):

    ## filling null CompetitionDistances with average distance
    ## filling null Promo2Since* with 0
    ## filling null PromoInterval with empty string
    
    df['CompetitionDistance'] = df['CompetitionDistance'].fillna(df['CompetitionDistance'].median())

    df['Promo2SinceWeek'] = df['Promo2SinceWeek'].fillna(0)
    df['Promo2SinceYear'] = df['Promo2SinceYear'].fillna(0)

    df['PromoInterval']   = df['PromoInterval'].fillna("")

    return df

def load_clean_data():
    df = load_and_merge_data()
    df = fill_null_features(df)

    df['Date'] = pd.to_datetime(df['Date'])

    return df
import pandas as pd

# Choosing 2015-02-02 as the first cutoff date so that 28-day
# lookbacks avoid the gaps that end on 2015-01-01.

FIRST_CUTOFF = '2015-02-02'

def get_cutoffs(data_end, horizon_days, first_cutoff = FIRST_CUTOFF, step_days = 7):
    
    """Returns a DatetimeIndex of cutoff dates for backtesting, starting from the first_cutoff date and going up to the last cutoff, derived from data_end. Each cutoff date is a pandas Timestamp."""
    
    last_cutoff = pd.Timestamp(data_end) - pd.Timedelta(days=horizon_days-1)

    cutoffs = pd.date_range(start=first_cutoff, end=last_cutoff, freq=pd.DateOffset(days=step_days))

    if len(cutoffs) == 0:
        raise ValueError(f"No cutoffs generated. Check that first_cutoff ({first_cutoff}) is before last_cutoff ({last_cutoff}).")

    return cutoffs
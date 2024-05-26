import pandas as pd


def resampling(df):
    df.datetime = pd.to_datetime(df.datetime)
    df.set_index(df.datetime, inplace = True)
    df.drop(columns=["time","fsym","tsym"])

        # Resample to 5 minutes
    df_5min = df.resample('5T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volumefrom': 'sum',
        'volumeto' : 'sum'
    })


    # Resample to 15 minutes
    df_15min = df.resample('15T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volumefrom': 'sum',
        'volumeto' : 'sum'
    })


    # Resample to 1 hour
    df_1h = df.resample('1H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volumefrom': 'sum',
        'volumeto' : 'sum'
    })



    # Resample to 6 hours
    df_6h = df.resample('6H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volumefrom': 'sum',
        'volumeto' : 'sum'
    })


    # Resample to 6 hours
    df_1d = df.resample('1D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volumefrom': 'sum',
        'volumeto' : 'sum'
    })

    # Now df_5min, df_15min, df_20min, df_1h, df_3h, df_6h are your aggregated DataFrames
    df_list = [df_5min, df_15min, df_1h, df_6h, df_1d]

    return df_list
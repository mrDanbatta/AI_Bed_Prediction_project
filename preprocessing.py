import pandas as pd
import numpy as np
import math

def preprocess_ward_data(df):

    df = df.copy()

    # Convert date columns to datetime
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime')
    df = df.set_index('datetime')

    # resampling ward data to daily frequency using mean for occupied beds
    ts = df['occupied_beds'].resample('D').mean()

    # Handling missing values by forward filling and backward filling
    ts = ts.ffill().bfill()

    return ts
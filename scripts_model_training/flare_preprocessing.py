import numpy as np
import pandas as pd
from utilities import *


# a function to preprocess the operation data
def opr_data_preprocessing(opr_path, rescale=False):
    # Load the data
    opr = pd.read_csv(opr_path)
    opr['peak_time'] = pd.to_datetime(opr['peak_time'])
    opr = opr[opr['peak_time'] >= '2010-05-01']
    opr['log_intensity'] = opr['label'].apply(get_log10_intensity)
    if rescale:
        # after 2020-01-01, the GOES X-ray flux is rescaled
        sub = opr[opr['peak_time'] >= '2020-01-01']
        sub['log_intensity'] = sub['label'].apply(rescale)
        opr = pd.concat([opr[opr['peak_time'] < '2020-01-01'], sub])

    opr = opr[opr['log_intensity'] >= -7.0] # only keep the B/C/M/X class flares
    opr.index = range(len(opr))

    print('Shape of the operation data:', opr.shape)
    return opr


# a function to preprocess the science-quality data
def sci_data_preprocessing(sci_path, opr):
    # Load the data
    sci = pd.read_csv(sci_path)
    sci['peak_time'] = pd.to_datetime(sci['peak_time'])
    sci = sci[sci['peak_time'] > '2010-05-01']
    sci = sci.sort_values(by='peak_time')
    #sci['log_intensity'] = sci['fl_cls'].apply(get_log10_intensity)
    sci.rename(columns={'logintensity': 'log_intensity'}, inplace=True)
    sci.index = range(sci.shape[0])

    march2011 = opr[(opr['peak_time'] >= '2011-03-01') & (opr['peak_time'] < '2011-04-01')]
    march2011 = march2011.drop(columns=['obs', 'cls', 'unique_id'])
    march2011.rename(columns={ 'noaa_ar': 'assigned_ar'}, inplace=True)
    march2011['log_intensity'] = march2011['label'].apply(rescale)
    # fill the missing columns with np.nan
    for col in sci.columns:
        if col not in march2011.columns:
            march2011[col] = np.nan
    
    sci = pd.concat([sci, march2011])
    sci = sci.sort_values(by='peak_time')
    #sci = sci[sci['log_intensity'] >= -7.0] # only keep the B/C/M/X class flares
    sci.rename(columns={'assigned_ar': 'noaa_ar'}, inplace=True) #202547 update
    #sci.drop(columns=['noaa_ar_5s'], inplace=True)
    sci.index = range(sci.shape[0])

    print('Shape of the science-quality data:', sci.shape)
    return sci
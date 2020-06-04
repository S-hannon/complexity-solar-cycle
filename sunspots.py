from __future__ import division
import numpy as np
import pandas as pd
import datetime as dt

filepath = 'N:\Documents\Projects\CME_Complexity_Ranking\Data\sunspot_record\\SN_d_tot_V2.0.csv'


def get_sunspot_record(filepath):
    """Reads in daily total sunspot number and returns as pd df
    """
    # year, month, day, decimal year, SNvalue , SNerror, Nb observations
    # 1818;01;01;1818.001;  -1; -1.0;   0;1
    
    dates = []
    dates_dec = []
    sunspot_n = []
    sunspot_n_err = []
    n_obs = []
    
    # Loop over file lines and separate
    with open(filepath) as f:
        lines = f.readlines();
        # Remove unhelpful whitespace and split on semi-colons
        for l in lines:
            elements = l[:-1].replace(' ','').split(';')
            # Append date as datetime object
            dates.append(dt.datetime(int(elements[0]), int(elements[1]), int(elements[2])))
            # Append rest of data
            dates_dec.append(elements[3])
            sunspot_n.append(elements[4])
            sunspot_n_err.append(elements[5])
            n_obs.append(elements[6])

    # Convert to pd df
    df = pd.DataFrame({'date' : dates, 'dec_date' : dates_dec,
                       'daily_total_sunspot_n' : sunspot_n,
                       'error' : sunspot_n_err, 'n_obs' : n_obs})
    
    # Value of -1 in sunspot number indicates missing data, replace with np.NaN
    df['daily_total_sunspot_n'] = df['daily_total_sunspot_n'].where(df['daily_total_sunspot_n']!='-1', np.NaN) 
    
    return df

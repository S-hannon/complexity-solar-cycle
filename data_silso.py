from __future__ import division
import os
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt


class SILSO:
    def __init__(self, data_loc):
        self.data_loc = data_loc
    def load(self):
        """Reads in daily total sunspot number and returns as pd df
        SILSO, World Data Center - Sunspot Number and Long-term Solar Observations,
        Royal Observatory of Belgium, on-line Sunspot Number catalogue:
            http://www.sidc.be/SILSO/, ‘2008-2017’.
        """
        # Get data location from config.txt
        filepath = os.path.join(self.data_loc, 'SILSO', 'SN_d_tot_V2.0.csv')
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
                dates.append(dt.datetime(int(elements[0]),
                                         int(elements[1]),
                                         int(elements[2])))
                # Append rest of data
                dates_dec.append(elements[3])
                sunspot_n.append(elements[4])
                sunspot_n_err.append(elements[5])
                n_obs.append(elements[6])
        # Convert to pd df
        df = pd.DataFrame({'date' : dates,
                           'dec_date' : dates_dec,
                           'daily_total_sunspot_n' : sunspot_n,
                           'error' : sunspot_n_err, 
                           'n_obs' : n_obs})
        df['date'] = pd.to_datetime(df['date'])
        df['daily_total_sunspot_n'] = df['daily_total_sunspot_n'].astype(int)
        # Value of -1 in sunspot number indicates missing data, replace with np.NaN
        df['daily_total_sunspot_n'] = df['daily_total_sunspot_n'].replace(to_replace=-1,
                                                                          value=np.NaN)
        return df
    def plot(self):
        sunspots = self.load()
        plt.figure(figsize=(10,6))
        plt.plot(sunspots.date, sunspots.daily_total_sunspot_n)
        plt.xlabel("Time", fontsize=16)
        plt.ylabel("Daily Total Sunspot Number", fontsize=16)
        plt.xlim((dt.datetime(1986, 1, 1), dt.datetime(2018, 1, 1)))
        plt.ylim((0, 450))

"""
This contains functions which take a pandas DataFrame, which must have a column
of HELCATS CME names called 'helcats_name', add data from various sources
corresponding to those CMEs, and return the df with extra data.
"""

from __future__ import division
import os
import sys
import numpy as np
import pandas as pd
import astropy.units as u
from datetime import datetime, timedelta
import popfss_image_processing as ip

#sys.path.insert(0, r'N:\\Documents\\Code\\useful_code')
sys.path.insert(0, r'C:\\Users\\shann\\OneDrive\\Documents\\Research\\Workspace\\Code\\useful_code')
import misc
from data_helcats import HELCATS
from data_heliomas import HelioMAS
from data_corset import CORSET
from data_seeds import SEEDS
from data_cactus import CACTus
from data_yutian_insitu import Yutian_insitu


data_loc = r'C:\\Users\\shann\\OneDrive\\Documents\\Research\\Workspace\\Data'

# Functions to add new columns to the df
def add_col_to_df(df, col_1, col_2, operation, name, abs_col=False):
    """Adds an extra column to df: col_1 operation col_2
    e.g. operation = "multiply" new col = col_1 * col_2
    :param: df: pandas data frame
    :param: col_1: string, first column to use
    :param: col_2: string, second column to use
    :param: operation: string, 'multiply', 'divide', 'add', 'subtract'
    :param: name: string, name of new column to add to df
    :param: abs_col: bool, take absolute value of calculated values
    """
    new_vals = []
    for i in range(len(df)):
        try:
            if operation == 'add':
                result = df[col_1][i] + df[col_2][i]
            elif operation == 'subtract':
                result = df[col_1][i] - df[col_2][i]
            elif operation == 'multiply':
                result = df[col_1][i] * df[col_2][i]   
            elif operation == 'divide':
                result = df[col_1][i] / df[col_2][i]
            if abs_col == True:
                new_vals.append(abs(result))
            elif abs_col == False:
                new_vals.append(result)
        except:
            new_vals.append(np.NaN)
    # Add new values to df
    df[name] = pd.Series(new_vals, index=df.index)
    return df


def add_helcats_to_df(df, col, target=None, name=None, astype=None):
    """Finds property h_name from a HELCATS catalogue for CMEs in df.
    :param: df: pandas dataframe
    :param: h_name: string, helcats name of property to add
    :param: cat: string, helcats catalogue to find property h_name from
        options: 'hicat', 'higeocat', 'hijoincat', 'kincat', 'linkcat'
    :param: name: string, label for new column in dataframe
    :param: date: bool True or False, if True format values to datetime objects
    """
    hc = HELCATS(data_loc)
    col_data = hc.get_col_data(col, list(df.helcats_name.values), target=target)   
    if name == None:
        name = col
    df[name] = pd.Series(col_data, index=df.index)
    if astype != None:
        if astype == datetime:
            df[name] = pd.to_datetime(df[name])
        else:
            df[name] = df[name].astype(astype)
    return df


def add_craft_and_time_to_df(df):
    """Adds columns; craft = 'sta' or 'stb'; and time = datetime - appearance
    of CME in HI1.
    """
    hc = HELCATS(data_loc)
    craft_list, time_list = hc.get_cme_details_list(df.helcats_name)
    df['craft'] = pd.Series(craft_list, index=df.index)
    df['time'] = pd.Series(time_list, index=df.index)
    return df


def add_geo_indicator_to_df(df):
    """Adds a column 'geo' which is 1 if the CME was geoeffective, and 0
    otherwise.
    """
    hc = HELCATS(data_loc)
    ind_data = hc.get_geo_indicator(df.helcats_name.values)
    df['geo'] = pd.Series(ind_data, index=df.index)
    return df


def add_matches_to_df(df):
    """Adds a column 'match' containing the helcats name of the same CME
    observed by the other spacecraft, if it was.
    """
    hc = HELCATS(data_loc)
    match_data = hc.get_matches(df.helcats_name.values)    
    df['match'] = pd.Series(match_data, index=df.index)
    return df


def add_width_to_df(df):
    """Adds an extra column "width" to df which is the angular width of the CME
    in degrees.
    """
    df = add_helcats_to_df(df, 'PA-N [deg]')
    df = add_helcats_to_df(df, 'PA-S [deg]')
    df = add_col_to_df(df, 'PA-N [deg]', 'PA-S [deg]', 'subtract', 'width', abs_col=True)
    return df


def add_te_track_times_to_df(df):
    """
    Adds 'start_time', 'mid_time' and 'end_time' of the CME in HI FOV.
    These values are found from the HELCATS time-elongation profiles.
    """
    helcats = HELCATS(data_loc)
    start, mid, end, mid_el = helcats.get_te_track_times_list(df.helcats_name)
    df['start_time'] = pd.Series(start, index=df.index)
    df['mid_time'] = pd.Series(mid, index=df.index)
    df['end_time'] = pd.Series(end, index=df.index)
    df['mid_el'] = pd.Series(mid_el, index=df.index)
    return df


def add_corset_to_df(df):
    corset = CORSET(misc.get_project_dirs()['data'])
    corset = corset.load()
    helcats = HELCATS()
    window = [timedelta(hours=48), timedelta(hours=-12)]
    helcats_names = helcats.match_to_helcats(corset.time, corset.craft,
                                             window=window)
    corset['helcats_name'] = pd.Series(helcats_names, index=corset.index)
    df = pd.merge(df, corset, on=['helcats_name'])
    return df


def add_seeds_to_df(df):
    seeds = SEEDS(data_loc)
    seeds = seeds.load()
    helcats = HELCATS(data_loc)
    window = [timedelta(hours=48), timedelta(hours=-12)]
    helcats_names = helcats.match_to_helcats(seeds.time, seeds.craft,
                                             window=window)
    seeds['helcats_name'] = pd.Series(helcats_names, index=seeds.index)
    df = pd.merge(df, seeds, on=['helcats_name'])
    return df


def add_cactus_to_df(df):
    cactus = CACTus(data_loc)
    cactus = cactus.load()
    helcats = HELCATS(data_loc)
    window = [timedelta(hours=48), timedelta(hours=-12)]
    helcats_names = helcats.match_to_helcats(cactus.time, cactus.craft,
                                             window=window)
    cactus['helcats_name'] = pd.Series(helcats_names, index=cactus.index)
    df = pd.merge(df, cactus, on=['helcats_name'])
    return df


def add_yutian_insitu_to_df(df):
    y = Yutian_insitu(data_loc)
    ydf = y.load()
    helcats = HELCATS(data_loc)
    window = [timedelta(hours=12), timedelta(hours=12)]
    helcats_names = helcats.match_to_helcats(ydf.start_time,
                                             insitu_craft='Wind',
                                             window=window)
    ydf['helcats_name'] = pd.Series(helcats_names, index=ydf.index)
    df = pd.merge(df, ydf, on=['helcats_name'])
    return df


def add_spacecraft_separation_to_df(df):
    """Adds column "sc_sep" which is distance of spacecraft from CME in km.
    """
    df = add_helcats_to_df(df, 'SSE Phi [deg]', 'higeocat', 'SSEphi')
    scs = []
    for i in df.index:
        try:
            distance = 695700 * 1000
            sin_phi = np.sin(df.SSEphi[i])
            el = 14
            scs_i = (distance * sin_phi) / np.sin(180-el-df.SSEphi[i])
            scs.append(scs_i/1000)
        except:
            scs.append(np.NaN)
    # Add spacecraft separation from CME to df
    df['sc_sep'] = pd.Series(scs, index=df.index)
    return df


def add_image_stats_to_df(df, tag=""):
    """reads in image stats from .csv file created and adds a new column for 
    each image stat.
    """
    df_stats = ip.load_image_stats()
    # sort both data frames and make sure the are the same length
    if len(df) != len(df_stats):
        raise ValueError("Data Frames are not the same length")
    df = df.sort_values(by='complexity')
    df_stats = df_stats.sort_values(by='complexity')
    # remove duplicate columns
    df_stats = df_stats.drop(['Unnamed: 0', 'complexity', 'craft',
                              'helcats_name', 'ranking_place', 'ssw_name',
                              'time'], axis=1)
    merged_df = pd.concat([df, df_stats], axis=1)
    return merged_df


def add_heliomas_to_df(df):
    df = add_helcats_to_df(df, 'PA-fit [deg]', 'higeocat', name='CPA')
    df = add_helcats_to_df(df, 'PA-N [deg]', 'hicat', name='PA-N')
    df = add_helcats_to_df(df, 'PA-S [deg]', 'hicat', name='PA-S')
    lon_mean = []
    lon_std = []
    lat_mean = []
    lat_std = []
    r_mean = []
    r_std = []
    sw = []
    lat_diff = []
    lon_diff = []
    for i in df.index:
        mas = HelioMAS(df.time[i])
        if df.craft[i] == 'stb':
            CPA = 360 - df['CPA'][i]
            PAN = 360 - df['PA-S'][i]
            PAS = 360 - df['PA-N'][i]
        else:
            CPA = df['CPA'][i]
            PAN = df['PA-N'][i]
            PAS = df['PA-S'][i]
        try:
            lon_vals = mas.extract('speed', lon=None, lat=CPA*u.deg, r=30*u.R_sun)
            lat_vals = mas.extract('speed', lon=df.phi[i]*u.deg, lat=None, r=30*u.R_sun)
            r_vals = mas.extract('speed', lon=df.phi[i]*u.deg, lat=CPA*u.deg, r=None)
            lat_diff.append(abs(mas.extract('speed', lon=df.phi[i]*u.deg, lat=PAN*u.deg, r=30*u.R_sun) -
                                mas.extract('speed', lon=df.phi[i]*u.deg, lat=PAS*u.deg, r=30*u.R_sun)).value)
            lon_diff.append(abs(mas.extract('speed', lon=df.phi[i]*u.deg, lat=CPA*u.deg, r=15*u.R_sun) -
                                mas.extract('speed', lon=df.phi[i]*u.deg, lat=CPA*u.deg, r=40*u.R_sun)).value)
            lon_mean.append(np.mean(lon_vals.value))
            lat_mean.append(np.mean(lat_vals.value))
            r_mean.append(np.mean(r_vals.value))
            lon_std.append(np.std(lon_vals.value))
            lat_std.append(np.std(lat_vals.value))
            r_std.append(np.std(r_vals.value))
            sw.append(mas.extract(lon=df.phi[i]*u.deg, lat=CPA*u.deg, r=30*u.R_sun))
        except:
            lon_mean.append(np.NaN)
            lat_mean.append(np.NaN)
            r_mean.append(np.NaN)
            lon_std.append(np.NaN)
            lat_std.append(np.NaN)
            r_std.append(np.NaN)
            sw.append(np.NaN)
            lat_diff.append(np.NaN)
            lon_diff.append(np.NaN)
    df['lon_mean'] = pd.Series(lon_mean, index=df.index)
    df['lon_std'] = pd.Series(lon_std, index=df.index)
    df['lat_mean'] = pd.Series(lat_mean, index=df.index)
    df['lat_std'] = pd.Series(lat_std, index=df.index)
    df['r_mean'] = pd.Series(r_mean, index=df.index)
    df['r_std'] = pd.Series(r_std, index=df.index)
    df['sw'] = pd.Series(sw, index=df.index)
    df['lat_diff'] = pd.Series(lat_diff, index=df.index)
    df['lon_diff'] = pd.Series(lon_diff, index=df.index)
    return df

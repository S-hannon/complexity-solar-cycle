from __future__ import division
import os
import sys
# import glob
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import popfss_asset_production as ap

#sys.path.insert(0, r'N:\\Documents\\Code\\useful_code')
sys.path.insert(0, r'C:\\Users\\shann\\OneDrive\\Documents\\Research\\Workspace\\Code\\useful_code')
from misc import get_project_dirs
from helcats_tools import load_helcats_cat, find_cme_match
from corset import load_corset

# def get_project_dirs():
#     """
#     A function to load in dictionary of project directories stored in a config.txt file stored in the
#     :return proj_dirs: Dictionary of project directories, with keys 'data','figures','code', and 'results'.
#     """
#     files = glob.glob(r'C:\Users\shann\OneDrive\Documents\Code\protect_our_planet_from_solar_storms\project_directories.txt')
#     # Open file and extract
#     with open(files[0], "r") as f:
#         lines = f.read().splitlines()
#         proj_dirs = {l.split(',')[0]: l.split(',')[1] for l in lines}
#     # Just check the directories exist.
#     for val in iter(proj_dirs.items()):
#         if not os.path.exists(val[1]):
#             print('Error, invalid path, check config: ' + val[1])
#     return proj_dirs


def get_df_index(df, helcats_name):
    """Takes the dataframe and finds the index of the required cme.
    :param: helcats_name: str, helcats id of cme e.g. HCME_B__20130830_01
    """
    if len(df[df.helcats_name == helcats_name]) == 1:
        return df[df.helcats_name == helcats_name].index.values[0]
    

def get_cme_details(helcats_name):
    """Returns the craft and date of helcats_name.
    :param: helcats_name: str, helcats id of cme e.g. HCME_B__20130830_01
    """
    parts = helcats_name.split('_')
    if parts[1] == 'A':
        craft = 'sta'
    elif parts[1] == 'B':
        craft = 'stb'
    date = pd.datetime.strptime(parts[3], '%Y%m%d')
    return craft, date


def get_helcats_name(image_name):
    """returns HELCATS name string given image name
    e.g.ssw_067_helcats_HCME_B__20131128_02_stb_diff_20131129_005001.jpg
    returns HCME_B__20131128_02
    """
    parts = image_name.split('_')
    return parts[3] + '_' + parts[4] + '__' + parts[6] + '_' + parts[7]
    
    
###############################################################################
# Get complexity values from .csv file, and make the base df of 1100 CMEs
def get_model_fit_from_r(tag=""):
    # Read in the model parameters
    project_dirs = get_project_dirs()
    params = pd.read_csv(os.path.join(project_dirs['data'], 'POPFSS',
                                      'popfss_model_fit_r' + tag + '.csv'))
    return params


def extract_data(params):
    # Get the data
    craft = []
    time = []
    ssw_name = []
    helcats_name = []
    complexity = []
    rank = []
    # Order storms by complexity
    params = params.sort_values(by='x')
    # Loop over the storms
    for i in range(len(params)):
        name = params['Unnamed: 0'].values[i]
        ssw_name.append(name[2:(len(name)-57)])
        helcats_name.append(name[(len(name)-48):(len(name)-29)])
        craft.append(name[(len(name)-28):(len(name)-25)])
        time.append(datetime.strptime(name[(len(name)-19):(len(name)-4)], '%Y%m%d_%H%M%S'))
        complexity.append(params['x'].values[i])
        rank.append(i)
    df = pd.DataFrame({'craft' : craft, 'time' : time,
                       'ssw_name' : ssw_name, 'helcats_name' : helcats_name,
                       'complexity' : complexity,
                       'rank' : rank})
    return df


###############################################################################
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


def add_corset_to_df(df):
    corset = load_corset()
    df = pd.merge(df, corset, on=['helcats_name', 'craft'])
    return df


def add_helcats_to_df(df, col, cat, name=False, date=False):
    """Finds property h_name from a HELCATS catalogue for CMEs in df.
    :param: df: pandas dataframe
    :param: h_name: string, helcats name of property to add
    :param: cat: string, helcats catalogue to find property h_name from
        options: 'hicat', 'higeocat', 'hijoincat', 'kincat', 'linkcat'
    :param: name: string, label for new column in dataframe
    :param: date: bool True or False, if True format values to datetime objects
    """
    cat = load_helcats_cat(cat)
    vals = []
    # Loop over items in df
    for i in range(len(df)):
        # Find matching event in helcats cat
        data = cat[cat['helcats_name'] == df['helcats_name'].values[i]]
        # Save the value if it exists, or np.NaN if it doesn't
        p = data[col].values
        if len(p) == 1:
            vals.append(p[0])
        else:
            vals.append(np.NaN)
    # Add the requested paramater to the original dataframe
    if name != False:
        new_name = name
    else:
        new_name = col
    df[new_name] = pd.Series(vals, index=df.index)
    if date == True:
        df[new_name] = pd.to_datetime(df[new_name])
    return df


def add_geo_indicator_to_df(df):
    """Adds a column 'geo' which is 1 if the CME was geoeffective, and 0
    otherwise.
    """
    # Get data
    linkcat = load_helcats_cat('linkcat')
    # Is the event geoeffective? Add 'geo' indicator to df
    ind = []
    for i in range(len(df)):
        data = linkcat[linkcat['HICAT_ID'] == df['helcats_name'].values[i]]
        bz = data['MO_BZMEAN'].values
        if len(bz) == 1:
            if data['TARGET_NAME'].values[0] == 'EARTH_L1':
                ind.append(len(bz))
            else:
                ind.append(0)
        else:
            ind.append(len(bz))
    df['geo'] = pd.Series(ind, index=df.index)
    return df


def add_matches_to_df(df):
    """Adds a column 'match' containing the helcats name of the same CME
    observed by the other spacecraft, if it was.
    """
    # Get data
    hijoincat = load_helcats_cat('hijoincat')
    match = []
    for i in range(len(df)):
        if df.craft.values[i] == 'sta':
            # Find this CME in hijoincat
            match_name = find_cme_match(df.helcats_name.values[i])
            if match_name == None:
                match.append(np.NaN)
            else:
                match.append(match_name)
        if df.craft.values[i] == 'stb':
            hijc = hijoincat[hijoincat.stb == df.helcats_name.values[i]]
            if len(hijc) > 0:
                hijc_index = hijc.index[0]
                match.append(hijoincat.sta[hijc_index])
            else:
                match.append(np.NaN)
    df['match'] = pd.Series(match, index=df.index)
    return df


def add_width_to_df(df):
    """Adds an extra column "width" to df which is the angular width of the CME
    in degrees.
    """
    df = add_helcats_to_df(df, 'PA-N [deg]', 'hicat')
    df = add_helcats_to_df(df, 'PA-S [deg]', 'hicat')
    df = add_col_to_df(df, 'PA-N [deg]', 'PA-S [deg]', 'subtract', 'width', abs_col=True)
    return df


def add_mid_el_to_df(df):
    """Add column 'mid_el' which is the elongation of the CME when it is
    half-way through the HI FOV.
    Also adds 'start_time', 'mid_time' and 'end_time' of the CME in HI FOV.
    These values are found from the HELCATS time-elongation profiles.
    """
    # Get mid el df
    hc = ap.load_comp_events()
    mid_els = []
    start_times = []
    mid_times = []
    end_times = []
    for i in df.index:
        flag = True
        for j in hc.index:
            if df.helcats_name[i] == hc.event_id[j]:
                mid_els.append(hc.mid_el[j])
                start_times.append(hc.start_time[j])
                mid_times.append(hc.mid_time[j])
                end_times.append(hc.end_time[j])
                flag = False
        if flag:
            mid_els.append(np.NaN)
            start_times.append(np.NaN)
            mid_times.append(np.NaN)
            end_times.append(np.NaN)
    df['mid_el'] = pd.Series(mid_els, index=df.index)
    df['start_time'] = pd.Series(start_times, index=df.index)
    df['mid_time'] = pd.Series(mid_times, index=df.index)
    df['end_time'] = pd.Series(end_times, index=df.index)
    return df


def add_spacecraft_separation_to_df(df):
    """Adds column "sc_sep" which is distance of spacecraft from CME in km.
    """
    df = add_helcats_to_df(df, 'SSE_phi', 'higeocat')
    scs = []
    for i in df.index:
        try:
            distance = 695700 * 1000
            sin_phi = np.sin(df.SSE_phi[i])
            el = 14
            scs_i = (distance * sin_phi) / np.sin(180-el-df.SSE_phi[i])
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
    df_stats = pd.read_csv('N:\Documents\Data\HI_Image_Processing\image_stats' + tag + '.csv')
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


###############################################################################
# Functions to extract data from the df
def get_match_vals(df, col):
    """Finds the STEREO-A and STEREO-B values of the property specified.
    """
    a_vals = []
    b_vals = []
    new_df = df[df.craft == 'sta']
    for i in range(len(new_df)):
        matched_b = new_df.match.values[i]
        matched_b = df[df.helcats_name.values == matched_b]
        # if the matched CME is in the df, append values
        if len(matched_b) > 0:
            a_vals.append(new_df[col].values[i])
            b_vals.append(matched_b[col].values[0])
    return a_vals, b_vals     


def find_means(df, t, start_month=1, err='std'):
    """t is time length in months of data to average.
    month is the month to start the means on (for running means)
    err can be std or sem"""
    # Arrays for outputs
    dates = []
    means = dict()
    errs = dict()
    # work out average annual difference
    diffs = []
    for craft in ['sta', 'stb']:
        dfx = df[df.craft.values == craft]
        year = np.min(df.time).year
        month = start_month
        meansx = []
        errsx = []
        while year <= np.max(df.time).year:
            # Sort start date
            start_date = datetime(year, month, 1)
            # sort end date
            end_year = year
            end_month = month + t
            if end_month > 12:
                end_year = year + 1
                end_month = end_month - 12
            end_date = datetime(end_year, end_month, 1)
            # get the relevant points 
            points = []
            for i in dfx.index:
                if dfx.time[i] >= start_date:
                    if dfx.time[i] < end_date:
                        points.append(dfx.complexity[i])
            dates.append(start_date + timedelta(weeks=26))
            if points != []:
                meansx.append(np.mean(points))
                s = np.std(points)
                if err == 'sem':
                    s = s / np.sqrt(len(points))
                errsx.append(s)
            else:
                meansx.append(np.NaN)
                errsx.append(np.NaN)              
            month = month + t
            if month > 12:
                year = year + 1
                month = month - 12
        means[craft] = meansx
        errs[craft] = errsx
        # diffs.append(np.mean(sta_points) - np.mean(stb_points))
    dates = np.unique(dates)
    df = pd.DataFrame({'dates' : dates, 'sta_means' : means['sta'],
                       'stb_means' : means['stb'], 'sta_s' : errs['sta'],
                       'stb_s' : errs['stb']})
#    print "mean sta complexity %s" %(np.nanmean(dfa.complexity.values))
#    print "mean stb complexity %s" %(np.nanmean(dfb.complexity.values))
#    print "stdev sta complexity %s" %(np.nanstd(dfa.complexity.values))
#    print "stdev stb complexity %s" %(np.nanstd(dfb.complexity.values))
#    print "num sta CMEs %s" %(np.count_nonzero(~np.isnan(dfa.complexity.values)))
#    print "num stb CMEs %s" %(np.count_nonzero(~np.isnan(dfb.complexity.values)))
#    t_stat, p_val = sps.ttest_ind(dfa.complexity.values, dfb.complexity.values, axis=0, equal_var=True, nan_policy='omit')
#    print "test statistic: %s, p-value: %s" %(t_stat, p_val)
    return df


# def find_means(df, t, month=1):
#     """t is time length in months of data to average."""
#     # Get the time complexity data
#     dfa = df[df.craft.values == 'sta']
#     dfb = df[df.craft.values == 'stb']
#     # Arrays for outputs
#     dates = []
#     sta_means = []
#     sta_s = []
#     stb_means = []
#     stb_s = []
#     # work out average annual difference
#     diffs = []

#     # Start at the beginning
#     year = 2008
#     # TODO: change hardcoded values to get min / max from data
#     while year < 2017:
#         # Sort date range
#         if month > 12:
#             year = year + 1
#             month = month - 12
#         start_date = datetime.strptime(str(year)+str(month), '%Y%m')
#         dates.append(start_date + timedelta(weeks=(t*(52/12))/2))
        
#         end_year = year
#         end_month = month + t
#         if end_month > 12:
#             end_year = year + 1
#             end_month = end_month - 12
#         end_date = datetime.strptime(str(end_year)+str(end_month), '%Y%m')

#         sta_points = []
#         for a in dfa.index:
#             if dfa.time[a] >= start_date:
#                 if dfa.time[a] < end_date:
#                     sta_points.append(dfa.complexity[a])
#         sta_means.append(np.mean(sta_points))
#         sta_s.append(np.std(sta_points))
# #        sta_s.append(sps.sem(sta_points))

#         stb_points = []
#         for b in dfb.index:
#             if dfb.time[b] >= start_date:
#                 if dfb.time[b] < end_date:
#                     stb_points.append(dfb.complexity[b])
#         stb_means.append(np.mean(stb_points))
#         stb_s.append(np.std(stb_points))
# #        stb_s.append(sps.sem(stb_points))
#         month = month + t
        
#         diffs.append(np.mean(sta_points) - np.mean(stb_points))

#     df = pd.DataFrame({'dates' : dates, 'sta_means' : sta_means,
#                        'stb_means' : stb_means, 'sta_s' : sta_s,
#                        'stb_s' : stb_s})
    
# #    print "mean sta complexity %s" %(np.nanmean(dfa.complexity.values))
# #    print "mean stb complexity %s" %(np.nanmean(dfb.complexity.values))
# #    print "stdev sta complexity %s" %(np.nanstd(dfa.complexity.values))
# #    print "stdev stb complexity %s" %(np.nanstd(dfb.complexity.values))
# #    print "num sta CMEs %s" %(np.count_nonzero(~np.isnan(dfa.complexity.values)))
# #    print "num stb CMEs %s" %(np.count_nonzero(~np.isnan(dfb.complexity.values)))
# #    t_stat, p_val = sps.ttest_ind(dfa.complexity.values, dfb.complexity.values, axis=0, equal_var=True, nan_policy='omit')
# #    print "test statistic: %s, p-value: %s" %(t_stat, p_val)
#     return df


def find_running_means(df, t):
    """t is time length in months of data to average."""
    dates = []
    sta_means = []
    sta_s = []
    stb_means = []
    stb_s = []
    # need to find means starting with each month of the year
    for i in range(1, 13):
        df2 = find_means(df, t, month=i)
        dates = np.append(dates, df2.dates.values)
        sta_means = np.append(sta_means, df2.sta_means.values)
        stb_means = np.append(stb_means, df2.stb_means.values)
        sta_s = np.append(sta_s, df2.sta_s.values)
        stb_s = np.append(stb_s, df2.stb_s.values)
#    return dates, sta_means, stb_means, sta_sds, stb_sds
    df = pd.DataFrame({'dates' : dates, 'sta_means' : sta_means,
                       'stb_means' : stb_means, 'sta_s' : sta_s,
                       'stb_s' : stb_s})
    return df


###############################################################################
# Code to create df of paired comparisons


def load_paired_comparison_results(df, tag=""):
    """Loads popfss_comparison_results.
    """
    proj_dirs = get_project_dirs()
    pc_df = pd.read_csv(os.path.join(proj_dirs['data'], 'POPFSS',
                                     'popfss_comparison_results' + tag + '.csv'))
    left_c_vals = []
    right_c_vals = []
    left_craft = []
    right_craft = []
    left_date = []
    right_date = []
    left_year = []
    right_year = []
    winner = []
    for i in pc_df.index:
        # determine the overall winner
        votes = pc_df.left_wins[i] + pc_df.right_wins[i]
        if (pc_df.left_wins[i] / votes) > (pc_df.right_wins[i] / votes):
            winner.append('left')
        elif (pc_df.left_wins[i] / votes) < (pc_df.right_wins[i] / votes):
            winner.append('right')
        else:
            winner.append('draw')
        for j in ['left', 'right']:
            img_name = pc_df[j + '_subject'][i]
            helcats_name = get_helcats_name(img_name)
            craft, date = get_cme_details(helcats_name)
            if j == 'left':
                left_craft.append(craft)
                left_date.append(date)
                left_year.append(date.year)
                try:
                    df_i = get_df_index(df, helcats_name)
                    left_c_vals.append(df.complexity[df_i])                    
                except:
                    left_c_vals.append(np.NaN)
            else:
                right_craft.append(craft)
                right_date.append(date)
                right_year.append(date.year)
                try:
                    df_i = get_df_index(df, helcats_name)
                    right_c_vals.append(df.complexity[df_i])
                except:
                    right_c_vals.append(np.NaN)
    pc_df['left_complexity'] = pd.Series(left_c_vals, index=pc_df.index)
    pc_df['right_complexity'] = pd.Series(right_c_vals, index=pc_df.index)
    pc_df['left_craft'] = pd.Series(left_craft, index=pc_df.index)
    pc_df['right_craft'] = pd.Series(right_craft, index=pc_df.index)
    pc_df['left_date'] = pd.Series(left_date, index=pc_df.index)
    pc_df['right_date'] = pd.Series(right_date, index=pc_df.index)
    pc_df['left_year'] = pd.Series(left_year, index=pc_df.index)
    pc_df['right_year'] = pd.Series(right_year, index=pc_df.index)
    pc_df['winner'] = pd.Series(winner, index=pc_df.index)
    pc_df = add_col_to_df(pc_df, 'left_complexity', 'right_complexity', 'subtract', 'complexity_diff', abs_col=True)
    pc_df = add_col_to_df(pc_df, 'left_wins', 'right_wins', 'add', 'total_votes', abs_col=False)
    return pc_df

    
"""
Miscellaneous useful functions.

# To import these (as useful_code is not a module) use this code: (one of the locations)
sys.path.insert(0, 'N:\\Documents\\Code\\useful_code')
sys.path.insert(0, 'C:\\Users\\shann\\OneDrive\\Documents\\Research\\Workspace\\Code\\useful_code')
import misc
"""
import os
import glob
import numpy as np
import pandas as pd
import astropy.units as u
from datetime import datetime, timedelta


# def get_project_dirs():
#     """
#     A function to load in dictionary of project directories stored in a
#     project_directories.txt file stored in the working directory.
#     :return proj_dirs: Dictionary of project directories.
#     """
#     files = glob.glob(os.path.join(os.getcwd(), 'project_directories.txt'))
#     # Open file and extract
#     with open(files[0], "r") as f:
#         lines = f.read().splitlines()
#         proj_dirs = {l.split(',')[0]: l.split(',')[1] for l in lines}
#     # Check the directories exist.
#     # for val in iter(proj_dirs.items()):
#     #     if not os.path.exists(val[1]):
#     #         print('Error, invalid path, check config: ' + val[1])
#     return proj_dirs


def find_nearest(array, value):
    """
    Finds the index of the array value nearest to input value.
    :param: array: array in which to search for nearest value
    :paran: value: value which wish to find nearest in array
    :return: index of nearest value in array
    """
    # handle pandas Series case
    if isinstance(array, pd.Series):
        array = array.values
    # add units if none
    value = value * u.dimensionless_unscaled
    array = array * u.dimensionless_unscaled
    value = value.to(array.unit)
    value = value.value
    array = array.value
    ds = []
    for i in range(len(array)):
        ds.append(array[i] - value)
    idx = (np.abs(ds)).argmin()
    return idx


def reshape_df(df, x_list, y_list, x, y, c):
    """Reshapes df.c to have shape len(x_list)xlen(y_list) for use with
    plt.pcolormesh.
    x, y, c are names of df columns
    """
    c_vals = []
    for yn in y_list:
        row = []
        dfy = df[df[y] == yn]
        for xn in x_list:
            dfyx = dfy[dfy[x] == xn]
            if len(dfyx) == 1:
                row.append(dfyx[c].values[0])
            else:
                row.append(np.NaN)
        c_vals.append(row)
    return c_vals


def find_time_means(val_list, time_list, t, start_month=1, err='std'):
    """ Find means of time data.
    val_list: data to find means of
    time_list: times linked to each of the values in val_list
    t: int, time length in months of data to average.
    start_month: int, month of year to start the means on (for running means)
    err: 'std' for standard deviation or 'sem' for standard error
    """
    if len(time_list) != len(val_list):
        raise ValueError('val_list and time_list must have same length')
    # Arrays for outputs
    dates = []
    means = []
    errs = []
    year = np.min(time_list).year
    month = start_month
    while year <= np.max(time_list).year:
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
        for n, i in enumerate(time_list):
            if i >= start_date:
                if i < end_date:
                    points.append(val_list[n])
        dates.append(start_date + timedelta(weeks=26))
        if points != []:
            means.append(np.nanmean(points))
            s = np.std(points)
            if err == 'sem':
                s = s / np.sqrt(len(points))
            errs.append(s)
        else:
            means.append(np.NaN)
            errs.append(np.NaN)              
        month = month + t
        if month > 12:
            year = year + 1
            month = month - 12
    dates = np.unique(dates)
    return dates, means, errs


def find_time_running_means(val_list, time_list, t, err='std'):
    """Finds running means from time data.
    val_list: data to find means of
    time_list: times linked to each of the values in val_list
    t: int, time length in months of data to average.
    err: 'std' for standard deviation or 'sem' for standard error
    """
    dates = []
    means = []
    errs = []
    # need to find means starting with each month of the year
    for i in range(1, 13):
        dates2, means2, errs2 = find_time_means(val_list, time_list, t, err=err,
                                           month=i)
        dates = np.append(dates2)
        means = np.append(means2)
        errs = np.append(errs2)
    return dates, means, errs

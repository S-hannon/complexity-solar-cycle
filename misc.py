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
import stereo_spice.StereoSpice as StereoSpice
from astropy.time import Time
from datetime import timedelta


def get_project_dirs():
    """
    A function to load in dictionary of project directories stored in a
    project_directories.txt file stored in the working directory.
    :return proj_dirs: Dictionary of project directories.
    """
    files = glob.glob(os.path.join(os.getcwd(), 'project_directories.txt'))
    # Open file and extract
    with open(files[0], "r") as f:
        lines = f.read().splitlines()
        proj_dirs = {l.split(',')[0]: l.split(',')[1] for l in lines}
    # Check the directories exist.
    # for val in iter(proj_dirs.items()):
    #     if not os.path.exists(val[1]):
    #         print('Error, invalid path, check config: ' + val[1])
    return proj_dirs


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


def add_stereo(fig, ax, time):
    labels = ['STEREO-A', 'STEREO-B', 'Earth']
    colours = ['r', 'b', 'limegreen']
    time = pd.date_range(time, time + timedelta(days=1))
    time = Time(time.to_pydatetime())
    for n, craft in enumerate(['sta', 'stb', 'ert']):
        spice = StereoSpice.StereoSpice()
        coords = spice.get_lonlat(time, craft, "CARR")
        r = (coords[0][0] * u.km).to(u.solRad)
        lon = (coords[0][1] * u.deg).to(u.rad)
        spice.clear_kernals()
        # convert m to solar radii
        ax.scatter(lon, r, color=colours[n], marker='o',
                   label=labels[n])
        ax.legend()
        if craft == 'ert':
            ax.set_theta_offset((3*np.pi/2)-lon.value)  # Rotates plot counter-clockwise
    return fig, ax
    
    
def add_stereo_fov(fig, ax, time):
    colours = ['r', 'b']
    labels = ['STEREO-A', 'STEREO-B']
    for n, craft in enumerate(['sta', 'stb']):
        ###########################################
        # get ecliptic pa
        spice = StereoSpice.StereoSpice()
        times = pd.date_range(time, time + timedelta(days=1))
        times = Time(times.to_pydatetime())
        ert_hpc = spice.get_lonlat(times, 'earth', system='hpc',
                                   observatory=craft)
        hpr_el, hpr_pa = spice.convert_hpc_to_hpr(ert_hpc[:, 1], ert_hpc[:, 2])
        ecliptic = hpr_pa[0] * u.deg
        spice.clear_kernals()
        ###########################################
        # get craft FOV lons & rs
        # get any old r in km
        r = (215 * u.solRad).to(u.km)
        lons = []
        rs = []
        els = [4, 24]
        for el in els:
            spice = StereoSpice.StereoSpice()
            hpc_lon, hpc_lat = spice.convert_hpr_to_hpc(el, ecliptic.value)
            rtn_lon, rtn_lat = spice.convert_hpc_to_rtn(hpc_lon, hpc_lat)
            carr = spice.convert_lonlat(dates=Time(time),
                                        coord_src=np.array([r.value, rtn_lon, rtn_lat]),
                                        system_src='rtn',
                                        system_dst='carr',
                                        observe_src=craft)
            lons.append((carr[1] * u.deg).to(u.rad))
            new_r = (carr[0] * u.km).to(u.solRad)
            rs.append(new_r)
            spice.clear_kernals()
        ############################################
        # now get craft location
        time2 = pd.date_range(time, time + timedelta(days=1))
        time2 = Time(time2.to_pydatetime())
        spice = StereoSpice.StereoSpice()
        coords = spice.get_lonlat(dates=time2,
                                  target=craft,
                                  system="CARR")
        r_craft = (coords[0][0] * u.km).to(u.solRad)
        lon_craft = (coords[0][1] * u.deg).to(u.rad)
        spice.clear_kernals()
        ############################################
        # and add to the plot
        ax.plot([lon_craft.value, lons[0].value],
                [r_craft.value, rs[0].value],
                color=colours[n], linestyle='--',
                label=labels[n] + ' FOV')
        ax.plot([lon_craft.value, lons[1].value],
                [r_craft.value, rs[1].value],
                color=colours[n], linestyle='--')
    ax.legend()
    return fig, ax
    

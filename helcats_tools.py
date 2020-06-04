"""
Useful functions for dealing with HI images, using Luke's hi_processing module.
hi_map is a SunPy Map object, containing coordinate informaton for the image.

To import these (as useful_code is not a module) use this code:
sys.path.insert(0, r'C:\\...\\useful_code')
import helcats_tools as ht
"""
from __future__ import division
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib as mpl
import sunpy.map as smap
from datetime import datetime, timedelta
import hi_processing.images as hip
import matplotlib.pyplot as plt
from PIL import Image
from misc import get_project_dirs, find_nearest
from date_tools import date2CRnum
from helcats_cats import HELCATSCatalogue


def get_cme_details(helcats_name):
    """Returns the craft and date of helcats_name.
    :param: helcats_name: str, helcats id of cme e.g. HCME_B__20130830_01
    """
    if helcats_name[5] == 'A':
        craft = 'sta'
    elif helcats_name[5] == 'B':
        craft = 'stb'
    date = datetime.strptime(helcats_name[8:16], '%Y%m%d')
    return craft, date


###############################################################################
# input checks
def check_helcats_name(helcats_name):
    """Checks helcats_name is a string, then checks whether helcats_name is a
    proper HELCATS id by searching for it in hicat.
    """
    if not isinstance(helcats_name, str):
        raise ValueError("helcats_name is not str")
    hicat = load_helcats_cat("hicat")
    if helcats_name not in hicat.id.values:
        raise ValueError("helcats_name not found")


def check_craft(craft):
    """Checks craft input is either "sta" or "stb".
    """
    if (craft != 'sta') & (craft != 'stb'):
        raise ValueError("craft input must be sta or stb")


def check_date(date):
    """Checks date input is a datetime instance.
    """
    if not isinstance(date, datetime):
        raise ValueError("date input is not a datetime instance")
        

###############################################################################
# Using HELCATS catalogues
# project_directories.txt: helcats_cats,C:\...\HELCATS,
    
def load_helcats_cat(cat):
    """Loads the required HELCATS catalogue.
    cat = str
    """
    hc = HELCATSCatalogue()
    return hc.load(cat)


def find_nearest_event(time, craft, max_timedelta=timedelta(hours=24)):
    """Returns the helcats name of the helcats CME nearest to the specified
    time. If there is no match within max_timedelta returns None.
    """
    df = load_helcats_cat("hicat")
    # loop over events, and find the closest match
    smallest_time_diff = timedelta(hours=24*365*20)
    best_index = None
    for i in df.index:
        if df.craft[i] == craft:
            if abs(df.time[i] - time) < smallest_time_diff:
                smallest_time_diff = abs(df.time[i] - time)
                best_index = i
    if smallest_time_diff < max_timedelta:
        return df.helcats_name[best_index]
    else:
        return None


def find_cme_match(helcats_name):
    """For a given CME helcats_name, returns the name of the same CME as
    observed by the other spacecraft, or None if there is no match.
    """
    df = load_helcats_cat('hijoincat')
    for i in df.index:
        if df['sta'][i] == helcats_name:
            return df['stb'][i]
        elif df['stb'][i] == helcats_name:
            return df['sta'][i]
    return None


def match_to_helcats(time_list, craft_list, time_diff=1, time_lag=0):
    """Matches a whole list of CMEs to HELCATS CMEs. MUCH faster than using
    find_nearest_event and looping over the whole catalogue for every event.
    time_diff: +/- days events much match within
    time_lag: allowable time for CME to travel between time_list and HI1
    """
    # check inputs
    if len(time_list) != len(craft_list):
        raise ValueError("time and craft lists must have same length")
    # make df of input cmes so easy to sort
    df = pd.DataFrame({'time' : time_list, 'craft' : craft_list})
    # ensure sorted in time order
    df = df.sort_values(by='time')
    d = dict()
    # import hicat
    hicat = load_helcats_cat("hicat")
    hicat = hicat.sort_values(by='time')
    # do matching
    for craft in ['sta', 'stb']:
        dfs = df[df['craft'] == craft]
        his = hicat[hicat['craft'] == craft]
        # now... match
        for i in dfs.index:
            helcats_name = np.NaN
            j_c = 0
            while j_c < len(his):
                j = his.index[j_c]
                this_td = (dfs.time[i] - his.time[j]).days
                # if greater than time_max, remove it and move on
                # his will start further back in time
                if this_td > time_diff:
                    his = his.drop(j)
                    j_c = j_c + 1
                # if less than time_min, stop looking
                # then after time
                elif this_td < - time_diff - time_lag:
                    break
                # if this is a match
                else:
                    # save the name
                    helcats_name = his.helcats_name[j]
                    # remove from the list
                    his = his.drop(j)
                    break
            d[i] = helcats_name
    name_list = []
    for cme in df.index:
        name_list.append(d[cme])
    # print(name_list[1185:1195])
    # print(craft_list[1185:1195])
    return name_list


def event_spacing(hours):
    """Loops over HELCATS catalogue, finding the number of CMEs with a 2nd CME
    following within the time specified.
    """
    # check inputs
    if not isinstance(hours, int):
        raise ValueError("hours input must be int")
    # get craft subset
    for craft in ['sta', 'stb']:
        hicat = load_helcats_cat("hicat")
        hicat = hicat[hicat.craft == craft]
        # TODO check hicat in order
        count = 0
        prev_cme_time = None
        # Seperate case for CME 0
        for n, i in enumerate(hicat.index):
            # first CME has no wait
            if n == 0:
                prev_cme_time = hicat['Date [UTC]'][i]
            else:
                wait = hicat['Date [UTC]'][i] - prev_cme_time
                if wait < timedelta(hours=hours):
                    count = count + 1
                prev_cme_time = hicat['Date [UTC]'][i]
        percent = int((count / len(hicat)) * 100)
        print("%s CMEs (%s%s) within %s hours of previous CME in %s."%(count, str(percent), chr(37), str(hours), craft))


def CRnum_HELCATS(data="TE"):
    """Finds the range of Carrington Rotations for which HELCATS CMEs observed.
    """
    if data == "TE":
        df = get_all_helcats_cmes()
        col = 'date'
    else:
        df = load_helcats_cat("hicat")
        col = 'Date [UTC]'
    for n, i in enumerate(df.index):
        if n == 0:
            start_date = df[col][i]
            start_CR = int(date2CRnum(start_date))
        elif n == len(df) - 1:
            end_date = df[col][i]
            end_CR = int(date2CRnum(end_date))
    print(start_date, start_CR, end_date, end_CR)
    

###############################################################################
# Using time-elongation tracks
# project_directories.txt: helcats_te,C:\...\HCME_WP3_V06_TE_PROFILES,
 
def time_in_fov(camera='hi1'):
    """Looks into the average time the CMEs remain in the camera FOV.
    """
    # get time in FOV values
    df = get_all_helcats_cmes()
    time_in_fov = []
    for helcats_name in df.helcats_name:
        start, mid, end = get_hi_times_from_te(helcats_name, camera=camera)
        if (start != None) & (end != None):
            diff = end - start
            in_hours = diff / timedelta(hours=1)
            time_in_fov.append(in_hours)
    # calc values
    mean = int(np.mean(time_in_fov))
    median = int(np.median(time_in_fov))
    q75, q25 = np.percentile(time_in_fov, [75 ,25])
    q75, q25 = int(q75), int(q25)
    mini = int(np.min(time_in_fov))
    maxi = int(np.max(time_in_fov))
    # print them
    print("Mean: %s hours"%(mean))
    print("Median: %s hours"%(median))
    print("IQR: %s to %s hours"%(q25, q75))
    print("Range of times: %s to %s hours"%(mini, maxi))
    # do a box plot
    font = {'size' : 16}
    plt.figure(figsize=(5, 12))
    plt.boxplot(time_in_fov)
    plt.ylabel("Time in %s FOV (hours)"%(camera), **font)
    plt.xlabel("All HELCATS CMEs", **font)
    

def multiple_cmes_at_mid():
    """Finds the number of events with a 2nd CME present when the CME reaches
    the mid time in hi1.
    """
    df = get_all_helcats_cmes()
    for craft in ['sta', 'stb']:
        count = 0
        dfc = df[df.craft == craft]
        for n, i in enumerate(dfc.index):
            start, mid, end = get_hi_times_from_te(dfc.helcats_name[i])
            if (mid != None):
                # look at next cme
                for m in range(n, len(dfc)-1):
                    next_i = dfc.index.values[m+1]
                    next_start, next_mid, next_end = get_hi_times_from_te(dfc.helcats_name[next_i])
                    if next_start != None:
                        diff = (mid - next_start) / timedelta(hours=1)
                        if diff > 0:
                            # mid is after next start, so next CME present
                            count = count + 1
                        break
        percent = int((count / len(dfc))*100)
        print("%s CMEs (%s%s) have second CMEs present at their mid-time in %s."%(str(count), str(percent), chr(37), craft))   


def get_all_helcats_cmes():
    """Returns df containing all helcats names with craft and date from
    time-elongation tracks.
    """# nb could be multiple on same day
    df = pd.DataFrame(columns=['helcats_name', 'craft', 'date'])

    proj_dirs = get_project_dirs()
    # Loop through all time elongation profiles in the folder
    for filename in os.listdir(proj_dirs['helcats_te']):
        # extract helcats name from file name
        helcats_name = ('_').join(filename.split('_')[0:5])
        craft, date = get_cme_details(helcats_name)
        df = df.append({'helcats_name' : helcats_name, 'craft' : craft,
                        'date' : date}, ignore_index=True)
    return df


def find_cmes(date_range=[], craft='both'):
    """Returns a list of helcats CMEs observed by selected craft, during
    date_range.
    date_range: array with [0] start date, [1] end date, datetime objects
    craft: 'both', 'sta', or 'stb'
    """
    # TODO: Check inputs
    cmes = get_all_helcats_cmes()
    if date_range != []:
        cmes = cmes[cmes.date > date_range[0]]
        cmes = cmes[cmes.date < date_range[1]]
    if craft != 'both':
        cmes = cmes[cmes.craft == craft]
    return cmes.helcats_name.values


def get_te_profile(helcats_name):
    """Returns pandas df containing time-elongation tracks for one CME.
    """
    proj_dirs = get_project_dirs()
    # Loop through all time elongation profiles in the folder
    for filename in os.listdir(proj_dirs['helcats_te']):
        if filename.startswith(helcats_name):
            # Read the data
            df = pd.read_table(os.path.join(proj_dirs['helcats_te'], filename),
                               delim_whitespace=True,
                               names=['profile', 'time', 'el', 'pa', 'craft'])
    # drop craft column as unneccesary (helcats_name tells you the craft)
    df = df.drop(columns='craft')
    # Use converter function for correct date format
    df.time = pd.to_datetime(df.time)
    return df


def get_hi_times_from_te(helcats_name, camera='hi1'):
    """Gets the start, mid, and end time for a given helcats CME, using the 
    time elongation tracks.
    camera = str, 'both', 'hi1' or 'hi2'
    """
    df = get_te_profile(helcats_name)
    # Split into hi1 and hi2 FOVs if needed
    if camera == 'hi1':
        el_range = [4, 24]  # HI-1 observes from 4 to 24 deg el
    elif camera == 'hi2':
         # HELCATS J-Maps use HI-1 to 24 deg el, then HI-2
        el_range = [24, 88.7]  # HI-2 from 18.7 to 88.7
    else:
        el_range = [4, 88.7]
    in_view = df[df.el >= el_range[0]]
    in_view = in_view[in_view.el < el_range[1]]
    # Check there are any observations with correct elongation
    if len(in_view) > 0:
        # Sort TE profile, so dates in ascending order
        in_order = in_view.sort_values(by='time')
        # Append first time in view to list
        start = in_order.time[in_order.index[0]]
        # Append last time in view to list
        end = in_order.time[in_order.index[len(in_order)-1]]
        # Append time CME reaches halfway through FOV
        mid_el = sum(el_range) / 2
        idx = find_nearest(in_order.el.values, mid_el)
        mid = in_order.time.values[idx]
    else:
        start = None
        mid = None
        end = None
    return start, mid, end


###############################################################################
# Using HI hard-drive
# project_directories.txt: hi_data,D:\STEREO\ares.nrl.navy.mil\lz
#                          hi_data2,E:

def save_fits_files_one_cme(helcats_name, camera='hi1', background_type=1):
    """function to download .fits files from hard drive for one HELCATS event.
    """
    craft, date = get_cme_details(helcats_name)
    # get times the CME is in the HI FOV
    start, mid, end = get_hi_times_from_te(helcats_name, camera=camera)
    
    hi_files = hip.find_hi_files(start, end, craft=craft, camera=camera,
                                 background_type=background_type)
    for i in range(len(hi_files)):
        # using first file, make new folders
        path_end_plus_file_name = hi_files[i].split(get_project_dirs()['hi_data'])[1]
        parts = path_end_plus_file_name.split('\\')
        path_end = ('\\').join(parts[1:len(parts)-1])            
        # make folder to store data
        out_dir = get_project_dirs()['out_data']
        if not os.path.exists(os.path.join(out_dir, 'HIdata', path_end)):
            os.makedirs(os.path.join(out_dir, 'HIdata', path_end))
        file_name = parts[len(parts)-1]
        shutil.copy2(hi_files[i], os.path.join(out_dir, 'HIdata', path_end, file_name))
    

def get_hi_map(craft, date):
    """Retrieves hi_map from hard drive for a given CME, using the time the
    CME was first observed in HI, given in the HELCATS ID for the CME.
    """
    day_string = str(date.year) + str(date.month).zfill(2) + str(date.day).zfill(2)
    time_string = str(date.hour).zfill(2) + str(date.minute).zfill(2)
    # folder path depends on hard drive
    project_dirs = get_project_dirs()
    # Look on correct hard drive!
    if date.year < 2014:
        # Find the folder on the correct day
        hi_folder = os.path.join(project_dirs['hi_data'], "L2_1_25", str(craft[2]), 'img\hi_1', day_string)
    else:
        hi_folder = os.path.join(project_dirs['hi_data2'], "L2_1_25", str(craft[2]), "img\hi_1", day_string)
    # find correct file from folder
    for filename in os.listdir(hi_folder):
        if filename.startswith(day_string + "_" + time_string):
            hi_map = smap.Map(os.path.join(hi_folder, filename))
    return hi_map


def get_CME_img(helcats_name, time, folder):
    """Loads the image for the required CME from the folder specified in the
    project directiories.
    """
    time_string = time.strftime("%Y%m%d_%H%M%S")
    for file_name in os.listdir(folder):
        if file_name.endswith(helcats_name + '_diff_' + time_string + '.jpg'):
            img_path = os.path.join(folder, file_name)
            img = Image.open(img_path)
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
    return img


def make_diff_img(helcats_name, time, camera='hi1', background_type=1,
                  save=False):
    """Makes the differenced image for the required CME using the .FTS files
    on the hard-drives.
    date should be datetime object with date and time!
    craft = "sta" or "stb"
    
    save = False or path/to/folder to save into
    """
    craft, hi1_start_date = get_cme_details(helcats_name)
    out_img = False
    out_name = False
    hi_map = False
    # TODO Check date input within range
    # Search hard drive for matching HI image
    hi_files = hip.find_hi_files(time - timedelta(hours=2), time,
                                 craft=craft, camera=camera,
                                 background_type=background_type)
    if len(hi_files) == 0:
        # Try 5 mins later, should avoid errors with the seconds being wrong
        # this is okay as cadence ~40 mins
        hi_files = hip.find_hi_files(time - timedelta(hours=2),
                                     time + timedelta(minutes=5),
                                     craft=craft, camera=camera,
                                     background_type=background_type)   
    if len(hi_files) > 1:
        # Loop over the hi_files, make image
        fc = hi_files[len(hi_files)-1]  # current frame files, last in list
        fp = hi_files[len(hi_files)-2]  # previous frame files, 2nd last
        # Make the differenced image
        hi_map = hip.get_image_diff(fc, fp, align=True, smoothing=True)
        # Make it grey
        diff_normalise = mpl.colors.Normalize(vmin=-0.05, vmax=0.05)
        out_img = mpl.cm.gray(diff_normalise(hi_map.data), bytes=True)
        out_img = Image.fromarray(out_img)
        if save != False:
            # Save the image
            out_name = "_".join([helcats_name, craft, hi_map.date.strftime('%Y%m%d_%H%M%S')]) + '.jpg'
            out_path = os.path.join(save, out_name)
            out_img = out_img.convert(mode='RGB')
            out_img.save(out_path)    
    return out_img, hi_map

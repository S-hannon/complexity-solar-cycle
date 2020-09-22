"""Script to load HELCATS catalogue.
author: S. R. Jones s.jones2@pgr.reading.ac.uk

The HELCATS project... blah...
Contains many "workpackages", hereafter referred to as 'catalogues' or 'cats'

Example usage:
### Need to run this before doing anything else (data_loc should be the path
    to the folder where you have saved the helcats data)
hc = HELCATS(data_loc)

### to load the data
# load a catalogue as pandas DataFrame
df = hc.load("hicat")

# load one time-elongation as a pandas DataFrame
df = hc.load_te_profile('HCME_A__20081212_01')

### to get info about the catalogues
# print list of catalogues included
hc.info()

# find out about a catalogue
hc.cat_info("hicat")

# find out about a column
hc.col_info("SSE Phi [deg]")

# search for columns containing string
hc.search("speed")

# find out which catalogue contains a column
cat_name = hc.find_col("SSE Phi [deg]")

### finding HELCATS CMEs
# find list of HELCATS CMEs meeting certain criteria
date_range = [datetime(2007, 1, 1, 0, 0), datetime.now()]
cme_list = hc.find_cmes(date_range=date_range, craft='both')

# find list of HELCATS CMEs to match existing cme time list
time_list = [datetime(2008, 12, 12, 12, 12), datetime(2008, 12, 12, 12, 12)]
craft_list = ['sta', 'stb']
cme_list = hc.match_to_helcats(time_list, craft_list)

### retrieve data about a CME
# get values from a column, for a list of helcats cmes
helcats_name_list = ['HCME_A__20081212_01', 'HCME_B__20081212_01']
col_data = hc.get_col_data("SSE Phi [deg]", helcats_name_list)
"""

import os
import glob
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import misc


class HELCATS:
    def __init__(self, data_loc):
        # find HELCATS data folder, make it if it doesn't exist
        self.folder = os.path.join(data_loc, 'HELCATS')
        if os.path.exists(self.folder) == False:
            os.makedir(self.folder)
        # load the catalogues
            # TODO stop the order of this array mattering
        cat_names = ['hicat', 'hijoincat', 'higeocat', 'arrcat', 'kincat',
                     'icmecat', 'linkcat']
        cat_instances = [WP2HICAT(), WP2HIJoinCAT(), WP3HIGeoCAT(),
                         WP3ARRCAT(), WP3KINCAT(), WP4ICMECAT(),  WP4LINKCAT()]
        self.cat_instances = dict({})
        for n, name in enumerate(cat_names):
            self.cat_instances[name] = cat_instances[n]
    

    ############### load a catalogue
    def __download(self, cat_name):
        """Downloads a catalogue. Only works for some of them!
        """
        cat = self.cat_instances[cat_name.lower()]
        df = pd.read_json(cat.data_url, orient='split')
        df = df.rename(columns=cat.to_rename)
        # save as .csv file as easy to read
        file_path = os.path.join(self.folder, cat.name + '.csv')
        df.to_csv(file_path, sep=",", index=False)      
        print('...successfully downloaded ' + cat_name)        
        
        
    def load(self, cat_name):
        """Returns HELCATS catalogue cat_name as pd df.
        """
        self.check_cat_name(cat_name.lower())
        cat = self.cat_instances[cat_name.lower()]
        file_path = os.path.join(self.folder, cat.name + '.csv')
        # if cat doesn't exist, download and save as nice .csv
        if os.path.exists(file_path) == False:
            self.__download(cat_name)
        # now load the cat from csv
        df = pd.read_csv(file_path, na_values=['nan', '9999-99-99 99:99'])
        # remove duplicate columns
        df = df.drop(columns=cat.to_drop)
        # converters
        for col in cat.to_str:
            df[col] = df[col].astype(str)   
        for col in cat.to_int:
            df[col] = df[col].astype(int)
        for col in cat.to_float:
            df[col] = df[col].astype(float)
        for col in cat.to_datetime:
            try:
                df[col] = pd.to_datetime(df[col], dayfirst=True)
            except:
                df[col] = df[col].astype(str)
                # add in catch for: ParserError: Unknown string format: 2011-11-0914:12Z
                times = []
                for i in df.index:
                    t = df[col][i]
                    if t[len(t)-1] == 'Z':
                        t = t[0:len(t)-1]
                        t = datetime.strptime(t, '%Y-%m-%d%H:%M')
                        times.append(t)
                    else:
                       times.append(t)
                df[col] = pd.Series(times, index=df.index)
                df[col] = pd.to_datetime(df[col], dayfirst=True)
        # add craft column
        if (cat_name != 'hijoincat') & (cat_name != 'icmecat'):
            craft_list = []
            for cme in df.helcats_name:
                if cme[5] == 'A':
                    craft = 'sta'
                elif cme[5] == 'B':
                    craft = 'stb'
                craft_list.append(craft)
            df['craft'] = pd.Series(craft_list, index=df.index)
        return df


    def load_te_track(self, helcats_name):
        """Returns pandas df containing time-elongation tracks for one CME.
        """
        # Loop through all time elongation profiles in the folder
        # TODO guidelines for downloading these
        te_data_loc = os.path.join(self.folder, 'HCME_WP3_V06_TE_PROFILES')
        for filename in os.listdir(te_data_loc):
            if filename.startswith(helcats_name):
                # Read the data
                df = pd.read_table(os.path.join(te_data_loc, filename),
                                   delim_whitespace=True,
                                   names=['profile', 'time', 'el', 'pa', 'craft'])
        # drop craft column as unneccesary (helcats_name tells you the craft)
        df = df.drop(columns='craft')
        # Use converter function for correct date format
        df.time = pd.to_datetime(df.time)
        return df
    
    
    ############### input checks     
    def check_cat_name(self, cat_name):
        """Checks cat_name is a valid HELCATS catalogue name.
        """
        if cat_name not in self.cat_instances:
            raise ValueError("Requested cat not found.")


    def check_helcats_name(self, helcats_name):
        """Checks helcats_name is a string, then checks whether helcats_name is a
        proper HELCATS id by searching for it in hicat.
        """
        if not isinstance(helcats_name, str):
            raise ValueError("helcats_name is not str")
        hicat = self.load("hicat")
        if helcats_name not in hicat.helcats_name.values:
            raise ValueError("helcats_name not found")
            
            
    ############### getting info about the catalogue
    def info(self):
        """Prints names of included catalogues.
        """
        print(['hicat', 'arrcat', 'higeocat',
               'hijoincat', 'icmecat', 'kincat', 'linkcat'])
                
                
    def col_info(self, col):
        """Prints infomation on selected column.
        """
        col = col.lower()
        file_path = os.path.join(self.folder, 'helcats_guide'+ '.csv')
        df = pd.read_csv(file_path, header=0)
        df['col'] = df['col'].str.lower()
        if col in df.col.values:
            print("Column Name: %s"%(col))
            print("Catalogue: %s"%(df[df.col == col]['cat'].values[0]))
            print("Unit: %s"%(df[df.col == col]['unit'].values[0]))
            print("Description: %s"%(df[df.col == col]['description'].values[0]))
        else:
            raise ValueError("Requested col not found.")


    def cat_info(self, cat_name):
        """Prints information on selected HELCATS catalogue.
        """
        cat_name = cat_name.lower()
        self.check_cat_name(cat_name)
        cat = self.cat_instances[cat_name]
        df = self.load(cat_name)
        print("Catalogue Name: %s"%(cat.name))
        print("Columns: %s"%([i for i in df.columns]))
        print("Reference paper: %s"%(cat.reference))
        print("Data downloaded from: %s"%(cat.data_url))
        print("More info can be found at: %s"%(cat.help_url)) 
 

    def search(self, value):
        """Searches for column names containing the value string. Case
        insensitive.
        Returns list of column names, with the cat that col is found in
        """
        file_path = os.path.join(self.folder, 'helcats_guide'+ '.csv')
        df = pd.read_csv(file_path, header=0)
        for n, col in enumerate(df.col.values):
            if value.lower() in col.lower():
                print(col, '(' + df.cat[n] + ')')
                
        
    def find_col(self, col):
        """Returns the name of the HELCATS cat containing the column "col".
        """
        if not isinstance(col, str):
            raise ValueError("col must be str")
        for cat in self.cat_instances.keys():
            df = self.load(cat)
            if col in df.columns:
                return cat
        raise ValueError("Requested property not found in any cats.")
        

    ############### finding CMEs in the catalogue
    def find_cmes(self, date_range=[], craft='either', geo=False, insitu=False,
                  solar=False, gcs=False, te_track=False):
        """Returns a list of helcats CMEs which meet the input criteria.
        date_range: array with [0] start date, [1] end date, datetime objects
        craft (str): 'either', 'sta', 'stb' or 'both' - which craft must cmes
            be observed by?
        geo (bool): if true, only returns CMEs which were geoeffective
        insitu (bool): if true, only returns CMEs with linked in-situ data
        solar (bool): if true, only returns CMEs with linked solar data
        gcs (bool): if true, only returns CMEs with GCS modelling data
        te_track (bool): if true, only returns CMEs for which the
            time-elongation tracking file is available.
        """
        if len(date_range) != 2:
            raise ValueError("date range must be list with length 2")
        for n in [0, 1]:
            if not isinstance(date_range[n], datetime):
                raise ValueError("date_range must contain datetime objects")
        valid_craft = ['either', 'sta', 'stb', 'both']
        if craft not in valid_craft:
            raise ValueError("craft input must be in: %s"%(valid_craft))
        for criteria in [geo, insitu, solar, gcs, te_track]:
            if criteria not in [True, False]:
                raise ValueError(criteria + " must be True or False")
        hicat = self.load('hicat')
        if date_range != []:
            hicat = hicat[hicat.time > date_range[0]]
            hicat = hicat[hicat.time < date_range[1]]
        if craft in ['sta', 'stb']:
            hicat = hicat[hicat.craft == craft]
        elif craft == 'both':
            matches = self.get_match_list(hicat.helcats_name)
            hicat['matches'] = pd.Series(matches, index=hicat.index)
            # drop rows with no match
            hicat = hicat.dropna(subset=['matches'])
        if geo == True:
            geo_list = self.get_geo_indicator_list(hicat.helcats_name)
            hicat['geo'] = pd.Series(geo_list, index=hicat.index)
            hicat = hicat[hicat.geo == True]
        if te_track == True:
            te_track_list = self.get_te_track_indicator_list(hicat.helcats_name)
            hicat['te_track'] = pd.Series(te_track_list, index=hicat.index)
            hicat = hicat[hicat.te_track == True]
        # TODO: Extend to insitu data only
        # TODO: Extend to solar data only
        # TODO: Extend to gcs modelling only
        return hicat.helcats_name.values


    def __do_helcats_matching(self, df, hdf, window=[timedelta(hours=12),
                                                     timedelta(hours=12)],
                              allow_duplicates=False):
        # initialise dictionary to store helcats names
        d = dict()
        # first search should start with the earliest HELCATS CME
        start_id = 0  
        # start looping over the input cmes
        # note this loops over the indexes of CMEs for this craft only
        for i in df.index:
            helcats_name = np.NaN  # initialse output
            # search for this CME in HELCATS, starting at index start_id
            cid = start_id  # current id
            while cid < len(hdf):  # if this is a valid id
                # get the index of hicatc we are comparing now
                j = hdf.index[cid]
                # find the difference between the times
                # this will be positive if the helcats CME occured earlier
                # and negative if the helcats CME occured later
                td = df.time[i] - hdf.time[j]
                ### 1. if the helcats CME occured more than window[0] before
                if td > window[0]:
                    # as the df sorted by time, if the helcats CME occurs before
                    # this one, it will occur before every CME in the list
                    # so start all later searches after this CME
                    start_id = start_id + 1
                    # move on to the next helcats CME
                    cid = cid + 1
                ### 2. if the helcats CME occured more than window[1] after
                elif td < - window[1]:
                    # there is no match for this CME
                    # End the while loop, and start a new search.
                    break
                ### 3. if the helcats CME is within the time window
                else:
                    # is this the only CME within the time window?
                    # make list of cmes in time window
                    cid_window = [cid]
                    td_window = [td]
                    j_window = [j]
                    # now continue searching later CMEs
                    cid2 = cid + 1
                    while cid2 < len(hdf):
                        j2 = hdf.index[cid2]
                        this_td2 = (df.time[i] - hdf.time[j2])
                        if this_td2 < - window[1]:
                            # end of cmes within window
                            break
                        else:
                            # this cme also in window
                            cid_window.append(cid2)
                            td_window.append(this_td2)
                            j_window.append(j2)
                            # now try the next one
                            cid2 = cid2 + 1
                    # now find the closest match in time from the list
                    if len(cid_window) == 1:
                        # if only one cme in window, select this one
                        best = 0
                    else:
                        # get absolute values of time differences
                        abs_td_window = []
                        for td in td_window:
                            abs_td_window.append(abs(td))
                        # select cme with smallest absolute time difference
                        best = np.argmin(abs_td_window)
                    # save the matching helcats name
                    helcats_name = hdf.helcats_name[j_window[best]]
                    if allow_duplicates == False:
                        # start the next search after this CME
                        start_id = cid_window[best] + 1
                    else:
                        # start the next search with this CME
                        start_id = cid_window[best]
                    break
            d[i] = helcats_name        
        return d
        

    def match_to_helcats(self, time_list, craft_list=[], insitu_craft=None,
                         window=[timedelta(hours=12), timedelta(hours=12)],
                         allow_duplicates=False):
        """Matches a whole list of CMEs to HELCATS CMEs. MUCH faster than 
        looping over the whole catalogue for every event.
        Looks for HELCATS cmes in time between:
            time_list - window[0] < x < time_list + window[1]
        Then matches to the CME nearest in time
        time_list, craft_list, window should all be lists or arrays
        window should contain 2 timedelta instances
        timedelta can be initialised with hours=24, days=1, seconds=27 etc.
        allow_duplicates is boolean. True indicates that a HELCATS cme can be
        matched to more than one input CME. False indicates that each HELCATS CME
        can only be matched once. This is useful for e.g. matching an AI catalogue
        which detects the same CME multiple times.
        Returns list of matching HELCATS cme names corresponding to time_list and
        craft_list.
        """
        # check window valid
        if len(window) != 2:
            raise ValueError("window must have length 2")
        for n in [0, 1]:
            if not isinstance(window[n], timedelta):
                raise ValueError("window must contain timedelta objects")
        # 1. if matching time_list & craft_list to HICAT HI1 times
        if craft_list != []:
            if len(time_list) != len(craft_list):
                raise ValueError("time and craft lists must have same length")
            df = pd.DataFrame({'time' : time_list, 'craft' : craft_list})
            hdf = self.load("hicat")
            # sort lists by time
            df = df.sort_values(by='time')
            hdf = hdf.sort_values(by='time')
            # initialise dictionary to store helcats names
            d = dict()
            # look at each craft separately
            for craft in ['sta', 'stb']:
                dfc = df[df['craft'] == craft]
                hdfc = hdf[hdf['craft'] == craft]
                dc = self.__do_helcats_matching(dfc, hdfc, window=window,
                                                allow_duplicates=allow_duplicates)
                d.update(dc)
        # 2. if matching time_list to LINKCAT icme times for insitu_craft
        else:
            if insitu_craft == None:
                raise ValueError("must provide insitu_craft or craft_list")
            df = pd.DataFrame({'time' : time_list})
            hdf = self.load("linkcat")
            if insitu_craft not in np.unique(hdf['SC_INSITU']):
                raise ValueError("insitu_craft not in list")
            hdf = hdf[hdf['SC_INSITU'] == insitu_craft]
            hdf['time'] = pd.Series(hdf['ICME_START_TIME [UTC]'],
                                    index=hdf.index)
            # sort lists by time
            df = df.sort_values(by='time')
            hdf = hdf.sort_values(by='time')
            d = self.__do_helcats_matching(df, hdf, window=window,
                                           allow_duplicates=allow_duplicates)
        # order df by the order that was input, not time order
        df = df.sort_index()
        # get list of helcats names from the dictionary, in this order
        name_list = []
        for k in df.index:
            name_list.append(d[k])
        return name_list


    ############### retrieve data about a CME
    def get_insitu_matches(self, helcats_name):
        """For a given CME helcats_name, returns any insitu measurements
        of the CME in linkcat.
        """
        df = self.load('linkcat')
        df = df[df.helcats_name == helcats_name]
        return df
        
        
    def get_cme_details(self, helcats_name):
        """Returns the craft and date of helcats_name.
        :param: helcats_name: str, helcats id of cme e.g. HCME_B__20130830_01
        """
        if helcats_name[5] == 'A':
            craft = 'sta'
        elif helcats_name[5] == 'B':
            craft = 'stb'
        date = datetime.strptime(helcats_name[8:16], '%Y%m%d')
        return craft, date

        
    def get_cme_details_list(self, helcats_name_list):
        """Returns the craft and date of helcats_name.
        :param: helcats_name: str, helcats id of cme e.g. HCME_B__20130830_01
        """
        craft_list = []
        date_list = []
        for helcats_name in helcats_name_list:
            craft, date = self.get_cme_details(helcats_name)
            craft_list.append(craft)
            date_list.append(date)
        return craft_list, date_list
    
    
    def get_te_track_indicator(self, helcats_name):
        """Looks up whether a given CME has a HELCATS time-elongation
        tracking file available. Returns True or False.
        """
        files = glob.glob(self.data_loc + '\\HCME_WP3_V06_TE_PROFILES\\' + helcats_name + '*')
        if len(files) == 1:
            return True
        elif len(files) == 0:
            return False
        else:
            raise ValueError("multiple te profiles found?!!!")


    def get_te_track_indicator_list(self, helcats_name_list):
        """Returns a bool list with True if the CME has a HELCATS time-
        elongation tracking available, and False otherwise.
        """
        te_track_ind = []
        for helcats_name in helcats_name_list:
            te_track_ind.append(self.__has_te_track(helcats_name))
        return te_track_ind
    
    
    def get_match(self, helcats_name, hijoincat=pd.DataFrame()):
        """For a given CME helcats_name, returns the name of the same CME as
        observed by the other spacecraft, or None if there is no match.
        """
        # if hasnt already been loaded
        if hijoincat.empty:
            hijoincat = self.load('hijoincat')
        craft, time = self.get_cme_details(helcats_name)
        row = hijoincat[hijoincat[craft] == helcats_name]
        # no match - helcats_name not in hijoincat
        if row.empty:
            return np.NaN
        else:
            if craft == 'sta':
                return row['stb'].values[0]
            else:
                return row['sta'].values[0]
    
    
    def get_match_list(self, helcats_name_list):
        """Adds a column 'match' containing the helcats name of the same CME
        observed by the other spacecraft, if it was.
        """
        hijoincat = self.load('hijoincat')
        match_list = []
        for helcats_name in helcats_name_list:
            match_list.append(self.get_match(helcats_name, hijoincat))
        return match_list


    def get_geo_indicator(self, helcats_name, linkcat=pd.DataFrame()):
        """Returns True if the CME was geoeffective, False otherwise.
        """
        if linkcat.empty:
            linkcat = self.load('linkcat')
        insitu_matches = linkcat[linkcat['helcats_name'] == helcats_name]
        if 'EARTH_L1' in insitu_matches['TARGET_NAME'].values:
            return True
        else:
            return False
 
    
    def get_geo_indicator_list(self, helcats_name_list):
        """Returns list which is True if the CME was geoeffective, False otherwise.
        """
        # Get data
        linkcat = self.load('linkcat')
        # Is the event geoeffective? Add 'geo' indicator to df
        geo_list = []
        for helcats_name in helcats_name_list:
            geo_list.append(self.get_geo_indicator(helcats_name, linkcat))
        return geo_list
    
    
    def get_te_track_times(self, helcats_name, camera='hi1'):
        """Gets the start, mid, and end time for a given helcats CME, using the 
        time elongation tracks.
        camera = str, 'both', 'hi1' or 'hi2'
        """
        df = self.load_te_track(helcats_name)
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
            idx = misc.find_nearest(in_order.el.values, mid_el)
            mid = in_order.time.values[idx]
        else:
            start = None
            mid = None
            end = None
            mid_el = None
        return start, mid, end, mid_el
    
    
    def get_te_track_times_list(self, helcats_name_list, camera='hi1'):
        """Gets the start, mid, and end time for a given helcats CME, using the 
        time elongation tracks.
        camera = str, 'both', 'hi1' or 'hi2'
        """
        start_list = []
        mid_list = []
        end_list = []
        mid_el_list = []
        for helcats_name in helcats_name_list:
            start, mid, end, mid_el = self.get_te_track_times(helcats_name,
                                                              camera=camera)
            start_list.append(start)
            mid_list.append(mid)
            end_list.append(end)
            mid_el_list.append(mid_el)
        return start_list, mid_list, end_list, mid_el_list

    
    def __get_col_vals(self, cat_df, col, helcats_name_list):
        col_vals = []
        for helcats_name in helcats_name_list:
            row = cat_df[cat_df['helcats_name'] == helcats_name]
            # if there's no data in cat for this cme
            if row.empty == True:
                col_vals.append(np.NaN)
            elif len(row) > 1:
                raise ValueError("cat has more than one value for this CME")
            else:
                col_vals.append(row[col].values[0])
        return col_vals

        
    def get_col_data(self, col, helcats_name_list, target=None):
        """Returns the col values for the cmes in helcats_name_list.
        target str from ['MESSENGER', 'VEX', 'Wind', 'STEREO-A', 'STEREO-B']
        """
        cat = self.find_col(col)
        cat_df = self.load(cat)
        # deal with the in-situ data (as there can be multiple for one cme)
        if cat in ['linkcat', 'icmecat', 'arrcat']:
            if target == None:
                raise ValueError('This column requires target input')
            target_list = ['MESSENGER', 'VEX', 'Wind', 'STEREO-A', 'STEREO-B']
            if target not in target_list:
                raise ValueError("target must be one of: " + str(target_list))            
            if cat == 'arrcat':
                if target == 'Wind':
                    target = 'EARTH_L1'
                elif target == 'VEX':
                    target = 'VENUS'
            cat_df = cat_df[cat_df['SC_INSITU'] == target]
        # get the data
        col_data = self.__get_col_vals(cat_df, col, helcats_name_list)
        return col_data  
 
    
    def get_matched_data(self, data_list, helcats_name_list):
        """Returns the col values for cmes in helcats_name_list which were
        observed by both STEREO-A and STEREO-B.
        """
        # get required helcats data
        match_list = self.get_match_list(helcats_name_list)
        craft_list, date_list = self.get_cme_details_list(helcats_name_list)
        # turn into a df as easier to sort
        df = pd.DataFrame({'helcats_name' : helcats_name_list,
                           'match' : match_list, 'data' : data_list,
                           'craft' : craft_list, 'time' : date_list})
        # drop cmes without a match
        df = df.dropna(subset=['match'])
        # join matched lists together
        dfa = df[df.craft == 'sta']
        dfb = df[df.craft == 'stb']
        df = dfa.merge(dfb, left_on=['helcats_name'], right_on=['match'],
                       suffixes=['_a', '_b'])
        return df['data_a'].values, df['data_b'].values


###############################################################################
# tedious details about each of the catalogues, needed to load it
class WP2HICAT:
    def __init__(self):
        self.name = 'hicat'
        self.help_url = "https://www.helcats-fp7.eu/catalogues/wp2_cat.html"
        self.data_url = 'https://www.helcats-fp7.eu/catalogues/data/HCME_WP2_V06.json'
        self.reference = ""
        self.to_drop = ['SC']
        self.to_rename = {'ID' : 'helcats_name',
                          'Date [UTC]' : 'time'}
        self.to_str = ['L:N',
                       'L:S',
                       'Quality']
        self.to_int = ['PA-N [deg]',
                       'PA-S [deg]']
        self.to_float = []
        self.to_datetime = ['time']


class WP2HIJoinCAT:
    def __init__(self):
        self.name = 'hijoincat'
        self.help_url = "https://www.helcats-fp7.eu/catalogues/wp2_joincat.html"
        self.data_url = 'https://www.helcats-fp7.eu/catalogues/data/HCME_WP2_JOINCAT_A_V01.json'
        self.reference = ""
        self.to_drop = []
        self.to_rename = {'ID' : 'sta',
                          'Linked ID' : 'stb'}
        self.to_str = ['sta', 'stb']
        self.to_int = []
        self.to_float = []
        self.to_datetime = []


class WP3HIGeoCAT:
    def __init__(self):
        self.name = 'higeocat'
        self.help_url = "https://www.helcats-fp7.eu/catalogues/wp3_cat.html"
        self.data_url = 'https://www.helcats-fp7.eu/catalogues/data/HCME_WP3_V06.json'
        self.reference = ""
        self.to_drop = ['SC',
                        'L-N',
                        'L-S',
                        'Quality',
                        'PA-N [deg]',
                        'PA-S [deg]',
                        'Date [UTC]']
        self.to_rename = {'ID' : 'helcats_name'}
        self.to_str = []
        self.to_int = ['PA-fit [deg]',
                       'FP Speed [kms-1]',
                       'FP Speed Err [kms-1]',
                       'FP Phi [deg]',
                       'FP Phi Err [deg]',
                       'FP HEEQ Long [deg]',
                       'FP HEEQ Lat [deg]',
                       'FP Carr Long [deg]', 
                       'SSE Speed [kms-1]',
                       'SSE Speed Err [kms-1]',
                       'SSE Phi [deg]',
                       'SSE Phi Err [deg]',
                       'SSE HEEQ Long [deg]',
                       'SSE HEEQ Lat [deg]',
                       'SSE Carr Long [deg]',
                       'HM Speed [kms-1]',
                       'HM Speed Err [kms-1]',
                       'HM Phi [deg]',
                       'HM Phi Err [deg]',
                       'HM HEEQ Long [deg]',
                       'HM HEEQ Lat [deg]',
                       'HM Carr Long [deg]']
        self.to_float = []
        self.to_datetime = ['FP Launch [UTC]',
                            'SSE Launch [UTC]',
                            'HM Launch [UTC]']
        

class WP3ARRCAT:
    def __init__(self):
        self.name = 'arrcat'
        self.help_url = "https://www.helcats-fp7.eu/catalogues/wp4_arrcat.html"
        self.data_url = 'https://www.helcats-fp7.eu/catalogues/data/HCME_WP4_V6.json'
        self.reference = ""
        self.to_drop = ['SC']
        self.to_rename = {'ID' : 'helcats_name'}
        self.to_str = []
        self.to_int = ['SSE_HEEQ_LONG',
                       'SSE_SPEED',
                       'TARGET_SPEED',
                       'TARGET_PA',
                       'PA_FIT',
                       'PA_N',
                       'PA_S',
                       'PA_CENTER']
        self.to_float = ['TARGET_DELTA',
                         'TARGET_DISTANCE',
                         'TARGET_HEEQ_LAT',
                         'TARGET_HEEQ_LONG']
        self.to_datetime = ['TARGET_ARRIVAL']
        

class WP3KINCAT:
    def __init__(self):
        self.name = 'kincat'
        self.help_url = "https://www.helcats-fp7.eu/catalogues/wp3_kincat.html"
        self.data_url = 'https://www.helcats-fp7.eu/catalogues/data/KINCAT_WP3_V02.json'
        self.reference = ""
        self.to_drop = []
        self.to_rename = {'ID' : 'helcats_name'}
        self.to_str = []
        self.to_int = ['GCS HEEQ Long [deg]',
                       'GCS Carr Long [deg]',
                       'GCS HEEQ Lat [deg]',
                       'GCS Tilt [deg]',
                       'Apex Speed [kms-1]']
        self.to_float = ['GCS Asp. Ratio',
                         'GCS h angle [deg]',
                         'CME Mass [kg]']
        self.to_datetime = ['Pre-even Date [UTC]',
                            'Last COR2 Date [UTC]']
        
        
# class WP4CORHITCAT:
#     def __init__(self):
#         self.name = 'corhitcat'
#         self.help_url = "https://figshare.com/articles/HELCATS_CORHITCAT/4903241"
#         self.data_url = 'Unknown'
#         self.reference = ""
#         self.to_rename = {'HEL_no' : 'helcats_name',
#                           'ICMECAT_ID' : 'icme_name'}
#         self.to_str = []
#         self.to_int = []
#         self.to_float = []
#         self.to_datetime = []


class WP4ICMECAT:
    def __init__(self):
        self.name = 'icmecat'
        self.help_url = "https://www.helcats-fp7.eu/catalogues/wp4_icmecat.html"
        self.data_url = 'https://www.helcats-fp7.eu/catalogues/data/ICME_WP4_V10.json'
        self.reference = ""
        self.to_drop = []
        self.to_rename = {'ICMECAT_ID' : 'icme_name'}
        self.to_str = []
        self.to_int = []
        self.to_float = []
        self.to_datetime = []


class WP4LINKCAT:
    def __init__(self):
        self.name = 'linkcat'
        self.help_url = "https://www.helcats-fp7.eu/catalogues/wp4_cat.html"
        self.data_url = 'https://www.helcats-fp7.eu/catalogues/data/HCME_WP4_V01.json'
        self.reference = ""
        self.to_drop = ['CME_SSE_LONG_HEEQ [deg]',
                        'CME_SSE_LAT_HEEQ [deg]',
                        'CME_SSE_SPEED [kms-1]']
        self.to_rename = {'HICAT_ID' : 'helcats_name',
                          'ICMECAT_ID' : 'icme_name'}
        self.to_str = ['TARGET_NAME',
                       'SC_INSITU',
                       'FLARE_CLASS']
        self.to_int = ['ARRIVAL_DIFFERENCE [hrs]',
                       'CME_TARGET_SPEED [kms-1]',
                       'SC_HELIODISTANCE [AU]',
                       'SC_LONG_HEEQ [deg]',
                       'SC_LAT_HEEQ [deg]',
                       'SOURCE_LONG_HEEQ [deg]',
                       'SOURCE_LAT_HEEQ [deg]', 
                       'MO_BMEAN [nT]',
                       'MO_BSTD [nT]',
                       'MO_BZMEAN [nT]',
                       'MO_BZMIN [nT]',
                       'MO_MVA_AXIS_LONG [deg]',
                       'MO_MVA_AXIS_LAT [deg]',
                       'MO_MVA_RATIO',
                       'GSR_AXIS_LONG [deg]',
                       'GSR_AXIS_LAT [deg]',
                       'GSR_IMPACT [AU]']
        self.to_float = ['VHTX [kms-1]', 
                         'VHTY [kms-1]',
                         'VHTZ [kms-1]']
        self.to_datetime = ['SSE_LAUNCH [UTC]',
                            'TARGET_ARRIVAL [UTC]',
                            'ICME_START_TIME [UTC]',
                            'FLARE_START_TIME [UTC]',
                            'FLARE_END_TIME [UTC]',
                            'FLARE_PEAK_TIME [UTC]',
                            'MO_START_TIME [UTC]',
                            'MO_END_TIME [UTC]',
                            'GSR_START_TIME [UTC]',
                            'GSR_START_TIME [UTC].1']

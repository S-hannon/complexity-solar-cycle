"""Script to load HELCATS catalogue.

Example usage:
# init catalogue
hc = HELCATSCatalogue()

# print list of catalogues
hc.cats()

# find out about a catalogue
hc.cat_info("hicat")

# search for columns containing string
hc.search("speed")

# find out about a column
hc.col_info("SSE_SPEED")

# load catalogue as pandas DataFrame
df = hc.load("hicat")

# get vals from col for cmes with helcats names in list
vals = hc.get_col_vals("SSE_SPEED", [helcats_name0, helcats_name1, ...])
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime
from misc import get_project_dirs


class HELCATSCatalogue:
    def __init__(self):
        #######################################################################
        # Change this to dir where you want to store data!!
        data_loc = get_project_dirs()['data']
        #######################################################################
        # find HELCATS data folder, make it if it doesn't exist
        self.folder = os.path.join(data_loc, 'HELCATS')
        if os.path.exists(self.folder) == False:
            os.makedir(self.folder)
        # load the catalogues
        cat_names = ['hicat', 'arrcat', 'corhitcat', 'higeocat',
                     'hijoincat', 'icmecat', 'kincat', 'linkcat']
        cat_instances = [HICat(), ArrCat(), CorHitCat(), HIGeoCat(),
                         HIJoinCat(), ICMECat(), KinCat(), LinkCat()]
        self.cat_instances = dict({})
        for n, name in enumerate(cat_names):
            self.cat_instances[name] = cat_instances[n]
            
            
    def cats(self):
        """Prints names of included catalogues.
        """
        print(['hicat', 'arrcat', 'corhitcat', 'higeocat',
               'hijoincat', 'icmecat', 'kincat', 'linkcat'])
        
        
    def search(self, value):
        """Searches for column names containing the value string. Case
        insensitive.
        """
        file_path = os.path.join(self.folder, 'helcats_guide'+ '.csv')
        df = pd.read_csv(file_path, header=0)
        for col in df.col.values:
            if value.lower() in col.lower():
                print(col)
                
                
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
            
            
    def check_cat_name(self, cat_name):
        """Checks cat_name is a valid HELCATS catalogue name.
        """
        if cat_name not in self.cat_instances:
            raise ValueError("Requested cat not found.")
            
            
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
        
        
    def load(self, cat_name):
        """Returns HELCATS catalogue cat_name as pd df.
        """
        cat_name = cat_name.lower()
        self.check_cat_name(cat_name)
        cat = self.cat_instances[cat_name]
        file_path = os.path.join(self.folder, cat.name + '.csv')
        # if cat doesn't exist, download and save as nice .csv
        if os.path.exists(file_path) == False:
            df = pd.read_json(cat.data_url, orient='split')
            df = df.rename(columns=cat.to_rename)
            # save as .csv file as easy to read
            df.to_csv(file_path, sep=",", index=False)        
        # now load the cat from csv
        df = pd.read_csv(file_path)
        # converters        
        for col in cat.to_str:
            df[col] = df[col].astype(str)   
        for col in cat.to_int:
            df[col] = df[col].astype(int)
        for col in cat.to_float:
            df[col] = df[col].astype(float)
        for col in cat.to_datetime:
            df[col] = pd.to_datetime(df[col], dayfirst=True)
        # add craft column
        if cat_name != 'hijoincat':
            craft_list = []
            for cme in df.helcats_name:
                if cme[5] == 'A':
                    craft = 'sta'
                elif cme[5] == 'B':
                    craft = 'stb'
                craft_list.append(craft)
            df['craft'] = pd.Series(craft_list, index=df.index)
        return df
    
    
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
        
        
    def get_col_vals(self, col, helcats_name_list):
        """Returns the col values for the cmes in helcats_name_list.
        """
        if not isinstance(helcats_name_list, list):
            raise ValueError("helcats_name_list needs to be a list")
        col = col.lower()
        cat = self.find_col(col)
        df = self.load(cat)
        col_vals = []
        # Loop over items in df
        for helcats_name in helcats_name_list:
            row = df[df['helcats_name'] == helcats_name]
            # Save the value if it exists, or np.NaN if it doesn't
            if row.empty() == True:
                col_vals.append(np.NaN)
            elif len(row) > 1:
                raise ValueError("cat has more than one value for this CME")
            else:
                col_vals.append(row[col][0])
        return col_vals


class HICat:
    def __init__(self):
        self.name = 'hicat'
        self.help_url = "https://www.helcats-fp7.eu/catalogues/wp2_cat.html"
        self.data_url = 'https://www.helcats-fp7.eu/catalogues/data/HCME_WP2_V06.json'
        self.reference = ""
        self.to_rename = {'ID' : 'helcats_name', 'Date [UTC]' : 'time'}
        self.to_str = ['SC', 'L:N', 'L:S', 'Quality']
        self.to_int = ['PA-N [deg]', 'PA-S [deg]']
        self.to_float = []
        self.to_datetime = ['time']


class ArrCat:
    def __init__(self):
        self.name = 'arrcat'
        self.help_url = "https://www.helcats-fp7.eu/catalogues/wp4_arrcat.html"
        self.data_url = 'https://www.helcats-fp7.eu/catalogues/data/HCME_WP4_V6.json'
        self.reference = ""
        self.to_rename = {'ID' : 'helcats_name'}
        self.to_str = ['SC', 'TARGET_NAME']
        self.to_int = ['SSE_HEEQ_LONG', 'SSE_SPEED', 'TARGET_SPEED',
                       'TARGET_PA', 'PA_FIT', 'PA_N', 'PA_S', 'PA_CENTER']
        self.to_float = ['TARGET_DELTA', 'TARGET_DISTANCE',
                         'TARGET_HEEQ_LAT', 'TARGET_HEEQ_LONG']
        self.to_datetime = ['SSE_START_TIME', 'SSE_TARGET_TIME']


class CorHitCat:
    def __init__(self):
        self.name = 'corhitcat'
        self.help_url = "https://figshare.com/articles/HELCATS_CORHITCAT/4903241"
        self.data_url = os.path.join(get_project_dirs()['data'], 'HELCATS',
                                     'final_corhitcat_12htol.json')
        self.reference = ""
        self.to_rename = {'HEL_no' : 'helcats_name',
                          'ICMECAT_ID' : 'icme_name'}
        self.to_str = []
        self.to_int = []
        self.to_float = []
        self.to_datetime = []
    

class HIGeoCat:
    def __init__(self):
        self.name = 'higeocat'
        self.help_url = "https://www.helcats-fp7.eu/catalogues/wp3_cat.html"
        self.data_url = 'https://www.helcats-fp7.eu/catalogues/data/HCME_WP3_V06.json'
        self.reference = ""
        self.to_rename = {'ID' : 'helcats_name'}
        self.to_str = ['SC', 'L-N', 'L-S', 'Quality']
        self.to_int = ['PA-N [deg]', 'PA-S [deg]', 'PA-fit [deg]',
                       'FP Speed [kms-1]', 'FP Speed Err [kms-1]',
                       'FP Phi [deg]', 'FP Phi Err [deg]',
                       'FP HEEQ Long [deg]', 'FP HEEQ Lat [deg]',
                       'FP Carr Long [deg]',  'SSE Speed [kms-1]',
                       'SSE Speed Err [kms-1]', 'SSE Phi [deg]',
                       'SSE Phi Err [deg]', 'SSE HEEQ Long [deg]',
                       'SSE HEEQ Lat [deg]', 'SSE Carr Long [deg]',
                       'HM Speed [kms-1]', 'HM Speed Err [kms-1]',
                       'HM Phi [deg]', 'HM Phi Err [deg]',
                       'HM HEEQ Long [deg]', 'HM HEEQ Lat [deg]',
                       'HM Carr Long [deg]']
        self.to_float = []
        self.to_datetime = ['Date [UTC]', 'FP Launch [UTC]',
                            'SSE Launch [UTC]', 'HM Launch [UTC]']


class HIJoinCat:
    def __init__(self):
        self.name = 'hijoincat'
        self.help_url = "https://www.helcats-fp7.eu/catalogues/wp2_joincat.html"
        self.data_url = 'https://www.helcats-fp7.eu/catalogues/data/HCME_WP2_JOINCAT_A_V01.json'
        self.reference = ""
        self.to_rename = {'ID' : 'sta', 'Linked ID' : 'stb'}
        self.to_str = ['sta', 'stb']
        self.to_int = []
        self.to_float = []
        self.to_datetime = []


class ICMECat:
    def __init__(self):
        self.name = 'icmecat'
        self.help_url = "https://www.helcats-fp7.eu/catalogues/wp4_icmecat.html"
        self.data_url = 'https://www.helcats-fp7.eu/catalogues/data/ICME_WP4_V10.json'
        self.reference = ""
        self.to_rename = {'ICMECAT_ID' : 'icme_name'}
        self.to_str = []
        self.to_int = []
        self.to_float = []
        self.to_datetime = []


class KinCat:
    def __init__(self):
        self.name = 'kincat'
        self.help_url = "https://www.helcats-fp7.eu/catalogues/wp3_kincat.html"
        self.data_url = 'https://www.helcats-fp7.eu/catalogues/data/KINCAT_WP3_V02.json'
        self.reference = ""
        self.to_rename = {'ID' : 'helcats_name'}
        self.to_str = []
        self.to_int = ['GCS HEEQ Long [deg]', 'GCS Carr Long [deg]',
                       'GCS HEEQ Lat [deg]', 'GCS Tilt [deg]',
                       'Apex Speed [kms-1]']
        self.to_float = ['GCS Asp. Ratio', 'GCS h angle [deg]',
                         'CME Mass [kg]']
        self.to_datetime = ['Pre-even Date [UTC]', 'Last COR2 Date [UTC]']


class LinkCat:
    def __init__(self):
        self.name = 'linkcat'
        self.help_url = "https://www.helcats-fp7.eu/catalogues/wp4_cat.html"
        self.data_url = 'https://www.helcats-fp7.eu/catalogues/data/HCME_WP4_V01.json'
        self.reference = ""
        self.to_rename = {'HICAT_ID' : 'helcats_name',
                          'ICMECAT_ID' : 'icme_name'}
        self.to_str = ['TARGET_NAME', 'SC_INSITU', 'FLARE_CLASS']
        self.to_int = ['ARRIVAL_DIFFERENCE [hrs]', 'SC_HELIODISTANCE [AU]',
                       'SC_LONG_HEEQ [deg]', 'SC_LAT_HEEQ [deg]',
                       'SOURCE_TYPE', 'SOURCE_LONG_HEEQ [deg]',
                       'SOURCE_LAT_HEEQ [deg]', 'CME_SSE_LONG_HEEQ [deg]',
                       'CME_SSE_LAT_HEEQ [deg]', 'CME_SSE_SPEED [kms-1]',
                       'CME_TARGET_SPEED [kms-1]', 'MO_BMEAN [nT]',
                       'MO_BSTD [nT]', 'MO_BZMEAN [nT]', 'MO_BZMIN [nT]',
                       'MO_MVA_AXIS_LONG [deg]', 'MO_MVA_AXIS_LAT [deg]',
                       'MO_MVA_RATIO', 'GSR_AXIS_LONG [deg]',
                       'GSR_AXIS_LAT [deg]', 'GSR_IMPACT [AU]']
        self.to_float = ['VHTX [kms-1]', 'VHTY [kms-1]', 'VHTZ [kms-1]']
        self.to_datetime = ['SSE_LAUNCH [UTC]', 'TARGET_ARRIVAL [UTC]',
                            'ICME_START_TIME [UTC]', 'FLARE_START_TIME [UTC]',
                            'FLARE_END_TIME [UTC]', 'FLARE_PEAK_TIME [UTC]',
                            'MO_START_TIME [UTC]', 'MO_END_TIME [UTC]',
                            'GSR_START_TIME [UTC]', 'GSR_START_TIME [UTC].1']

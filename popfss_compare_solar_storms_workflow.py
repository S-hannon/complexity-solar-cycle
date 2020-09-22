import os
import sys
import glob
import pandas as pd
import numpy as np
import simplejson as json
import hi_processing.images as hip
import matplotlib as mpl
import matplotlib.pyplot as plt
import PIL.Image as Image
import scipy.stats as sps
from datetime import datetime, timedelta
import misc
from data_helcats import HELCATS
from data_stereo_hi import STEREOHI


class CompareSolarStormsWorkflow:
    def __init__(self, data_loc, fig_loc):
        # set dirs
        for loc in [data_loc, fig_loc]:
            if not os.path.exists(loc):
                raise ValueError("path " + loc + " doesn't exist")
        self.data_loc = data_loc
        self.fig_loc = os.path.join(fig_loc, 'POPFSS')
        self.root = os.path.join(data_loc, 'POPFSS')
        for loc in [self.root, self.fig_loc]:
            if not os.path.exists(loc):
                os.mkdir(loc)
        # Zooniverse details
        self.workflow_name = 'Compare Solar Storms'
        self.workflow_id = 6496
        # project data comes saved with '-' instead of ' ' in the name
        self.project_name = 'Protect our Planet from Solar Storms'
        self.project_save_name = 'protect-our-planet-from-solar-storms'
        self.project_id = 6480
        # details of the different project phases, to split classifications
        self.diff = dict({'phase' : 1,
                          'workflow_version' : 19.32,
                          'subject_id_min' : 21571858,
                          'subject_id_max' : 27198377,
                          'classification_id_min' : 107482001,
                          'classification_id_max' : 134398497})
        self.diff_be = dict({'phase' : 2,
                             'workflow_version' : 19.32,
                             'subject_id_min' : 34364886,
                             'subject_id_max' : 34387958,
                             'classification_id_min' : 172529512,
                             'classification_id_max' : 240608296})
        self.norm = dict({'phase' : 3,
                          'workflow_version' : 19.32,
                          'subject_id_min' : 44304478,
                          'subject_id_max' : 46571066,
                          'classification_id_min' : 251146634})


    ############### setting up the project
        
        
    def make_assets(self, img_type, camera='hi1', background_type=1):
        """
        Function to loop over the HELCATS CMEs, find all relevant HI1A and HI1B 1-day background images, and produce
        plain, differenced and relative difference images.
        """
        helcats = HELCATS(self.data_loc)
        hi_data = STEREOHI(self.data_loc)
        cme_list = helcats.find_cmes(te_track=True)
        for n, cme in enumerate(cme_list):
            print(cme)
            craft, time = helcats.get_cme_details(cme)
            start, mid, end, mid_el = helcats.get_te_track_times(cme,
                                                                 camera=camera)
            hi_data.make_img(cme, mid, craft, 'POPFSS', img_type,
                             camera=camera, background_type=background_type)
    
    
    def find_comps(self, n, cycles='all', rounds=1, rn=0):
        """Finds pairwise comparisons to use in the manifest file.
        :param: n: number of objects to compare
        :param: cycles: number of cycles with n comparisons, must be between 1 and 
            ceil(n/2)
        :param: rounds: number of files to split total comparisons over
        :param: rn: current round number, adds an offset to the cycle numbers run
        :returns: lists containing indexes of each asset to compare
        """ 
        # calculate maximum values
        max_cycles = np.int(np.ceil(n/2))-1
        max_ccs = n*max_cycles
        max_comps = np.int((n/2)*(n-1))  
        # if no number of cycles chosen, set at maximum
        if cycles == 'all':
            cycles = max_cycles
        if cycles != np.int(cycles):
            raise ValueError("number of cycles must be an integer")
        if cycles < 1:
            raise ValueError("must be at least one cycle")
        if cycles > max_cycles:
            raise ValueError("number of cycles cannot be greater than ceil(n/2)-1")
        if (cycles * rounds) > max_cycles:
            raise ValueError("cycles*rounds must be less than %s" %(max_cycles))
        if rn > rounds:
            raise ValueError("round number cannot exceed number of rounds")
        # build nxn matrix
        matrix = np.zeros((n,n), dtype=int)
        spacing = np.int(np.floor(max_cycles/cycles))
        cycle_nos = np.arange(1, np.int(np.ceil(n/2)), spacing)[0:cycles]
        # change dependant on round number
        for c in range(len(cycle_nos)):
            cycle_nos[c] = cycle_nos[c] + rn - 1
        # each s is a loop with n comparisons
        # starts at diagonal under 1, as the 0 diagonal is the origin
        for s in cycle_nos:
            print(s)
            # change 0s to 1s for comparisons in this loop
            for i in range(0, n):
                j = np.mod(s+i, n)
                # Check this hasn't been compared already...
                if matrix[j, i] == 0:
                    # Do this comparison
                    matrix[i, j] = 1
        print('cycles run: %s out of %s' %(cycles, max_cycles))
        print('comparisons generated: %s out of %s' %(np.sum(matrix), max_ccs))
        m = self.matrix_to_list(matrix)
        return m
    
    
    def matrix_to_list(self, matrix):
        """
        Takes a matrix and returns a list of the rows/columns of the non-zero 
        values.
        """
        first = []
        second = []
        n = len(matrix)
        # loop over rows
        for i in range(0, n):
            # loop over values in row
            for j in range(0, n):
                if matrix[i, j] == 1:
                    first.append(i)
                    second.append(j)
        return first, second
            
    
    def make_manifest(self, img_type, cycles=16, m_files=30):
        """
        This function produces the manifest to serve the ssw assets. This has the format of a CSV file with:
        asset_name,file1,file2,...fileN.
        Asset names will be given the form of sswN_helcatsM_craft_type_t1_t3, where t1 and t3 correspond to the times of the
        first and third image in sets of three.
        This works by searching the 'out_data/comp_assets' folder for assets and
        creating a manifest file for these assets. 
        :param: m_files: number of manifest files to split comparisons into
        :return: Outputs a "manifest.csv" file in the event/craft/type directory of these images, or multiple files
        """
        # Want list of all images from both craft
        sta_data_dir = os.path.join(self.data_loc, 'STEREO_HI', 'Images',
                                    'POPFSS', img_type, 'sta')
        stb_data_dir = os.path.join(self.data_loc, 'STEREO_HI', 'Images',
                                    'POPFSS', img_type, 'stb')
        sta_files = glob.glob(os.path.join(sta_data_dir, '*'))
        stb_files = glob.glob(os.path.join(stb_data_dir, '*'))
        # get only filename not full path, exclude extension
        sta_files = [os.path.basename(f) for f in sta_files]
        stb_files = [os.path.basename(f) for f in stb_files]
        images = np.append(sta_files, stb_files)
        images.sort()
        print("found %s images, generating comparisons..." %(len(images)))
        # Create manifest files
        for r in range(m_files):
            # Make the manifest file
            manifest_path = os.path.join(self.root, 'manifest'+str(r+1)+'.csv')
            with open(manifest_path, 'w') as manifest:
                # Add in manifest headers
                manifest.write("subject_id,asset_0,asset_1\n")
                # Get comparisons list for this manifest file
                comps = self.find_comps(len(images), cycles=cycles, rounds=m_files, rn=r+1)
                # returns lists of left and right images to compare
                # Write comparisons list into correct columns
                # loop over each comparison
                i = 0
                # give each comparison a subject id
                sub_id = 0
                while i < len(comps[0]):
                    manifest_elements = [str(sub_id), images[comps[0][i]], images[comps[1][i]]]
                    i = i + 1
                    sub_id += 1
                    # Write out as comma sep list
                    manifest.write(",".join(manifest_elements) + "\n")
 
    
    def get_helcats_names(self, image_list):
        """returns HELCATS name string given image name
        e.g.ssw_067_helcats_HCME_B__20131128_02_stb_diff_20131129_005001.jpg
        returns HCME_B__20131128_02
        """
        helcats_name_list = []
        for img in image_list:
            parts = img.split('_')
            hn = parts[3] + '_' + parts[4] + '__' + parts[6] + '_' + parts[7]
            helcats_name_list.append(hn)
        return helcats_name_list
    
    
    def analyse_manifest(self, manifest_name):
        df = pd.read_csv(os.path.join(self.root, manifest_name))
        hc = HELCATS(self.data_loc)
        # add columns for helcats names, dates and craft of each image
        for n, side in enumerate(['left', 'right']):
            df[side + '_helcats_name'] = self.get_helcats_names(df['asset_' + str(n)])
            craft_list, time_list = hc.get_cme_details_list(df[side + '_helcats_name'])
            df[side + '_craft'] = pd.Series(craft_list, index=df.index)
            df[side + '_time'] = pd.Series(time_list, index=df.index)
        # CME occurence as left or right image
        l_occurences = []
        r_occurences = []
        for cme in np.unique(df['left_helcats_name']):
            l = df[df['left_helcats_name'] == cme]
            r = df[df['right_helcats_name'] == cme]
            l_occurences.append(len(l))
            r_occurences.append(len(r))
        plt.figure(figsize=[9, 9])
        plt.scatter(l_occurences, r_occurences)
        plt.xlabel("# times CME shown as left image", fontsize=16)
        plt.ylabel("# times CME shown as right image", fontsize=16)
        totals = [sum(x) for x in zip(l_occurences, r_occurences)]
        print("Each CME is compared to between %s and %s different CMEs." %(np.min(totals), np.max(totals)))
        # times of left and right images compared
        plt.figure(figsize=[9, 9])
        plt.scatter(df.left_time, df.right_time)
        plt.xlabel("Time of left image", fontsize=16)
        plt.ylabel("Time of right image", fontsize=16)
               
        
    ############### process project data

        
    def load_classifications(self, img_type):
        """Loads in the project-name-classifications.csv file from the
        Zooniverse project.
        """
        converters = dict(classification_id=int,
                          user_name=str,
                          user_id=str,
                          user_ip=str,
                          workflow_id=int,
                          workflow_name=str,
                          workflow_version=float,
                          metadata=json.loads,
                          annotations=json.loads,
                          subject_data=json.loads,
                          subject_ids=int)
        clas_path = os.path.join(self.root,
                                 self.project_save_name + '-classifications.csv')
        cdf = pd.read_csv(clas_path, converters=converters) 
        # get classifications for this workflow only
        cdf = cdf[cdf['workflow_name'] == self.workflow_name]
        cdf = cdf[cdf['workflow_id'] == self.workflow_id]
        # get classifications for this img_type only
        cdf = cdf[cdf['workflow_version'] == getattr(self, img_type)['workflow_version']]
        cdf = cdf[cdf['subject_ids'] >= getattr(self, img_type)['subject_id_min']]
        cdf = cdf[cdf['subject_ids'] <= getattr(self, img_type)['subject_id_max']]
        return cdf


    def process_classifications(self, img_type):
        # subjects = self.load_subjects() 
        classifications = self.load_classifications(img_type)
        # Initialise output lists
        df = pd.DataFrame(columns={'subject_id', 'left_subject',
                                   'right_subject', 'left_wins', 'right_wins',
                                   'winner'})
        for s in classifications['subject_ids'].unique():
            left_wins = 0
            right_wins = 0
            winner = 'draw'
            c_subset = classifications[classifications.subject_ids == s]
            for n, i in enumerate(c_subset.index):
                # get the names of the left and right images
                if n == 0:
                    left_subject = c_subset.subject_data[i][str(s)]['asset_0'].replace(' ', '')
                    right_subject = c_subset.subject_data[i][str(s)]['asset_1'].replace(' ', '')
                # Add result to left or right score
                result = c_subset.annotations[i][0]['value']
                if result == "Image on the left":
                    left_wins = left_wins + 1
                elif result == "Image on the right":
                    right_wins = right_wins + 1
            if left_wins > right_wins:
                winner = 'left'
            elif right_wins > left_wins:
                winner = 'right'
            df = df.append({'subject_id' : s,
                            'left_subject' : left_subject,
                            'right_subject' : right_subject,
                            'left_wins' : left_wins,
                            'right_wins' : right_wins,
                            'winner' : winner}, ignore_index=True)
        # now add extra useful columns
        for side in ['left', 'right']:
            helcats_name_list = self.get_helcats_names(df[side + '_subject'])
            df[side + '_helcats_name'] = helcats_name_list
            hc = HELCATS(self.data_loc)
            craft_list, time_list = hc.get_cme_details_list(df[side + '_helcats_name'])
            df[side + '_craft'] = craft_list
            df[side + '_time'] = time_list
        df['total_votes'] = df['left_wins'] + df['right_wins']
        # for some reason, some of the comparisons are listed twice
        # this is WEIRD and I don't know how it happened
        # anyway, this code sorts it out:
        df['both_names'] = df['left_subject'].astype(str) + df['right_subject'].astype(str)
        for j in np.unique(df['both_names']):
            dfj = df[df['both_names'] == j]
            if len(dfj) > 1:
                df.loc[dfj.index.values[0], ['left_wins']] = sum(dfj.left_wins)
                df.loc[dfj.index.values[0], ['right_wins']] = sum(dfj.right_wins)
                df.loc[dfj.index.values[0], ['total_votes']] = sum(dfj.total_votes)
                for n in range(1, len(dfj)):
                    df = df.drop(dfj.index.values[n])
        df = df.drop(columns={'both_names'})
        name = os.path.join(self.root,
                            'popfss_comparison_results_' + img_type + '.csv')
        df.to_csv(name, sep=',')
        self.make_sta_stb_only_files(img_type)


    def make_sta_stb_only_files(self, img_type):
        """makes separate files with only comparisons between STA and STA CMEs,
        and STB and STB CMEs respectively.
        """
        name = 'popfss_comparison_results_' + img_type + '.csv'
        df = pd.read_csv(os.path.join(self.root, name))
        for craft in ['sta', 'stb']:
            dfc = df[df.left_craft == craft]
            dfc = dfc[dfc.right_craft == craft]
            name = 'popfss_comparison_results_'+ img_type + '_' + craft + '.csv'
            dfc.to_csv(os.path.join(self.root, name))
            
            
    def load_data(self, img_type):
        name = 'popfss_comparison_results_' + img_type + '.csv'
        df = pd.read_csv(os.path.join(self.root, name))
        df['left_time'] = pd.to_datetime(df['left_time'], format="%Y-%m-%d")
        df['right_time'] = pd.to_datetime(df['right_time'], format="%Y-%m-%d")
        return df
    

    ############### investigate classification data
    
    
    def classifications_by_user(self, img_type):
        cdf = self.load_classifications(img_type)
        user_cs = []
        user_cs_anon = []
        for user in cdf['user_id'].unique():
            if user != "":
                user_cs.append(len(cdf[cdf['user_id'] == user]))
        # get classifications from users not logged in
        cdf_anon = cdf[cdf['user_id'] == ""]
        for user_anon in cdf_anon['user_name'].unique():
            user_cs_anon.append(len(cdf_anon[cdf_anon['user_name'] == user_anon]))
        user_classifications = user_cs + user_cs_anon
        plt.figure(figsize=[14, 9])
        n, bins, patches = plt.hist(user_classifications, bins=100,
                                    range=(0, 500))
        plt.xlabel('Number of classifications', fontsize=16)
        plt.ylabel('Frequency', fontsize=16)
        plt.savefig(os.path.join(self.fig_loc,
                                 img_type + ' classifications by user'))
        n = 0
        one = 0
        for i in user_classifications:
            if i == 1:
                one += 1
            elif i > 500:
                n += 1
        user_classifications.sort(reverse=True)
        # NB the first user in this list is '' corresponding to the total of all
        # the users who were not logged in at the time
        print('########## classifications summary ##########')
        print('in total', len(cdf), 'classifications were made')
        print(len(cdf) - sum(user_cs_anon),
              'classifications were made by', len(user_cs),
              'users logged into their Zooniverse accounts')
        print(sum(user_cs_anon),
              'classifications were completed by', len(user_cs_anon),
              'users not logged in')
        print(n, 'users completed more than 500 classifications')
        print(one, 'users completed just one classification')
        print('the top 10 users completed', user_classifications[0:10],
              'classifications')
        

    def plot_wins_per_year(self, img_type):
        df = self.load_data(img_type)
        # remove comparisons where both are from the same year
        left_years = []
        right_years = []
        for i in df.index:
            left_years.append(df['left_time'][i].year)
            right_years.append(df['right_time'][i].year)
        df['left_year'] = left_years
        df['right_year'] = right_years
        df = df[df['left_year'] != df['right_year']]
        # find percentage wins per year
        years = []
        per_a = []
        per_b = []
        for year in range(np.min(left_years + right_years),
                          np.max(left_years + right_years) + 1):
            wins_a = 0
            wins_b = 0
            year_total = 0
            for side in ['left', 'right']:
                dfy = df[df[side + '_year'] == year]
                dfys = dfy[dfy.winner == side]
                wins_a = wins_a + len(dfys[dfys[side + '_craft'] == 'sta'])
                wins_b = wins_b + len(dfys[dfys[side + '_craft'] == 'stb'])
                year_total = year_total + len(dfy)
            years.append(year)
            per_a.append(100 * (wins_a / year_total))
            per_b.append(100 * (wins_b / year_total))
        # make the plot
        plt.figure(figsize=[8, 5])
        plt.bar(years, per_a, color='pink', label='STEREO-A')
        plt.bar(years, per_b, color='lightskyblue', bottom=per_a,
                label='STEREO-B')
        plt.xlabel('Year')
        plt.ylabel('Percentage')
        plt.xticks(range(2008, 2017, 1))
        plt.legend(loc=0)
        plt.savefig(os.path.join(self.fig_loc, 'wins per year ' + img_type)) 


    def __find_binomial_percentages(self, df):
        df['total_votes'] = df['left_wins'] + df['right_wins']
        df = df[df.total_votes == 12]
        names = []
        actual_percentages = []
        binomial_percentages = []
        for i in range(7):
            result = str(i) + ' vs ' + str(12 - i)
            names.append(result)
            if i != 6:
                n = len(df[df.left_wins == i]) + len(df[df.right_wins == i])
                bp = sps.binom.pmf(i, 12, 0.5) + sps.binom.pmf(12 - i, 12, 0.5) 
            else:
                n = len(df[df.left_wins == i])
                bp = sps.binom.pmf(i, 12, 0.5)
            actual = (n / len(df)) * 100
            binomial = bp * 100
            actual_percentages.append(actual)
            binomial_percentages.append(binomial)
            print('########## random vs actual percentages ##########')
            print((' ').join(['result:', result, 'actual:',
                              "{:.2f}".format(actual), '%', 'binomial:',
                              "{:.2f}".format(binomial), '%']))
        return names, actual_percentages, binomial_percentages


    def plot_paired_comparison_results(self, img_type):
        df = self.load_data(img_type)
        names, actual_percentages, binomial_percentages = self.__find_binomial_percentages(df)
        # make bar plot
        plt.figure(figsize=[14, 9])    
        x = np.arange(len(names))
        w = 0.3
        plt.bar(x - w/2 - 0.02, actual_percentages, width=w, color='cornflowerblue', align='center',
                label='Actual')
        plt.bar(x + w/2 + 0.02, binomial_percentages, width=w, color='midnightblue',
                align='center', label='If choosing randomly')
        plt.xticks(x, names, fontsize=16)
        labels = []
        yticks = range(0, 41, 5)
        for j in yticks:
            labels.append(str(j) + '%')
        plt.yticks(yticks, labels, fontsize=16)
        plt.xlabel("Results (number of people choosing each image)", fontsize=16)
        plt.ylabel("Comparisons with this result", fontsize=16)
        plt.legend(fontsize=16, frameon=False)
    
    
    def left_vs_right_bias(self, img_type):
        df = self.load_data(img_type)
        df = df[df.total_votes == 12]
        # total left and right wins
        print('########## left vs right image bias? ##########')
        print("A 'win' occurs when 7 or more participants out of 12 choose the specified image.")
        left = len(df[df.left_wins > 6])
        right = len(df[df.left_wins < 6])
        draw = len(df[df.left_wins == 6])
        leftp = "(" + "{:.2f}".format((left / len(df)) * 100) + "%)"
        rightp = "(" + "{:.2f}".format((right / len(df)) * 100) + "%)"
        drawp = "(" + "{:.2f}".format((draw / len(df)) * 100) + "%)"
        print("The left image won", left, 'times', leftp)
        print("The right image won", right, 'times', rightp)
        print("Both images drew", draw, 'times', drawp)
        
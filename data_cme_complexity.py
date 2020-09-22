import os
import pandas as pd


class CMEComplexity:
    def __init__(self, data_loc):
        self.data_loc = data_loc
    def __get_helcats_names(self, image_list):
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
    def load(self, img_type):
        # Read in the model parameters
        df = pd.read_csv(os.path.join(self.data_loc, 'POPFSS',
                                      'popfss_model_fit_r_' + img_type + '.csv'))
        helcats_name = []
        complexity = []
        rank = []
        # Order storms by complexity
        df = df.sort_values(by='x')
        # Loop over the storms
        for n, i in enumerate(df.index):
            helcats_name.append(self.__get_helcats_names([df['Unnamed: 0'][i]])[0])
            complexity.append(df['x'][i])
            rank.append(n)
        df = pd.DataFrame({'helcats_name' : helcats_name,
                           'complexity' : complexity,
                           'rank' : rank})
        return df

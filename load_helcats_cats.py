import os
import pandas as pd


def load_helcats_cats(name):
    filepath = os.path.join('N:\Documents\Datasets\HELCATS_cats', str(name)+'.csv')
    cat = pd.read_csv(filepath)   
    return cat


#if __name__ == '__main__':
#    hicat = load_helcats_cats('hicat')
#    higeocat = load_helcats_cats('higeocat')
#    hijoincat = load_helcats_cats('hijoincat')
#    kincat = load_helcats_cats('kincat')
#    linkcat = load_helcats_cats('linkcat')
#    
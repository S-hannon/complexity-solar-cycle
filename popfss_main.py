from __future__ import division
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import popfss_asset_production as ap  # Code to make the CME images using hard drive
import popfss_processing as pc  # Code to read the Zooniverse classifcation data (to read into R)
import popfss_data_extraction as de  # Code to get the complexity data & manipulate after model fitting in R
import popfss_analysis as an  # All other plotting code
import popfss_image_processing as ip


# NB before running code, the 'project_directories.txt' file should be updated
# with correct locations and names.


def make_assets():
    """Creates assets from HI hard drive, and saves into 'out_data' folder.
    Requires:
        - access to HI hard drive
        - asset_production_tools and hi_processing scripts
        - python modules: glob, os, numpy, pandas, matplotlib, PIL.Image,
          datetime, scipy and sunpy
    """
    ap.make_output_directory_structure_comp()
    ap.make_ssw_comp_index_assets()
    ap.make_manifest_comp()
    

def process_classification_data(suffix=""):
    """Reads in classification and subject data file from 'data' folder, and
    converts to an easy to read csv file with the results for each comparison,
    saved into 'out_data' folder.
    Requires:
        - glob, os, numpy, pandas and simplejson modules
    """
    pc.match_classifications_to_subjects(suffix=suffix)


def read_model_output(tag=""):
    """Reads in model parameters from R, turns into pd df.
    """
    params = de.get_model_fit_from_r(tag=tag)
    df = de.extract_data(params)
    return df
    

def show_ranking():
    """Creates plots and animations to show all the storms in order of
    complexity.
    """
    df = read_model_output()
    an.animate_ranking(df)


def complexity_over_time(tag=""):
    """Plots complexity... over time.
    """
    # Get complexity data
    df = read_model_output(tag=tag)
    an.plot_vs_complexity(df, 'time', 'Year', corr=[datetime(2014,01,01),-3])
    an.plot_x_vs_y(df, 'time', 'rank', 'Year', 'Ranking Place', corr=False)
    an.plot_time_vs_complexity(df, means=['independent', 12])
    an.plot_time_vs_complexity(df, means=['running', 3])
    an.sunspot_means_plot(df)
    an.means_points_and_sunspots_plot(df, err='std')
    for means in ['independent', 'running']:
        for t in [3, 6, 12]:
            an.means_points_and_sunspots(df, t, means='independent')
    an.complexity_histogram(df)
    an.complexity_histogram_years(df)
    an.complexity_histogram_split(df, 2010)


def fit_a_b_separately():
    """Read in model output from fitting on STA and only STB CMEs separately,
    and plot the resulting complexity time data.
    """
    params_sta = de.get_model_fit_from_r(tag='_sta')
    params_stb = de.get_model_fit_from_r(tag='_stb')
    dfa = de.extract_data(params_sta)
    dfb = de.extract_data(params_stb)
    an.complexity_time_plot_means(dfa, craft='sta')
    an.complexity_time_plot_means(dfb, craft='stb')


def paired_comparison_data():
    df = read_model_output(tag="")
    dfpc = de.load_paired_comparison_results(df, tag="")
    # an.plot_wins_per_year(dfpc)
    return dfpc


def compare_a_b():
    """Find CMEs observed by both spacecraft and investigate.
    """
    # Get data
    df = read_model_output()
    df = de.add_helcats_to_df(df, 'SSE Phi [deg]', 'higeocat')
    # Make plot
    an.plot_a_vs_b(df, 'complexity', 'Relative Complexity')
    an.plot_a_b_diff_vs_time(df, 'complexity', 'Relative Complexity')
    an.plot_a_b_diff_vs_a_b_diff(df, 'SSE Phi [deg]', 'complexity', 'Phi', 'Relative Complexity')


def width_plots():
    """Looks at the relationship between complexity and angular width.
    """
    # Get data
    df = read_model_output()
    df = de.add_width_to_df(df)
    # Make plots
    # an.plot_vs_complexity(df, 'width', 'Angular Width', corr=[90,-3])
    # # Looking at width intervals
    b1, b2, b3 = df.width.quantile([0.25, 0.5, 0.75])
    g1 = df[df.width < b1]
    g2 = df[(df.width > b1) & (df.width < b2)]
    g3 = df[(df.width > b2) & (df.width < b3)]
    g4 = df[df.width > b3]
    titles = ['Angular Width < $%s\degree$'%(b1),
              '$%s\degree$ < Angular Width < $%s\degree$'%(b1, b2),
              '$%s\degree$ < Angular Width < $%s\degree$'%(b2, b3),
              'Angular Width > $%s\degree$'%(b3)]
    an.plot_complexity_time_four_groups([g1, g2, g3, g4], titles)


def process_images(tag=""):
    """Loops over images of all CMEs in folder, does various things like
    counting pixels.
    :param: tag: string, suffix of file to determine which set of images to use
    """
    # Get complexity data
    df = read_model_output()   
    # run image processing code on all 1100 images - takes 2-3 hours
    ip.save_image_stats(df, tag="", plot=False)


def analyse_image():
    df = read_model_output() 
    cme = "HCME_B__20130830_01"
    ip.plot_CME_mask(df, cme)
    ip.plot_CME_mask(df, cme, tag="BE")
    ip.plot_img_hist(cme)
    ip.plot_img_hist(cme, tag="BE")
    ip.get_image_summary(df, cme, mask=True)
    ip.get_image_summary(df, cme, mask=False)
    ip.get_image_summary(df, cme, mask=True, tag="BE")
    ip.get_image_summary(df, cme, mask=False, tag="BE")
    

def brightness_plots():
    # Get data
    df = read_model_output()
    df = de.add_image_stats_to_df(df, tag="")
    df = de.add_spacecraft_separation_to_df(df)
    df = de.add_width_to_df(df)
    df = de.add_col_to_df(df, 'bright_pix', 'width', 'divide', 'bright_pix_norm')
    # Make plots
    an.plot_vs_complexity(df, 'nan_count', 'No. NaN Pixels in Image', corr=[600000,-3])
    an.plot_vs_complexity(df, 'nan_count', 'No. NaN Pixels in Image', lim_x=[0, 30000], corr=[19000,-3])
    an.plot_vs_complexity(df, 'sat_pix', 'No. Saturated Pixels in Image', corr=[130000,-3])
    an.plot_vs_complexity(df, 'sat_pix_in_cme', 'No. Saturated Pixels in CME Area', corr=[0.4,-3])
    an.plot_vs_complexity(df, 'bright_pix', 'Bright Pixels in Image', corr=[100000,-3])
    an.plot_vs_complexity(df, 'mean_brightness_in_cme', 'Mean CME brightness', corr=[35,-3])
    an.plot_vs_complexity(df, 'std_abs_mean_brightess_in_cme', 'S. Dev of Mean CME brightness', corr=[65,-3])
    an.plot_vs_complexity(df, 'abs_mean_brightess_in_cme', 'Absolute Mean CME brightness', corr=[35,-3])
    an.plot_vs_complexity(df, 'std_abs_mean_brightess_in_cme', 'S. Dev of Abs Mean CME brightnesss', corr=[65,-3])
    an.plot_vs_complexity(df, 'sc_sep', 'Distance from CME to Spacecraft', corr=[2.5e7, -3])
    an.plot_x_vs_y(df, 'sc_sep', 'bright_pix', "Distance between CME and spacecraft", "Bright Pixels", corr=[2.5e7, 60000])
    an.plot_vs_helcats(df, 'SSE_phi', 'higeocat', 'SSE_phi', 'bright_pix', "Phi", "Bright Pixels", corr=False)
    an.plot_x_vs_y(df, 'width', 'bright_pix', 'Angular Width', 'Bright Pixels in Image', corr=[20,125000])
    an.plot_vs_complexity(df, 'bright_pix_norm', 'Bright Pixels in Image, Normalised by Angular Width', corr=[1000,-3])
    an.plot_helcats_vs_complexity(df, 'GCS_CME_Mass', 'kincat', 'CME Mass', corr=[3e17, -2])


if __name__ == '__main__':
    make_assets()
    process_classification_data(suffix="BRIGHTNESSEQUALISEDFINAL")
    show_ranking()
    complexity_over_time(tag='')
    complexity_over_time(tag='_be')
    analyse_image()
    fit_a_b_separately()
    paired_comparison_data()
    compare_a_b()
    process_images()
    brightness_plots()
    brightness_plots()
    width_plots()

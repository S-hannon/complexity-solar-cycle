"""
This script runs all the analysis included in the paper:
    Jones et al. The visual complexity of CMEs follows the solar cycle.
"""
import data2df
import popfss_image_processing as ip
import popfss_ranking_analysis as ra
from popfss_compare_solar_storms_workflow import CompareSolarStormsWorkflow
from data_cme_complexity import CMEComplexity

###############################################################################
# Change these to desired data and figure locations
data_loc = r"path\\to\\data\\folder"
fig_loc = r"path\\to\\fig\\folder"
###############################################################################
example_cme = 'HCME_A__20081212_01'


def make_images():
    """Requires .fits files for HI images.
    Creates ~1110+ running differenced images in folder:
        ...data_loc\\STEREO_HI\\Images\\POPFSS\\diff\\sta or stb
    Applies brightness equalisation to all running differenced images and 
    saves the new images in folder:
        ...data_loc\\STEREO_HI\\Images\\POPFSS\\diff_be\\sta or stb
    """
    # Make running-differenced CME images for all HELCATS CMEs with time-elongation
    # tracking 
    wf = CompareSolarStormsWorkflow(data_loc)
    wf.make_assets('diff', camera='hi1', background_type=1)
    # Plot image histograms before and after hist equalisation for example CME
    ip.plot_img_hist(example_cme, 'diff')
    ip.plot_img_hist(example_cme, 'diff_be')
    # Apply histogram equalisation to all the differenced images
    ip.brightness_equalise_images('POPFSS', 'diff')
    

def make_manifest(img_type='diff'):
    """Requires images to be created first.
    Creates a number of manifest files, and saves in folder:
        ...data_loc\\POPFSS
    """
    # Make a manifest file to use to upload the images to the zooniverse project
    # builder site, and choose which paired comparisons to do
    wf = CompareSolarStormsWorkflow(data_loc)
    wf.make_manifest(img_type)


def analyse_manifest(manifest_name):
    wf = CompareSolarStormsWorkflow(data_loc)
    wf.analyse_manifest(manifest_name)


def analyse_classification_data(img_type='diff'):
    """Requires protect-our-planet-from-solar-storms-classifications.csv to be
    downloaded from Zooniverse or figshare and saved in folder:
        ...data_loc\\POPFSS
    """
    # Process the project data from the Zooniverse site, into a usable format
    wf = CompareSolarStormsWorkflow(data_loc, fig_loc)
    # wf.process_classifications(img_type)
    # Investigate the number of classification by each user
    wf.classifications_by_user(img_type)
    # Look at the results of the paired comparison data: how often did people
    # agree?
    wf.plot_paired_comparison_results(img_type)
    # Investigate yearly trends in the data without fitting the B-T model
    wf.plot_wins_per_year(img_type)
    # How often were the left and right images picked?
    wf.left_vs_right_bias(img_type)
    
    
def analyse_images(img_type='diff'):
    """Requires popfss_model_fit_r_<img_type>.csv to be saved in folder:
        ...data_loc\\POPFSS
    Takes a while to run.
    """
    # Load the params from the B-T model fitting (done in R)
    # this has file name: popfss_model_fit_r_<img_type>.csv
    data = CMEComplexity(data_loc)
    df = data.load(img_type)
    df = data2df.add_craft_and_time_to_df(df)
    # Show example mask
    ip.plot_img_with_mask(example_cme, 'POPFSS', img_type)
    ip.image_summary(example_cme, img_type)
    # Loop over images, apply mask, count NaNs and number of bright pixels
    ip.save_img_stats(df, 'POPFSS', img_type)


def make_animation(img_type='diff', all_cmes=False):
    """Requires popfss_model_fit_r_<img_type>.csv to be saved in folder:
        ...data_loc\\POPFSS
    """
    # Load the params from the B-T model fitting (done in R)
    # this has file name: popfss_model_fit_r_<img_type>.csv
    data = CMEComplexity(data_loc)
    df = data.load(img_type)
    df = data2df.add_craft_and_time_to_df(df)
    # Create an animation showing how CMEs change throughout the ranking
    # use all_cmes=True to make a long animation showing all CMEs
    # use all_cmes=False to make a short animation showing a selection of CMEs
    ra.animate_ranking(df, img_type, all_cmes=all_cmes)


def analyse_ranking(img_type='diff'):
    """Requires popfss_model_fit_r_<img_type>.csv to be saved in folder:
        ...data_loc\\POPFSS
    """
    data = CMEComplexity(data_loc)
    df = data.load(img_type)
    df = data2df.add_craft_and_time_to_df(df)
    # Investigate the distribution of complexity values
    ra.complexity_distribution(df)
    # Investigate the distribution of annual complexity values
    ra.annual_complexity_distributions(df)
    # Plot average annual complexity over time
    ra.means_points_plot(df, err='std')
    ra.means_points_and_sunspots_plot(df, err='std')
    # Work out the average complexity difference between CMEs observed by
    # STEREO-A and STEREO-B
    ra.sta_stb_complexity_diff(df)
    # Investigate the subset of CMEs observed by both spacecraft
    ra.events_observed_by_both_craft(df)
    # Is complexity associated with artefacts in the images?    
    ra.plot_nan_count(df, img_type)
    # Calculate correlation between CME complexity and angular width
    ra.width_vs_complexity_corr(df)
    # Try splitting the CMEs into 4 groups
    ra.plot_complexity_time_four_groups_width(df)
    # Plot the number of bright pixels vs. complexity
    ra.plot_bright_pixels(df, img_type)
    

if __name__ == "__main__":
    make_images()
    make_manifest('diff')
    analyse_manifest('manifest_diff_used.csv')
    analyse_classification_data('diff')
    analyse_images('diff')
    make_animation('diff', all_cmes=False)
    analyse_ranking('diff')
    
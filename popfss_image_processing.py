from __future__ import division
import os
import fnmatch
import sys
import glob
#import cv2
import numpy as np
import pandas as pd
import scipy.stats as sps
import matplotlib as mpl
import matplotlib.pyplot as plt
import astropy.units as u
import sunpy.map as smap
from PIL import Image, ImageDraw
from datetime import datetime, timedelta
import popfss_asset_production as ap
import popfss_data_extraction as de
import hi_processing as hip


###############################################################################
# Functions for getting / making images
def get_CME_img(helcats_name, tag=""):
    """Loads the image for the required CME from the folder specified in the
    project directiories.
    """
    project_dirs = de.get_project_dirs()
    for file_name in os.listdir(project_dirs['images' + tag]):
        if fnmatch.fnmatch(file_name, '*' + helcats_name + '*'):
            img_path = os.path.join(project_dirs['images' + tag], file_name)
    # has the image been rotated? If so, rotate back.
    craft, date = de.get_cme_details(helcats_name)
    if craft == 'stb':
        img = Image.open(img_path).transpose(Image.FLIP_LEFT_RIGHT)
    elif craft == 'sta':
        if date.year > 2015:
            img = Image.open(img_path).transpose(Image.FLIP_LEFT_RIGHT)
        else:
            img = Image.open(img_path)
    return img


def make_CME_img(df, i, save=False):
    """Makes the differenced image for the required CME using the .FTS files
    on the hard-drives.
    """
    out_img = False
    out_name = False
    hi_map = False
    # TODO: do add_mid_el_to_df and check if done already
    if df.start_time[i].year > datetime(2007, 1, 1).year:
        # Search hard drive for matching HI image
        hi_files = hip.find_hi_files(df.start_time[i],
                                     df.end_time[i],
                                     craft=df.craft[i],
                                     camera='hi1',
                                     background_type=1)
        if len(hi_files) > 1:
            # Loop over the hi_files, make image
            files_c = hi_files[1:]  # current frame files
            files_p = hi_files[0:-1]  # previous frame files

            # Loop over images found and store times
            ts = []
            for j in range(len(files_c)):
                time_string = files_c[j][-25:-10]
                ts.append(pd.datetime.strptime(time_string, '%Y%m%d_%H%M%S'))
            # Find nearest HI frame to the time the CME reaches
            # half-way through the HI1 FOV
            idx = ap.find_nearest(ts, df.mid_time[i])
            fc = files_c[idx]  # current frame
            fp = files_p[idx]  # previous frame to fc (for differencing)

            # Make the differenced image
            hi_map = hip.get_image_diff(fc, fp, align=True, smoothing=True)
            # Make it grey
            diff_normalise = mpl.colors.Normalize(vmin=-0.05, vmax=0.05)
            out_img = mpl.cm.gray(diff_normalise(hi_map.data), bytes=True)
            # Get the image and flip upside down as there is no "origin=lower" option with PIL
            out_img = Image.fromarray(out_img)
#            out_img = Image.fromarray(np.flipud(out_img))

            proj_dirs = de.get_project_dirs()
            out_name = "_".join([df.helcats_name[i], df.craft[i], hi_map.date.strftime('%Y%m%d_%H%M%S')]) + '.jpg'
            if save == True:
                # Save the image
                out_path = os.path.join(proj_dirs['figures'], out_name)
                out_img = out_img.convert(mode='RGB')
                out_img.save(out_path)
    return out_img, hi_map, out_name


def get_hi_map(helcats_name):
    """Retrieves hi_map from hard drive for a given CME.
    """
    craft, date = de.get_cme_details(helcats_name)
    day_string = str(date.year) + str(date.month).zfill(2) + str(date.day).zfill(2)
    # folder path depends on hard drive
    project_dirs = de.get_project_dirs()
    if date.year < 2014:
        hi_folder = os.path.join(project_dirs['hi_data'], "L2_1_25", str(craft[2]), 'img\hi_1', day_string)
    else:
        hi_folder = os.path.join(project_dirs['hi_data2'], "L2_1_25", str(craft[2]), "img\hi_1", day_string)
    # find correct file from folder
    for filename in os.listdir(hi_folder):
        if filename.startswith(day_string + '_' + str(date.hour) + str(date.minute)):
            hi_map = smap.Map(os.path.join(hi_folder, filename))
    return hi_map


###############################################################################
# Code for applying a CME mask to an image
def get_CME_mask_bounds(df, i, hi_map):
    """finds coords of box of CME area.
    """
    # Add required data to df
    df = de.add_width_to_df(df)
    df = de.add_mid_el_to_df(df)
    # Extract limits of rough cme area
    coord1 = hip.convert_hpr_to_pix(4 * u.deg, df.pa_n[i] * u.deg, hi_map)
    coord2 = hip.convert_hpr_to_pix(df.mid_el[i] * u.deg, df.pa_n[i] * u.deg, hi_map)
    coord4 = hip.convert_hpr_to_pix(df.mid_el[i] * u.deg, df.pa_s[i] * u.deg, hi_map)
    coord5 = hip.convert_hpr_to_pix(4 * u.deg, df.pa_s[i] * u.deg, hi_map)
    coord3_list = []
    if df.pa_n[i] < df.pa_s[i]:  # CME goes right to left (STA pre 2016)
        # Loop over each position angle between PA_N and PA_S
        pa = int(df.pa_n[i]) + 1
        while pa < df.pa_s[i]:
            coord =  hip.convert_hpr_to_pix(df.mid_el[i] * u.deg, pa * u.deg, hi_map)
            coord3_list.append(coord)
            pa = pa + 1
    else:  # CME goes left to right (STB & STA post 2016)
        pa = df.pa_n[i] - 1
        while pa > df.pa_s[i]:
            coord =  hip.convert_hpr_to_pix(df.mid_el[i] * u.deg, pa * u.deg, hi_map)
            coord3_list.append(coord)
            pa = pa - 1
    return coord1, coord2, coord3_list, coord4, coord5


def get_CME_img_with_mask(df, i, img, hi_map):
    coord1, coord2, coord3_list, coord4, coord5 = get_CME_mask_bounds(df, i, hi_map)
    polygon = []
    polygon.append((coord1[0].value, coord1[1].value))
    polygon.append((coord2[0].value, coord2[1].value))
    for j in range(len(coord3_list)):
        polygon.append((coord3_list[j][0].value, coord3_list[j][1].value))
    polygon.append((coord4[0].value, coord4[1].value))
    polygon.append((coord5[0].value, coord5[1].value))
    # Crop image to required area
    # Code adapted from https://stackoverflow.com/questions/22588074/polygon-crop-clip-using-python-pil
    img_array = np.asarray(img)
    if len(np.shape(img)) > 2:
        img_array = img_array[:,:,0]  # only needed if 3 dimensions
    # create mask
    mask_img = Image.new('L', (img_array.shape[1], img_array.shape[0]), 0)
    ImageDraw.Draw(mask_img).polygon(polygon, outline=1, fill=1)
    mask = np.array(mask_img)
    # assemble new image (uint8: 0-255)
    new_img_array = np.empty(img_array.shape, dtype='uint8')
    new_img_array[:,:] = mask * img_array[:,:]
    new_img = Image.fromarray(new_img_array)
    return new_img


def plot_CME_mask(df, helcats_name, tag=""):
    """Makes a plot showing the CME image and CME masked image side-by-side.
    """
    hi_map = get_hi_map(helcats_name)
    img = get_CME_img(helcats_name, tag=tag)
    # need whole dataframe as needs to find PA_N and PA_S
    i = de.get_df_index(df, helcats_name)
    masked_img = get_CME_img_with_mask(df, i, img, hi_map)
    # Plot to compare original and cropped images
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=[9, 6])
    ax1.imshow(img)
    ax2.imshow(masked_img)
    ax1.set_title('Differenced Image')
    ax2.set_title('CME Area')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    # Save
    project_dirs = de.get_project_dirs()
    f.savefig(os.path.join(project_dirs['figures'], 'CME Area ' + helcats_name + '.png'))


###############################################################################
# Code for investigating one CME
def plot_img_hist(helcats_name, tag=""):
    """Makes plot showing image on left and histogram of pixel values on the 
        right.
    """
    # make Image instance
    img = get_CME_img(helcats_name, tag=tag)
    # Plot
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=[9, 6])
    ax1.imshow(img)
    a = list(img.getdata(band=0))
    ax2.hist(a, bins=np.arange(0, 255, 1))
    # Make pretty
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_ylim((0, 50000))
    asp = np.diff(ax2.get_xlim())[0] / np.diff(ax2.get_ylim())[0]
    ax2.set_aspect(asp)
    ax2.set_xlabel('Pixel Value')
    ax2.set_ylabel('Count')
    plt.tight_layout()
    # Save
    project_dirs = de.get_project_dirs()
    name = os.path.join(project_dirs['figures'], helcats_name + ' hist ' + tag + '.png')
    f.savefig(name)


def get_image_summary(df, helcats_name, mask=False, tag=""):
    """Does all the image processing stuff for one image, and prints the
        values.
    :param: df: pandas df
    :param: helcats_name: str, HELCATS id of CME
    :param: mask: bool, should the rest of the image be masked out, leaving
        only the CME?
    :param: tag: suffix for images folder
    """
    # get data
    hi_map = get_hi_map(helcats_name)
    img = get_CME_img(helcats_name, tag=tag)
    if mask == True:
        i = de.get_df_index(df, helcats_name)
        img = get_CME_img_with_mask(df, i, img, hi_map)
        data = list(img.getdata(0))
        # Remove masked values
        data[:] = [value for value in data if value != 0]
    elif mask == False:
        data = list(img.getdata(0))
    # plot the CME image
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=[9, 6])
    ax1.imshow(img)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.hist(data, bins=np.arange(0, 255, 1))
    ax2.set_ylim((0, 50000))
    asp = np.diff(ax2.get_xlim())[0] / np.diff(ax2.get_ylim())[0]
    ax2.set_aspect(asp)
    ax2.set_xlabel('Pixel Value')
    ax2.set_ylabel('Count')
    plt.tight_layout()
    # Print the summary
    print("Summary of CME: %s" %(helcats_name))
    print("Image type: %s" %("Differenced " + tag))
    print("CME mask: %s" %(str(mask)))
    # first use the hi_map
    print("NaNs count: %s" %(np.isnan(hi_map.data).sum()))
    # then look at the image
    print("Fraction of saturated pixels: %s" %(data.count(255)/len(data)))
    print("Mean pixel value: %s" %(np.mean(data)))
    print("Standard deviation: %s" %(np.std(data)))
    print("Absolute mean pixel value: %s" %(np.mean(abs(np.array(data)))))
    print("Standard dev of abs mean pixel value: %s" %(np.std(abs(np.array(data)))))


###############################################################################
# Code to apply image processing above to all CMEs in a df
def save_image_stats(df, tag=""):
    """Loops over all the images, does stuff, and saves it as a .csv file.
    :param: df: pandas df
    :param: tag: string, suffix for saving with different name
    """
    # Add required data to df
    df = de.add_width_to_df(df)
    df = de.add_mid_el_to_df(df)
    # Things to add to df
    nan_count_image = []
    sat_pix_image = []
    sat_pix_cme = []
    bright_pix = []
    mean_bri_cme = []
    std_mean_bri_cme = []
    abs_mean_bri_cme = []
    std_abs_mean_bri_cme = []
    # Loop over the storms
    for i in df.index:
        print(i)
        # Make the differenced image
        img, hi_map, name = make_diff_im(df, i, save=True)
        if hi_map != False:

#            # for brightness equalisation
#            proj_dirs = get_project_dirs()
#            out_path = os.path.join(proj_dirs['figures'], name)
#            print out_path
#            img3 = cv2.imread(out_path,0)
#            img4 = cv2.equalizeHist(img3)
#            print np.shape(img4)
#            img = Image.fromarray(img4)
#            print np.shape(img)
#
#            os.remove(out_path)

            # Stats from complete image
            nan_count_image.append(np.isnan(hi_map.data).sum())
            sat_pix_image.append(list(img.getdata(0)).count(255))
            # Stats from cropped image
            new_img = crop_to_cme_area(df, i, img, hi_map)
            cme = list(new_img.getdata(0))
            # cme.count(0) is number of masked out pixels
            # len(cme) is total pixels in the image
            cme_pix = len(cme) - cme.count(0)  # so this is pixels in cme
            print(df['helcats_name'][i])
            print(cme_pix)
            print()
            if cme_pix > 0:
                sat_pix_cme.append(cme.count(255)/cme_pix)
            else:
                sat_pix_cme.append(np.NaN)
            bright = [value for value in cme if value > 222]
            bright_pix.append(len(bright))
            mean_bri_cme.append(np.mean(cme))
            std_mean_bri_cme.append(np.std(cme))
            abs_mean_bri_cme.append(np.mean(abs(np.array(cme))))
            std_abs_mean_bri_cme.append(np.std(abs(np.array(cme))))
        else:
            nan_count_image.append(np.NaN)
            sat_pix_image.append(np.NaN)
            sat_pix_cme.append(np.NaN)
            bright_pix.append(np.NaN)
            mean_bri_cme.append(np.NaN)
            std_mean_bri_cme.append(np.NaN)
            abs_mean_bri_cme.append(np.NaN)
            std_abs_mean_bri_cme.append(np.NaN)
    print(len(nan_count_image), len(sat_pix_image), len(sat_pix_cme), len(mean_bri_cme), len(std_mean_bri_cme), len(abs_mean_bri_cme), len(std_abs_mean_bri_cme))
    # Add to df
    df['nan_count'] = pd.Series(nan_count_image, index=df.index)
    df['sat_pix'] = pd.Series(sat_pix_image, index=df.index)
    df['bright_pix'] = pd.Series(bright_pix, index=df.index)
    df['sat_pix_in_cme'] = pd.Series(sat_pix_cme, index=df.index)
    df['mean_brightness_in_cme'] = pd.Series(mean_bri_cme, index=df.index)
    df['std_abs_mean_brightess_in_cme'] = pd.Series(std_mean_bri_cme, index=df.index)
    df['abs_mean_brightess_in_cme'] = pd.Series(abs_mean_bri_cme, index=df.index)
    df['std_abs_mean_brightess_in_cme'] = pd.Series(std_abs_mean_bri_cme, index=df.index)
    # save to csv file
    df.to_csv('N:\Documents\Projects\CME_Complexity_Ranking\Out_Data\image_stats' + tag + '.csv')

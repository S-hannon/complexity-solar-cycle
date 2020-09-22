from __future__ import division
import os
# import cv2
import fnmatch
import sys
import glob
import numpy as np
import pandas as pd
import scipy.stats as sps
import matplotlib as mpl
import matplotlib.pyplot as plt
import astropy.units as u
import sunpy.map as smap
from PIL import Image, ImageDraw
from datetime import datetime, timedelta
import hi_processing as hip
import misc
import data2df
from data_cme_complexity import CMEComplexity
from plotting_stereo import STEREOPlot
from data_stereo_hi import STEREOHI
from data_helcats import HELCATS



data_loc = r'C:\\Users\\shann\\OneDrive\\Documents\\Research\\Workspace\\Data'
fig_loc = r"C:\\Users\\shann\\OneDrive\\Documents\\Research\\Workspace\\Plots"
hi_data = STEREOHI(data_loc)


def brightness_equalise_images(tag, img_type):
    root = "C:\\Users\\shann\\OneDrive\\Documents\\Research\\Workspace\\Data\\STEREO_HI\\Images"
    for craft in ['sta', 'stb']:
        img_path = os.path.join(root, tag, img_type, craft)
        be_path = os.path.join(root, tag, img_type + '_be', craft)
        if not os.path.exists(be_path):
            os.mkdir(be_path)
        for filename in os.listdir(img_path):
            if filename.endswith(".jpg"):
                img = cv2.imread(os.path.join(img_path, filename),0)
                be = cv2.equalizeHist(img)
                cv2.imwrite(os.path.join(be_path, filename), be)


###############################################################################
# add a crude CME mask
def find_cme_mask_bounds(helcats_name):
    """finds coords of box of CME area.
    """
    # get the upper and lower PAs, and elongation of the CME in the image
    helcats = HELCATS(data_loc)
    craft, time = helcats.get_cme_details(helcats_name)
    pa_n = helcats.get_col_data('PA-N [deg]', [helcats_name])[0]
    pa_s = helcats.get_col_data('PA-S [deg]', [helcats_name])[0]
    start, mid, end, mid_el = helcats.get_te_track_times(helcats_name)
    hi_data = STEREOHI(data_loc)
    hi_map = hi_data.get_hi_map(craft, mid)
    # convert these points into pixel coordinates
    coord1 = hip.convert_hpr_to_pix(4 * u.deg, pa_n * u.deg, hi_map)
    coord2 = hip.convert_hpr_to_pix(mid_el * u.deg, pa_n * u.deg, hi_map)
    coord4 = hip.convert_hpr_to_pix(mid_el * u.deg, pa_s * u.deg, hi_map)
    coord5 = hip.convert_hpr_to_pix(4 * u.deg, pa_s * u.deg, hi_map)
    coord3_list = []
    if pa_n < pa_s:  # CME goes right to left (STA pre 2016)
        # Loop over each position angle between PA_N and PA_S
        pa = int(pa_n) + 1
        while pa < pa_s:
            coord =  hip.convert_hpr_to_pix(mid_el * u.deg, pa * u.deg, hi_map)
            coord3_list.append(coord)
            pa = pa + 1
    else:  # CME goes left to right (STB & STA post 2016)
        pa = pa_n - 1
        while pa > pa_s:
            coord =  hip.convert_hpr_to_pix(mid_el * u.deg, pa * u.deg, hi_map)
            coord3_list.append(coord)
            pa = pa - 1
    # append coords in order
    polygon = [(coord1[0].value, coord1[1].value),
               (coord2[0].value, coord2[1].value)]
    for j in range(len(coord3_list)):
        polygon.append((coord3_list[j][0].value, coord3_list[j][1].value))
    polygon.append((coord4[0].value, coord4[1].value))
    polygon.append((coord5[0].value, coord5[1].value))
    return polygon


def add_cme_mask(helcats_name, img):
    polygon = find_cme_mask_bounds(helcats_name)
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


###############################################################################
# plots to demonstrate
def plot_img_with_mask(helcats_name, tag, img_type):
    """Makes a plot showing the CME image and CME masked image side-by-side.
    """
    hi_data = STEREOHI(data_loc)
    helcats = HELCATS(data_loc)
    craft, time = helcats.get_cme_details(helcats_name)
    img = hi_data.load_img(helcats_name, craft, tag, img_type)
    masked_img = add_cme_mask(helcats_name, img)
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
    f.savefig(os.path.join(fig_loc, 'CME Area ' + helcats_name + '.png'))


def plot_img_hist(helcats_name, tag, img_type):
    """Makes plot showing image on left and histogram of pixel values on the 
        right.
    """
    hi_data = STEREOHI(data_loc)
    helcats = HELCATS(data_loc)
    craft, time = helcats.get_cme_details(helcats_name)
    img = hi_data.load_img_from_file(tag, img_type, craft, name=helcats_name)
    # Plot
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=[9, 6])
    ax1.imshow(img, cmap='gray')
    a = list(img.getdata(band=0))
    ax2.hist(a, bins=np.arange(0, 255, 1), color='orange')
    # Make pretty
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_ylim((0, 50000))
    asp = np.diff(ax2.get_xlim())[0] / np.diff(ax2.get_ylim())[0]
    ax2.set_aspect(asp)
    ax2.set_xlabel('Pixel Value')
    ax2.set_ylabel('Count')
    plt.tight_layout()
    f.savefig(os.path.join(fig_loc, helcats_name + ' hist ' + img_type + '.png'))


def image_summary(df, helcats_name, tag, img_type, mask=False):
    """Does all the image processing stuff for one image, and prints the
        values.
    :param: df: pandas df
    :param: helcats_name: str, HELCATS id of CME
    :param: mask: bool, should the rest of the image be masked out, leaving
        only the CME?
    :param: tag: suffix for images folder
    """
    # get data
    hi_data = STEREOHI(data_loc)
    helcats = HELCATS(data_loc)
    craft, time = helcats.get_cme_details(helcats_name)
    img = hi_data.load_img_from_file(tag, img_type, craft, name=helcats_name)
    if mask == True:
        masked_img = add_cme_mask('helcats_name', img)
        data = list(masked_img.getdata(0))
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
    print("Image type: %s" %(img_type))
    print("CME mask: %s" %(str(mask)))
    # first use the hi_map
    # print("NaNs count: %s" %(np.isnan(hi_map.data).sum()))
    # then look at the image
    print("Fraction of saturated pixels: %s" %(data.count(255)/len(data)))
    print("Mean pixel value: %s" %(np.mean(data)))
    print("Standard deviation: %s" %(np.std(data)))
    print("Absolute mean pixel value: %s" %(np.mean(abs(np.array(data)))))
    print("Standard dev of abs mean pixel value: %s" %(np.std(abs(np.array(data)))))


###############################################################################
# Code to apply image processing above to all CMEs in a df
def save_img_stats(df, tag, img_type):
    """Loops over all the images, does stuff, and saves it as a .csv file.
    :param: df: pandas df
    :param: tag: str
    """
    df = data2df.add_width_to_df(df)
    df = data2df.add_te_track_times_to_df(df)
    hi_data = STEREOHI(data_loc)
    nan_count = []
    bright_pix = []
    for i in df.index:
        print('processing image %s of %s'%(i, len(df)))
        img = hi_data.load_img_from_file(tag, img_type, df['craft'][i],
                                         name=df['helcats_name'][i])
        hi_map = hi_data.get_hi_map(df['craft'][i], df['time'][i])
        if hi_map != False:
            # Stats from complete image
            nan_count.append(np.isnan(hi_map.data).sum())
            # sat_pix_image.append(list(img.getdata(0)).count(255))
            # stats from cropped image
            new_img = add_cme_mask(df['helcats_name'][i], img)
            cme = list(new_img.getdata(0))
            # cme.count(0) is number of masked out pixels
            # len(cme) is total pixels in the image
            # cme_pix = len(cme) - cme.count(0)  # so this is pixels in cme
            bright = [value for value in cme if value > 222]
            bright_pix.append(len(bright))
        else:
            print('error: hi_map not found for cme', df['helcats_name'][i])
            nan_count.append(np.NaN)
            bright_pix.append(np.NaN)
    df['nan_count'] = pd.Series(nan_count, index=df.index)
    df['bright_pix'] = pd.Series(bright_pix, index=df.index)
    df.to_csv(os.path.join(data_loc, 'popfss_image_stats_' + img_type + '.csv'))


def load_img_stats(img_type):
    """reads in image stats from .csv file created and adds a new column for 
    each image stat.
    """
    df = pd.read_csv(os.path.join(data_loc, 'POPFSS', 'popfss_image_stats_' + img_type + '.csv'))
    return df

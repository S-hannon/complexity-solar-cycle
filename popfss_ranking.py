from __future__ import division
import os
import glob
import fnmatch
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import PIL.Image as Image

from popfss_data_extraction import get_project_dirs


#def visualise_ranking(params):
#    # Locate the assets
#    project_dirs = get_project_dirs()
#    images = os.path.join(project_dirs['out_data'], 'comp_assets')
#    
#    # Load assets at each end
#    least_complex = params['Unnamed: 0'][np.argmin(params['x'])][2:4]
#    most_complex = params['Unnamed: 0'][np.argmax(params['x'])][2:3]
#    
#    print len(params['Unnamed: 0'][np.argmin(params['x'])])
#    print len(params['Unnamed: 0'][np.argmax(params['x'])])    
#    
#    ims_to_show = [least_complex, most_complex]
#    nam_to_show = ["least complex", "most complex"]
#    
#    for i in range(len(ims_to_show)):
#        for filename in os.listdir(images):
#            if filename.endswith(ims_to_show[i]+".jpg"):
#                print nam_to_show[i]
#                imfile = Image.open(os.path.join(images, ims_to_show[i]+".jpg"))
#                plt.imshow(imfile)
#
#
#def visualise_whole_ranking(params):
#
#    if len(params) <= 100:
#        plot_ranking(params, 'all')
#        
#    else:
#        num_plots = np.int(np.ceil(len(params) / 100))
#        print "num plots %s" %(num_plots)
#        # Order ranking
#        params2 = params.sort_values(by='x')
#        for p in range(num_plots):
#            end = (p + 1) * 100
#            if end > len(params2):
#                end = len(params2)
#            rows = range((p * 100), end)
#            plot_ranking(params2.iloc[rows], '_' + str(p))
#
#
#def plot_ranking(params, plotname):
#    # Locate the assets
#    project_dirs = get_project_dirs()
#    images = os.path.join(project_dirs['out_data'], 'comp_assets', 'sta')
#    # Make sure ranking in order
#    params2 = params.sort_values(by='x')    
#    # Create the containing plot
#    f, axes = plt.subplots(10, 10, figsize=(60, 60))
#    # Initialise sub plot counters
#    i = 0
#    j = 0
#    # Loop over the storms    
#    for v in range(len(params2)):
#        name = params2['Unnamed: 0'].values[v]
#        time = name[(len(name)-19):(len(name)-4)]
#        # Drop dots before filename
#        name = name[2:len(name)]
#        # Loop over files to find matching image
#        for filename in os.listdir(images):
#            if filename.endswith(name):
#                imfile = Image.open(os.path.join(images, name))
#                axes[i, j].imshow(imfile)
#                axes[i, j].set_xticks([])
#                axes[i, j].set_yticks([])
#                title = "{0:.3f}".format(params2['x'].values[v])
#                axes[i, j].set_title(time+' '+title)
#                i = i + 1
#                if i >= 10:
#                    i = i - 10
#                    j = j + 1
#    # Save and move to next plot
#    savename = os.path.join(project_dirs['figures'], 'ranking' + plotname)
#    plt.savefig(savename)
#    plt.close('all')
#
#
#def ranking_overview(params):
#    # Locate the assets
#    project_dirs = get_project_dirs()
#    images = os.path.join(project_dirs['out_data'], 'comp_assets')
#    
#    # Check whether to use all images or a subset
#    spacing = np.int(np.floor(len(params)/100))
#    if spacing < 1:
#        # Use all images
#        im_nos = range(0, len(params), 1)
#        # Create the containing plot
#        f, axes = plt.subplots(len(params), 1, figsize=(8, 60))
#    else:
#        im_nos = range(0, len(params), spacing)
#        sub_plots = int(np.ceil(np.sqrt(len(im_nos))))
#        f, axes = plt.subplots(sub_plots, sub_plots, figsize=(60, 60))
#    # Put the ranking in order
#    params2 = params.sort_values(by='x')
#    # Initialise sub plot counters
#    i = 0
#    j = 0
#    # Loop over the storms    
#    # Check if this value used
#    for v in im_nos:
#        name = params2['Unnamed: 0'].values[v]
#        time = name[(len(name)-19):(len(name)-4)]
#        # Drop dots before filename
#        name = name[2:len(name)]
#        # Loop over files to find matching image
#        for filename in os.listdir(images):
#            if filename.endswith(name):
#                imfile = Image.open(os.path.join(images, name))
#                axes[i, j].imshow(imfile)
#                axes[i, j].set_xticks([])
#                axes[i, j].set_yticks([])
#                title = "{0:.3f}".format(params2['x'].values[v])
#                axes[i, j].set_title(time+' '+title)
#        i = i + 1
#        if i >= sub_plots:
#            i = i - sub_plots
#            j = j + 1
#    # Save and move to next plot
#    savename = os.path.join(project_dirs['figures'], 'whole_ranking')
#    plt.savefig(savename)
#    plt.close('all')
#    
    
#def animate_ranking_part(params):
#    # Locate the assets
#    project_dirs = get_project_dirs()
#    images = os.path.join(project_dirs['out_data'], 'comp_assets')
#    # Put the ranking in order
#    params2 = params.sort_values(by='x')
#    # choose subset
#    spacing = np.int(np.floor(len(params)/100))
#    im_nos = range(0, len(params), spacing)
#    # Loop over the storms
#    count = 0
#    for i in range(len(im_nos)):
#        # Create the containing plot
#        f = plt.figure(figsize=(10, 10))
#        name = params2['Unnamed: 0'].values[im_nos[i]]
#        # Drop dots before filename
#        name = name[2:len(name)]
#        # Loop over files to find matching image
#        for filename in os.listdir(images):
#            if filename.endswith(name):
#                imfile = Image.open(os.path.join(images, name))
#                plt.imshow(imfile)
#                plt.xticks([])
#                plt.yticks([])
#        name = "_".join(["ani_f{0}".format(count)])
#        name = os.path.join(project_dirs['figures'], name)
#        f.savefig(name)
#        count = count + 1
#        name = "_".join(["ani_f{0}".format(count)])
#        name = os.path.join(project_dirs['figures'], name)
#        f.savefig(name)
#        count = count + 1
#        name = "_".join(["ani_f{0}".format(count)])
#        name = os.path.join(project_dirs['figures'], name)
#        f.savefig(name)
#        count = count + 1
#    # Add three black slides
#    f = plt.figure(figsize=(10, 10))
#    plt.xticks([])
#    plt.yticks([])
#    ax = plt.gca()
#    ax.set_facecolor('white')
#    name = "ani_f%s" %(count)
#    name = os.path.join(project_dirs['figures'], name)
#    f.savefig(name)  
#    count = count + 1
#    f = plt.figure(figsize=(10, 10))
#    plt.xticks([])
#    plt.yticks([])
#    ax = plt.gca()
#    ax.set_facecolor('white')
#    name = "ani_f%s" %(count)
#    name = os.path.join(project_dirs['figures'], name)
#    f.savefig(name) 
#    count = count + 1
#    f = plt.figure(figsize=(10, 10))
#    plt.xticks([])
#    plt.yticks([])
#    ax = plt.gca()
#    ax.set_facecolor('white')
#    name = "ani_f%s" %(count)
#    name = os.path.join(project_dirs['figures'], name)
#    f.savefig(name)  
#    plt.close('all')
            
        
def animate_ranking(df):
    # Locate the assets
    project_dirs = get_project_dirs()
    # Put the ranking in order
    df = df.sort_values(by='complexity')
    big_font = {'size':'20'}
    norm = mpl.colors.Normalize(np.min(df.complexity), np.max(df.complexity))
    range_norm = np.max(df.complexity) - np.min(df.complexity)
    cmap = mpl.cm.magma
    # Loop over the storms
    for i in range(len(df)):
        # Create the containing plot
        f, ax = plt.subplots(1, 1, figsize=(10, 10))
        name = df['helcats_name'][i]
        # Loop over files to find matching image
        for filename in os.listdir(project_dirs['images']):
            if fnmatch.fnmatch(filename, '*' + name + '*'):
                imfile = Image.open(os.path.join(project_dirs['images'], filename))
                ax.imshow(imfile)
                ax.set_xticks([])
                ax.set_yticks([])
                if df['craft'][i] == 'sta':
                    colour = 'pink'
                else:
                    colour = 'lightskyblue'
                ax.text(8, 34, name, color=colour, **big_font)
                ax.text(8, 1014,'Relative Complexity: ' + str('{:+.2f}'.format(round(df['complexity'][i], 2))), color=cmap(norm(df['complexity'][i])), **big_font)
                # sort out the colourbar
                cb = plt.cm.ScalarMappable(cmap='magma', norm=plt.Normalize(np.min(df.complexity), np.max(df.complexity)))
                cb._A = []  # pretend to set some axis
                cbar = plt.colorbar(cb, fraction=0.046, pad=0.04)
                ticklabs = cbar.ax.get_yticks()
                cbar.ax.set_yticklabels(['{0:+.0f}'.format(np.min(df.complexity)+x*range_norm) for x in ticklabs], fontsize=12)
                cbar.ax.axhline(y=norm(df['complexity'][i]), c='w')
                plt.tight_layout()
        num = str(i).zfill(4)
        name = "_".join(["ani_f{0}".format(num)])
        name = os.path.join(project_dirs['out_data'], 'Animation', name)
        f.savefig(name)
        plt.close('all')
    # Make animation and clean up.
    src = os.path.join(project_dirs['out_data'], "ani_f*.png")
    dst = os.path.join(project_dirs['out_data'], "complexity_index.gif")
    cmd = " ".join(["magick convert -delay 50 -loop 0", src, dst])
    os.system(cmd)
    # Tidy.
    files = glob.glob(src)
    for f in files:
        os.remove(f)
    
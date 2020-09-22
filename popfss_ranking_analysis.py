from __future__ import division
import os
import sys
import glob
import fnmatch
import random
import numpy as np
import pandas as pd
import PIL.Image as Image
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as sps
from datetime import datetime, timedelta
import popfss_image_processing as ip
import data2df
from plotting_stereo import STEREOPlot
from data_helcats import HELCATS
from data_silso import SILSO


data_loc = r"C:\\Users\\shann\\OneDrive\\Documents\\Research\\Workspace\\Data"
fig_loc = r"C:\\Users\\shann\\OneDrive\\Documents\\Research\\Workspace\\Plots"
    
    
def animate_ranking(df, img_type, all_cmes=True):
    # Put the ranking in order
    df = df.sort_values(by='complexity')
    norm = mpl.colors.Normalize(vmin=np.min(df.complexity),
                                vmax=np.max(df.complexity))
    range_norm = np.max(df.complexity) - np.min(df.complexity)
    cmap = mpl.cm.magma
    count = 0
    # Loop over the storms
    for n, i in enumerate(range(len(df))):
        # should an image be created for this CME?
        if not all_cmes:
            if (n/25).is_integer():
                flag = True
            else:
                flag = False
        else:
            flag = True
        # if so, make the image
        if flag == True:  
            count = count + 1
            # Create the containing plot
            f, ax = plt.subplots(1, 1, figsize=(10, 10))
            name = df['helcats_name'][i]
            # Loop over files to find matching image
            img_path = os.path.join(data_loc, 'STEREO_HI', 'Images', 'POPFSS',
                                    img_type, df['craft'][i])
            for filename in os.listdir(img_path):
                if fnmatch.fnmatch(filename, '*' + name + '*'):
                    imfile = Image.open(os.path.join(img_path, filename))
                    ax.imshow(imfile)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if df['craft'][i] == 'sta':
                        colour = 'pink'
                    else:
                        colour = 'lightskyblue'
                    ax.text(8, 34, name, color=colour, fontsize=20)
                    ax.text(8, 1014,
                            'Relative Complexity: ' +
                            str('{:+.2f}'.format(round(df['complexity'][i], 2))),
                            color=cmap(norm(df['complexity'][i])),
                            fontsize=20)
                    # sort out the colourbar
                    cb = plt.cm.ScalarMappable(norm=None, cmap=cmap)
                    cbar = plt.colorbar(cb, fraction=0.046, pad=0.04)
                    ticklabs = cbar.ax.get_yticks()
                    cbar.ax.set_yticklabels(['{0:+.0f}'.format(np.min(df.complexity)+x*range_norm) for x in ticklabs],
                                            fontsize=12)
                    cbar.ax.axhline(y=norm(df['complexity'][i]), c='w')
                    plt.tight_layout()
            num = str(i).zfill(4)
            name = os.path.join(fig_loc,
                                "_".join(["ani_f{0}".format(num)]))
            f.savefig(name)
            plt.close('all')
    # Make animation and clean up.
    src = os.path.join(fig_loc, "ani_f*.png")
    dst = os.path.join(fig_loc,
                       "complexity_index_"+ img_type + '_' + str(count) + ".gif")
    cmd = " ".join(["magick convert -delay 50 -loop 0", src, dst])
    os.system(cmd)
    # Tidy.
    files = glob.glob(src)
    for f in files:
        os.remove(f)


def complexity_distribution(df):
    plotting = STEREOPlot(data_loc, fig_loc)
    fig, ax = plt.subplots(figsize=[6, 6])
    sta_vals = df[df['craft'] == 'sta']['complexity']
    stb_vals = df[df['craft'] == 'stb']['complexity']
    ax = plotting.histogram(ax, sta_vals, stb_vals, 'Relative Complexity')
    ax.set_title('All CMEs', fontsize=16)
    ax.set_xlim([-5, 5])
    fig.savefig(os.path.join(fig_loc, 'complexity_histogram_all_cmes'))  
        
        
def annual_complexity_distributions(df):
    plotting = STEREOPlot(data_loc, fig_loc)
    # loop over years
    for year in range(np.min(df['time']).year, np.max(df['time']).year + 1):
        dfy = df[df['time'] > datetime(year, 1, 1, 1, 1)]
        dfy = dfy[dfy['time'] < datetime(year + 1, 1, 1, 1, 1)]
        # plot distribution
        fig, ax = plt.subplots(figsize=[6, 6])
        sta_vals = dfy[dfy['craft'] == 'sta']['complexity']
        stb_vals = dfy[dfy['craft'] == 'stb']['complexity']
        ax = plotting.histogram(ax, sta_vals, stb_vals, 'Relative Complexity')
        ax.set_title(str(year), fontsize=16)
        ax.set_xlim([-5, 5])
        ax.set_ylim([0, 30])
        fig.savefig(os.path.join(fig_loc, 'complexity_histogram_' + str(year)))        
        # print test results
        for craft in ['sta', 'stb']:
            dfyc = dfy[dfy.craft == craft]
            if len(dfyc) > 0:
                stat, pval = sps.normaltest(dfyc.complexity)
                stat1, pval1 = sps.skewtest(dfyc.complexity)
                stat2, pval2 = sps.kurtosistest(dfyc.complexity)
                print(year)
                print("normal test:", "stat", "{:.2f}".format(stat),
                      "pval", "{:.2f}".format(pval))
                print("skew test:", "stat", "{:.2f}".format(stat1),
                      "pval", "{:.2f}".format(pval1))
                print("kurtosis test:", "stat", "{:.2f}".format(stat2),
                      "pval", "{:.2f}".format(pval2))


def means_points_plot(df, err='std'):
    plotting = STEREOPlot(data_loc, fig_loc)
    f, ax = plt.subplots(1, figsize=[14, 9])
    ax = plotting.plot_x_vs_y_ax(ax, df, 'time', 'complexity', 'Year',
                                 'Relative Complexity', 
                                 lim_x=False,
                                 lim_y=False)
    # add means
    ax = plotting.add_time_means(ax, df, 'complexity',
                                 means_type='independent', means_t=12,
                                 err='std')
    ax_lo = np.min(df['time']) - timedelta(weeks=26)
    ax_hi = np.max(df['time']) + timedelta(weeks=26.5)
    ax.set_xlim((ax_lo, ax_hi))
    ax.set_ylabel('Relative Complexity', fontsize=16) 
    ax.set_xlabel('Time', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(loc=0, prop={'size': 12})
    plt.savefig(os.path.join(fig_loc, 'means points plot'))
    

def means_points_and_sunspots_plot(df, err='std'):
    plotting = STEREOPlot(data_loc, fig_loc)
    f, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=[14, 9])
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    # top panel
    ax1 = plotting.plot_x_vs_y_ax(ax1, df, 'time', 'complexity', 'Year',
                                  'Relative Complexity', 
                                  lim_x=False,
                                  lim_y=False)
    # add means
    ax1 = plotting.add_time_means(ax1, df, 'complexity',
                                  means_type='independent', means_t=12,
                                  err='std')
    # bottom panel
    sdf = SILSO(data_loc)
    sdf = sdf.load()
    ax2.plot(sdf.date, sdf.daily_total_sunspot_n, color='gold',
             label='Sunspots: daily totals', zorder=1)
    # means
    years = []
    sun_means = []
    sun_errs = []
    for year in range(np.min(df['time']).year, np.max(df['time']).year + 1):
        years.append(datetime(year, 1, 1) + timedelta(weeks=26))
        sdfy = sdf[sdf['date'] > datetime(year, 1, 1, 1, 1)]
        sdfy = sdfy[sdfy['date'] < datetime(year + 1, 1, 1, 1, 1)]
        sun_means.append(np.nanmean(sdfy['daily_total_sunspot_n']))
        s = np.nanstd(sdfy['daily_total_sunspot_n'])
        if err == 'sem':
            s = s / np.sqrt(len(sdfy))
        sun_errs.append(s)
    ax2.plot(years, sun_means, color='orangered', marker='s',
             linestyle='dashed',
             label='Sunspots: yearly means', zorder=2)
    ax2.errorbar(years, sun_means, yerr=sun_errs, color='orangered', capsize=3,
                 linestyle='None', zorder=2)
    # Format plot
    ax_lo = np.min(df['time']) - timedelta(weeks=26)
    ax_hi = np.max(df['time']) + timedelta(weeks=26.5)
    ax1.set_xlim((ax_lo, ax_hi))
    ax2.set_xlim((ax_lo, ax_hi))
    ax2.set_ylim(0, 210)
    ax1.set_ylabel('Relative Complexity', fontsize=20) 
    ax2.set_ylabel('Total Sunspots', color='k', fontsize=20)
    ax2.set_xlabel('Time', fontsize=20)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax1.set_xticks([])
    ax2.tick_params('y', colors='k') 
    ax1.legend(loc=0, prop={'size': 16})
    ax2.legend(loc=0, prop={'size': 16})
    # Make pretty
    plt.tight_layout()
    f.subplots_adjust(hspace=0)
    plt.savefig(os.path.join(fig_loc, 'means_points_sunspots'))


def sta_stb_complexity_diff(df):
    dfa = df[df.craft == 'sta']
    dfb = df[df.craft == 'stb']
    mean_sta = np.nanmean(dfa.complexity.values)
    mean_stb = np.nanmean(dfb.complexity.values)
    print("mean sta complexity",
          "{:.2f}".format(mean_sta))
    print("mean stb complexity",
          "{:.2f}".format(mean_stb))
    print("stdev sta complexity",
          "{:.2f}".format(np.nanstd(dfa.complexity.values)))
    print("stdev stb complexity",
          "{:.2f}".format(np.nanstd(dfb.complexity.values)))
    print("num sta CMEs", np.count_nonzero(~np.isnan(dfa.complexity.values)))
    print("num stb CMEs", np.count_nonzero(~np.isnan(dfb.complexity.values)))
    print("average sta-stb complexity difference",
          "{:.2f}".format(mean_sta - mean_stb))
    t_stat, p_val = sps.ttest_ind(dfa.complexity.values, dfb.complexity.values,
                                  axis=0, equal_var=True, nan_policy='omit')
    print("independent t-test:", "test statistic", "{:.2f}".format(t_stat),
          "p-value", "{:.2f}".format(p_val))
    t_stat2, p_val2 = sps.ks_2samp(dfa.complexity, dfb.complexity)
    print("K-S test 2 samples:", "test statistic", "{:.2f}".format(t_stat2),
          "p-value", "{:.2f}".format(p_val2))


def events_observed_by_both_craft(df):
    helcats = HELCATS(data_loc)
    sta_vals, stb_vals = helcats.get_matched_data(df.complexity,
                                                  df.helcats_name)
    t_stat, p_val = sps.ttest_rel(sta_vals, stb_vals)
    print("t-test related samples:", "test statistic", "{:.2f}".format(t_stat),
          "p-value", "{:.2f}".format(p_val))
    plotting = STEREOPlot(data_loc, fig_loc)
    fig, ax = plt.subplots(figsize=[12, 12])
    ax = plotting.plot_a_vs_b(ax, sta_vals, stb_vals, 'Visual Complexity')


def plot_nan_count(df, img_type):
    # df = df.merge(ip.load_img_stats(img_type), on=['helcats_name', 'craft',
    #                                                 'complexity'])
    df = ip.load_img_stats(img_type)
    # dist ns
    df = df[pd.notnull(df['nan_count'])]
    plt.figure(figsize=[10,7])
    plt.hist(df.nan_count, bins=np.arange(0, 40000, 500))
    plt.xlabel('NaN Count', fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    # now complexity vs nans
    plotting = STEREOPlot(data_loc, fig_loc)
    plotting.plot_x_vs_y(df, 'nan_count', 'complexity',
                         'NaN count', 'Visual complexity')
    plotting.plot_x_vs_y(ip.load_img_stats(img_type), 'nan_count', 'complexity',
                         'NaN count (close-up)', 'Visual complexity', lim_x=(0, 25000))
    

def width_vs_complexity_corr(df):
    plotting = STEREOPlot(data_loc, fig_loc)
    df = data2df.add_width_to_df(df)
    plotting.plot_x_vs_y(df, 'width', 'complexity', 'Angular Width ($\degree$) from HELCATS',
                      'Relative Complexity', corr=[90,-3])
    colors = ['pink', 'lightskyblue']
    colors2 = ['crimson', 'navy']
    names = ['STEREO-A', 'STEREO-B']
    print("Complexity vs Angular Width:")
    plt.figure(figsize=(10, 7))
    for n, craft in enumerate(['sta', 'stb']):
        dfc = df[df.craft == craft]
        corr, pval = sps.spearmanr(dfc['complexity'].values, dfc['width'].values,
                                   nan_policy='omit')
        print(names[n], "correlation coefficient:", "{:.2f}".format(corr),
              "p-value:", "{:.2f}".format(pval))
       # try shuffling indexes
        runs = 1000
        c = df[df.craft == craft].complexity.values
        w = df[df.craft == craft].width.values
        real_corr, real_pval = sps.spearmanr(c, w, nan_policy='omit')
        corrs = []
        pvals = []
        for i in range(1, runs):
            random.shuffle(c)
            random.shuffle(w)
            corr, pval = sps.spearmanr(c, w, nan_policy='omit')
            corrs.append(corr)
            pvals.append(pval)
        plt.xlabel("Spearman's Rank Correlation Coefficient", fontsize=16)
        plt.xlim((-1, 1))
        plt.ylabel("Frequency", fontsize=16)
        plt.hist(corrs, color=colors[n], alpha=0.5,
                 label=names[n] + ' %s Randomised Pairings'%(runs))
        plt.axvline(real_corr, color=colors2[n], label=names[n] + " Actual Correlation Coefficient")
        plt.legend(loc=0)
    print("How robust is this relationship?")
    print("Assume no relationship between CME complexity and angular width.")
    print("Shuffle the pairings 1000 times, and calculate correlation coefficient for each.")
    print("As the actual correlation coefficients are far into the tails, we can conclude the correlation is likely significant!")

    
def plot_complexity_time_four_groups(dfs, titles):
    plotting = STEREOPlot(data_loc, fig_loc)
    # Create the plot
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=[13, 10])
    plt.xlabel('Year')
    plt.ylabel('Relative Complexity')
    i = 0
    j = 0
    # Loop over plots
    for n in range(0, 4):
        plotting.plot_x_vs_y_ax(axes[i, j], dfs[n], 'time', 'complexity', 'Year',
                                'Relative Complexity', 
                                lim_x=(datetime(2008, 1, 1), datetime(2017, 1, 1)))
        plotting.add_time_means(axes[i, j], dfs[n], 'complexity')
        axes[i, j].set_title(titles[n] + ' (%s CMEs)'%(len(dfs[n])), fontsize=12)
        axes[i, j].set_xlabel('Year', fontsize=12)
        axes[i, j].set_ylabel('Relative Complexity', fontsize=12)
        # Sort counters
        if j < 1:
            j = j + 1
        else:
            i = i + 1
            j = 0
    plt.savefig(os.path.join(fig_loc, 'Complexity Time Means All Widths'))
    

def plot_complexity_time_four_groups_width(df):
    b1, b2, b3 = df.width.quantile([0.25, 0.5, 0.75])
    g1 = df[df.width < b1]
    g2 = df[(df.width > b1) & (df.width < b2)]
    g3 = df[(df.width > b2) & (df.width < b3)]
    g4 = df[df.width > b3]
    titles = ['Angular Width < $%s\degree$'%(b1),
              '$%s\degree$ < Angular Width < $%s\degree$'%(b1, b2),
              '$%s\degree$ < Angular Width < $%s\degree$'%(b2, b3),
              'Angular Width > $%s\degree$'%(b3)]
    plot_complexity_time_four_groups([g1, g2, g3, g4], titles)    
    

def plot_bright_pixels(df, img_type):
    df = ip.load_img_stats(img_type)
    # normalise by width / CME area
    df = data2df.add_width_to_df(df)
    bright_pix_norm = []
    for i in df.index:
        bright_pix_norm.append(df['bright_pix'][i] / df['width'][i])
    df['bright_pix_norm'] = bright_pix_norm
    plotting = STEREOPlot(data_loc, fig_loc)
    plotting.plot_x_vs_y(df, 'bright_pix_norm', 'complexity',
                          'Bright pixels, normalised by angular width',
                          'Visual complexity', corr=[1000,-3])    
    
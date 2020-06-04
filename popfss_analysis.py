from __future__ import division
import os
import glob
import fnmatch
import numpy as np
import pandas as pd
import PIL.Image as Image
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.stats as sps
from datetime import datetime, timedelta
from matplotlib.patches import Polygon
import popfss_data_extraction as de
from sunspots import get_sunspot_record


def animate_ranking(df):
    # Locate the assets
    project_dirs = de.get_project_dirs()
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
        
        
###############################################################################
def plot_complexity_time_four_groups(dfs, titles):
    # Create the plot
    fig, axes = plt.subplots(2, 2, sharex='all', sharey='all', figsize=[13, 10])
    big_font = {'size':'16'}
    i = 0
    j = 0
    # Loop over plots
    for n in range(0, 4):
        # Get data
        df = dfs[n]
        dfa = df[df.craft.values == 'sta']
        dfb = df[df.craft.values == 'stb']
        mdf = de.find_means(df, 12)
        # Plot points
        axes[i, j].scatter(dfa.time.values, dfa.complexity.values, color='pink', label='All STEREO-A storms')
        axes[i, j].scatter(dfb.time.values, dfb.complexity.values, color='lightskyblue', label='All STEREO-B storms')
        # Plot means and stds
        axes[i, j].plot(mdf.dates.values, mdf.sta_means.values, color='crimson', label='STEREO-A yearly means', marker='s', linestyle='dashed')
        axes[i, j].errorbar(mdf.dates.values, mdf.sta_means.values, yerr=mdf.sta_s.values, color='crimson', capsize=3, linestyle='None')
        axes[i, j].plot(mdf.dates.values, mdf.stb_means.values, color='navy', label='STEREO-B yearly means', marker='s', linestyle='dashed')
        axes[i, j].errorbar(mdf.dates.values, mdf.stb_means.values, yerr=mdf.stb_s.values, color='navy', capsize=3, linestyle='None')
        # Labels
        axes[i, j].set_xlabel('Time', **big_font)
        axes[i, j].set_ylabel('Relative Complexity', **big_font)
        axes[i, j].legend(loc=0)
        axes[i, j].set_xlim((datetime(2008, 1, 1), datetime(2017, 1, 1)))
        axes[i, j].set_title(titles[n] + ' (%s CMEs)'%(len(df)))
        # Sort counters
        if j < 1:
            j = j + 1
        else:
            i = i + 1
            j = 0
    # Save
    project_dirs = de.get_project_dirs()
    savename = os.path.join(project_dirs['figures'], 'Complexity Time Means All Widths')
    plt.savefig(savename)


def sunspot_means_plot(df):
    # Get data
    mdf = de.find_means(df, 12)
    project_dirs = de.get_project_dirs()
    sunspots = get_sunspot_record(project_dirs['sunspots'])
    # Create the plot
    fig, ax1 = plt.subplots(figsize=[8, 5])
    big_font = {'size':'16'}
    # Plot sunspots
    ax1.plot(sunspots.date, sunspots.daily_total_sunspot_n, color='lightgrey')
    ax1.set_xlim(mdf.dates[0] - timedelta(weeks=26),
                 mdf.dates[len(mdf.dates)-1] + timedelta(weeks=27))
    ax1.set_ylim(0, 210)
    ax1.set_ylabel('Daily Total Sunspot Number', color='grey', **big_font)
    ax1.tick_params('y', colors='grey')  
    ax1.set_xlabel('Time', **big_font)    
    # Plot means and stds
    ax2 = ax1.twinx()
    ax2.plot(mdf.dates.values, mdf.sta_means.values, color='crimson',
             label='STEREO-A yearly means', marker='s', linestyle='dashed')
    ax2.errorbar(mdf.dates.values, mdf.sta_means.values, yerr=mdf.sta_s.values,
                 color='crimson', capsize=3, linestyle='None')
    ax2.plot(mdf.dates.values, mdf.stb_means.values, color='navy',
             label='STEREO-B yearly means', marker='s', linestyle='dashed')
    ax2.errorbar(mdf.dates.values, mdf.stb_means.values, yerr=mdf.stb_s.values,
                 color='navy', capsize=3, linestyle='None')    
    ax2.set_ylabel('Relative Complexity', color='k', **big_font)
    ax2.tick_params('y', colors='k')
    ax2.legend(loc=0)
    # Save
    savename = os.path.join(project_dirs['figures'], 'sunspot_means')
    plt.savefig(savename)    


def means_points_and_sunspots_plot(df, err='std'):
    # Create the plot
    f, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=[9, 6])
    big_font = {'size':'16'}
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    
    # Plot points
    dfa = df[df.craft.values == 'sta']
    dfb = df[df.craft.values == 'stb']
    ax1.scatter(dfa.time.values, dfa.complexity.values, color='pink',
                label='STEREO-A: All CMEs', zorder=1)
    ax1.scatter(dfb.time.values, dfb.complexity.values, color='lightskyblue',
                label='STEREO-B: All CMEs', zorder=1)
    
    # Plot means and stds
    means = de.find_means(df, 12, err=err)
    ax1.plot(means.dates.values, means.sta_means.values, color='crimson',
             label='STEREO-A: Yearly Means', marker='s', linestyle='dashed',
             zorder=2)
    ax1.errorbar(means.dates.values, means.sta_means.values,
                 yerr=means.sta_s.values, color='crimson', capsize=3,
                 linestyle='None', zorder=2)
    ax1.plot(means.dates.values, means.stb_means.values, color='navy',
             label='STEREO-B: Yearly Means', marker='s', linestyle='dashed',
             zorder=2)
    ax1.errorbar(means.dates.values, means.stb_means.values,
                 yerr=means.stb_s.values, color='navy', capsize=3,
                 linestyle='None', zorder=2)
    
    # Plot sunspots
    sunspots = get_sunspot_record()
    years = []
    sun_means = []
    sun_ste = []
    for year in range(np.min(means.dates).year, np.max(means.dates).year + 1):
        year_start = datetime(year, 1, 1)
        year_end = datetime(year, 12, 31)
        years.append(year_start + timedelta(weeks=26))
        subset = sunspots[sunspots['date'] <= year_end]
        subset = subset[subset['date'] >= year_start]
        mean = np.mean(subset.daily_total_sunspot_n.values)
        sun_means.append(mean)
        s = np.std(subset.daily_total_sunspot_n.values)
        if err == 'sem':
            s = s / np.sqrt(len(subset))
        sun_ste.append(s)
    ax2.plot(sunspots.date, sunspots.daily_total_sunspot_n, color='gold',
             label='Sunspots: Daily Totals', zorder=1)
    ax2.plot(years, sun_means, color='orangered', marker='s', linestyle='dashed',
             label='Sunspots: Yearly Means', zorder=2)
    ax2.errorbar(years, sun_means, yerr=sun_ste, color='orangered', capsize=3,
                 linestyle='None', zorder=2)
    
    # Format plot
    ax_lo = np.min(means.dates) - timedelta(weeks=26)
    ax_hi = np.max(means.dates) + timedelta(weeks=26.5)
    ax1.set_xlim((ax_lo, ax_hi))
    ax2.set_xlim((ax_lo, ax_hi))
    ax2.set_ylim(0, 210)
    ax1.set_ylabel('Relative Complexity', **big_font) 
    ax2.set_ylabel('Total Sunspots', color='k', **big_font)
    ax2.set_xlabel('Time', **big_font)
    ax1.set_xticks([])
    ax2.tick_params('y', colors='k') 
    ax1.legend(loc=0)
    ax2.legend(loc=0)
    
    # Make pretty
    plt.tight_layout()
    f.subplots_adjust(hspace=0)
    
    # Save
    project_dirs = de.get_project_dirs()
    savename = os.path.join(project_dirs['figures'], 'means_points_sunspots')
    plt.savefig(savename)


def means_points_and_sunspots(df, t, means='independent'):
    # Get time complexity data and means
    dfa = df[df.craft.values == 'sta']
    dfb = df[df.craft.values == 'stb']
    if means == 'independent':
        means = de.find_means(df, t)
        mlabel = ''
        lstyle = 'dashed'
    elif means == 'running':
        means = de.find_running_means(df, t)
        mlabel = ' running'
        lstyle = 'None'
    project_dirs = de.get_project_dirs()
    sunspots = get_sunspot_record(project_dirs['sunspots'])
    # Create the plot
    f, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=[9, 6])
    big_font = {'size':'16'}
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    # Plot points
    ax1.scatter(dfa.time.values, dfa.complexity.values, color='pink',
                label='All STEREO-A storms')
    ax1.scatter(dfb.time.values, dfb.complexity.values, color='lightskyblue',
                label='All STEREO-B storms')
    # Plot means and stds
    ax1.plot(means.dates.values, means.sta_means.values, color='crimson',
             label='STEREO-A '+str(t)+' monthly'+mlabel+' means', marker='s',
             linestyle=lstyle)
    ax1.errorbar(means.dates.values, means.sta_means.values,
                 yerr=means.sta_s.values, color='crimson', capsize=3,
                 linestyle='None')
    ax1.plot(means.dates.values, means.stb_means.values, color='navy',
             label='STEREO-B '+str(t)+' monthly'+mlabel+' means', marker='s',
             linestyle=lstyle)
    ax1.errorbar(means.dates.values, means.stb_means.values, 
                 yerr=means.stb_s.values, color='navy', capsize=3,
                 linestyle='None')
    ax1.set_ylabel('Relative Complexity', **big_font) 
    ax1.legend(loc=0)
    # Plot sunspots
    ax2.plot(sunspots.date, sunspots.daily_total_sunspot_n, color='lightgrey')
    ax2.set_xlim(means.dates[0] - timedelta(weeks=26),
                 means.dates[len(means.dates)-1] + timedelta(weeks=26))
    ax2.set_ylim(0, 210)
    ax2.set_ylabel('Total Sunspots', color='grey', **big_font)
    ax2.tick_params('y', colors='grey')  
    ax2.set_xlabel('Time', **big_font)
    # Make pretty
    plt.tight_layout()
    f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    # Save
    savename = os.path.join(project_dirs['figures'],
                            str(t) +' monthly'+mlabel+' means points and sunspots')
    plt.savefig(savename)


###############################################################################
# Histograms of complexity values
def complexity_histogram(df):
    # Get data
    dfa = df[df.craft.values == 'sta']
    dfb = df[df.craft.values == 'stb']
    # Plot
    plt.figure(figsize=[6, 4])
    big_font = {'size':'16'}
    plt.hist(np.append(dfa.complexity.values, dfb.complexity.values),
             bins=np.arange(-5, 4, 0.5), color='indigo')
    plt.xlabel('Relative Complexity', **big_font)
    plt.ylabel('Frequency', **big_font)
    # Save
    project_dirs = de.get_project_dirs()
    savename = os.path.join(project_dirs['figures'],
                            'complexity_histogram_all')
    plt.savefig(savename)


def complexity_histogram_years(df):
    # Get data
    dfa = df[df.craft.values == 'sta']
    dfb = df[df.craft.values == 'stb']
    means = de.find_means(df, 12)
    # Loop over years
    years2 = []
    for j in range(len(means)):
        years2.append(means.dates[j].year)
    for y in years2:
        # Extract points from this year
        sta_points = []
        for i in dfa.index:
            if dfa.time[i].year == y:
                sta_points.append(dfa.complexity[i])
        stb_points = []
        for i in dfb.index:
            if dfb.time[i].year == y:
                stb_points.append(dfb.complexity[i])
        # Plot
        plt.figure(figsize=[6, 4])
        big_font = {'size':'16'}
        plt.hist(np.append(sta_points, stb_points), bins=np.arange(-5, 4, 0.5),
                 color='indigo', label=str(y))
        plt.xlabel('Relative Complexity', **big_font)
        plt.ylabel('Frequency', **big_font)
        plt.legend()
        # Save
        project_dirs = de.get_project_dirs()
        savename = os.path.join(project_dirs['figures'],
                                'complexity_histogram_' + str(y))
        plt.savefig(savename)
        
    
def complexity_histogram_split(df, split):
    """Split should be a year"""
    # Get data
    dfa = df[df.craft.values == 'sta']
    dfb = df[df.craft.values == 'stb']
    # Split points into two groups
    sta_points1 = []
    sta_points2 = []
    for i in dfa.index:
        if dfa.time[i].year < split:
            sta_points1.append(dfa.complexity[i])
        else:
            sta_points2.append(dfa.complexity[i])
    stb_points1 = []
    stb_points2 = []
    for i in dfb.index:
        if dfb.time[i].year < split:
            stb_points1.append(dfb.complexity[i])
        else:
            stb_points2.append(dfb.complexity[i])
    # Plot
    plt.figure(figsize=[6, 4])
    big_font = {'size':'16'}
    plt.hist(stb_points1, bins=np.arange(-5, 4, 0.5), alpha=0.5, color='y',
             label='Before '+str(split))
    plt.hist(stb_points2, bins=np.arange(-5, 4, 0.5), alpha=0.5, color='g',
             label='After '+str(split))
    plt.xlabel('Relative Complexity', **big_font)
    plt.ylabel('Frequency', **big_font)
    plt.legend()
    # Save
    project_dirs = de.get_project_dirs()
    savename = os.path.join(project_dirs['figures'],
                            'complexity_histogram_split_' + str(split))
    plt.savefig(savename)
        
    
###############################################################################
def plot_x_vs_y(df, col_x, col_y, label_x, label_y, lim_x=False, lim_y=False,
                corr=False):
    """
    :param: corr: prints correlation coefficients on the image. This should
    be [X,Y] coords for top left corner of text.
    """
    # Get data
    dfa = df[df.craft.values == 'sta']
    dfb = df[df.craft.values == 'stb']
    # Plot    
    plt.figure(figsize=[8, 5])
    big_font = {'size':'16'}
    plt.scatter(dfa[col_x].values, dfa[col_y].values, color='pink',
                label='STEREO-A')
    plt.scatter(dfb[col_x].values, dfb[col_y].values, color='lightskyblue',
                label='STEREO-B')
    if lim_x != False:
        plt.xlim(lim_x)
    if lim_y != False:
        plt.ylim(lim_y)
    plt.xlabel(label_x, **big_font)
    plt.ylabel(label_y, **big_font)
    plt.legend(loc=0)
    # add correlation coefficient, if required
    if corr != False:
        new_line = (np.nanmax(df[col_y]) - np.nanmin(df[col_y]))/15
        a_corr, a_pval = sps.spearmanr(dfa[col_x].values, dfa[col_y].values,
                                       nan_policy='omit')
        b_corr, b_pval = sps.spearmanr(dfb[col_x].values, dfb[col_y].values,
                                       nan_policy='omit')
        plt.text(corr[0], corr[1], 'Spearman Rank Corr. Coeffs.')
        plt.text(corr[0], corr[1]-new_line, '         STEREO-A: ' + str('{:+.2f}'.format(round(a_corr, 2))))
        plt.text(corr[0], corr[1]-2*new_line, '         STEREO-B: ' + str('{:+.2f}'.format(round(b_corr, 2))))
        print('STEREO-A: corr %s p-val %s'%(a_corr, a_pval))
        print('STEREO-B: corr %s p-val %s'%(b_corr, b_pval))
    # Save
    project_dirs = de.get_project_dirs()
    savename = os.path.join(project_dirs['figures'], col_x + ' vs ' + col_y)
    plt.savefig(savename)  


# additions for time vs complexity plot
def plot_time_vs_complexity(df, lim_x=False, lim_y=False, corr=False, means=['independent', 12]):
    """
    """
    # get the plot
    plot_x_vs_y(df, 'time', 'complexity', 'Year', 'Relative Complexity', lim_x=lim_x, lim_y=lim_y, corr=corr)
    # limit the time axis
    plt.xlim((datetime(2008, 1, 1), datetime(2017, 1, 1)))
    # Add means to the plot
    if means[0] == 'independent':
        mdf = de.find_means(df, means[1])
    elif means[0] == 'running':
        # NB RUNNING MEANS DOES NOT WORK
        mdf = de.find_running_means(df, means[1])
    if means == 12:
        label_a = 'STEREO-A annual means'
        label_b = 'STEREO-B annual means'
    else:
        label_a = 'STEREO-A ' + str(means[1]) + '-month means'
        label_b = 'STEREO-B ' + str(means[1]) + '-month means'
    print(mdf.sta_s.values)
    plt.plot(mdf.dates.values, mdf.sta_means.values, color='crimson', label=label_a, marker='s', linestyle='dashed')
    plt.plot(mdf.dates.values, mdf.stb_means.values, color='navy', label=label_b, marker='s', linestyle='dashed')
    plt.errorbar(mdf.dates.values, mdf.sta_means.values, yerr=mdf.sta_s.values, color='crimson', capsize=3, linestyle='None')
    plt.errorbar(mdf.dates.values, mdf.stb_means.values, yerr=mdf.stb_s.values, color='navy', capsize=3, linestyle='None')
    plt.legend(loc=0)
    

# Shortcut functions for using plot_x_vs_y without writing the same labels all the time
def plot_vs_complexity(df, col_x, label_x, lim_x=False, lim_y=False, corr=False):
    """
    """
    plot_x_vs_y(df, col_x, 'complexity', label_x, 'Relative Complexity', lim_x=lim_x, lim_y=lim_y, corr=corr)


def plot_vs_helcats(df, h_name, cat, col_x, col_y, label_x, label_y, lim_x=False, lim_y=False, corr=False):
    """
    """
    df = de.add_helcats_to_df(df, h_name, cat)
    # determine whether helcats is x or y
    if col_x == h_name:
        plot_x_vs_y(df, h_name, col_y, label_x, label_y, lim_x=lim_x, lim_y=lim_y, corr=corr)
    elif col_y == h_name:
        plot_x_vs_y(df, col_x, h_name, label_x, label_y, lim_x=lim_x, lim_y=lim_y, corr=corr)


def plot_helcats_vs_complexity(df, h_name, cat, label_x, lim_x=False, lim_y=False, corr=False):
    """Plots a HELCATS parameter against complexity.
    """
    df = de.add_helcats_to_df(df, h_name, cat)
    plot_vs_complexity(df, h_name, label_x, lim_x=lim_x, lim_y=lim_y, corr=corr)
    
    
def plot_helcats_vs_time(df, h_name, cat, label_y, lim_x=False, lim_y=False, corr=False):
    df = de.add_helcats_to_df(df, h_name, cat)
    plot_x_vs_y(df, 'time', h_name, 'Year', label_y, lim_x=lim_x, lim_y=lim_y, corr=corr)


###############################################################################
# Plots to compare properties of CMEs observed by both STEREO-A and STEREO-B
# These automatically drop any CMEs not observed by both
def plot_a_vs_b(df, col, label):
    # get data
    df = de.add_matches_to_df(df)
    df = df.dropna(subset=['match'])
    a_vals, b_vals = de.get_match_vals(df, col)
    # plot
    plt.figure(figsize=[8, 8])
    big_font = {'size':'16'}
    plt.scatter(a_vals, b_vals)
    plt.xlabel('STEREO-A ' + label, **big_font)
    plt.ylabel('STEREO-B ' + label, **big_font)
    # Add 1:1 line
    plt.plot((plt.gca().get_xlim()), (plt.gca().get_ylim()), ls="--", c=".3")
    # Save
    project_dirs = de.get_project_dirs()
    savename = os.path.join(project_dirs['figures'], label + 'A vs B')
    plt.savefig(savename)       


def plot_a_b_diff_vs_time(df, col, label):
    # get data
    df = de.add_matches_to_df(df)
    df = df.dropna(subset=['match'])
    a_vals, b_vals = de.get_match_vals(df, col)
    a_times, b_times = de.get_match_vals(df, 'time')
    diffs = np.array(a_vals) - np.array(b_vals)
    # Make plot
    plt.figure(figsize=[8, 5])
    big_font = {'size':'16'}
    plt.scatter(a_times, diffs)
    plt.xlabel('Time', **big_font)
    plt.ylabel(label + ' A-B', **big_font)
    # Add zero y line
    plt.axhline(y=0, ls='--', color='.3')
    # Save
    project_dirs = de.get_project_dirs()
    savename = os.path.join(project_dirs['figures'], label + 'A-B over time')
    plt.savefig(savename)
    
    
def plot_a_b_diff_vs_a_b_diff(df, col_x, col_y, label_x, label_y):
    # get data
    df = de.add_matches_to_df(df)
    df = df.dropna(subset=['match'])
    a_vals_x, b_vals_x = de.get_match_vals(df, col_x)
    a_vals_y, b_vals_y = de.get_match_vals(df, col_y)
    x_diffs = np.array(a_vals_x) - np.array(b_vals_x)
    y_diffs = np.array(a_vals_y) - np.array(b_vals_y)
    # Make plot
    plt.figure(figsize=[8, 5])
    big_font = {'size':'16'}
    plt.scatter(x_diffs, y_diffs)
    plt.xlabel(label_x + ' A-B', **big_font)
    plt.ylabel(label_y + ' A-B', **big_font)
    # Add zero x and y lines
    plt.axvline(x=0, ls='--', color='.3')
    plt.axhline(y=0, ls='--', color='.3')
    # Save
    project_dirs = de.get_project_dirs()
    savename = os.path.join(project_dirs['figures'], label_x + ' diff vs ' + label_y + ' diff')
    plt.savefig(savename)
    
    
############################################################################### 
def plot_nan_count(df):
    df = de.add_image_nans_to_df(df)
    df = df[pd.notnull(df['nan_count'])]
    plt.hist(df.nan_count, bins=np.arange(0, 40000, 500))
    plt.xlabel('NaN Count')
    plt.ylabel('Frequency')


def plot_wins_vs_diff(pc, diff=0.1):
    """pc = paired comparisons pandas df.
    """
    diffs = []
    draws = []
    left_wins= []
    right_wins = []
    ns = []
    for diff in np.arange(round(np.min(pc.complexity_diff), 2), round(np.max(pc.complexity_diff), 2), 0.01):
        subset_low = pc[pc.complexity_diff < diff]
        subset = subset_low[subset_low.complexity_diff > diff-0.01]
        n = len(subset)
        if n > 0:
            ns.append(n)
            diffs.append(diff)
            draws.append(100*len(subset[subset.left_wins == (subset.total_votes/2)])/n)
            left_wins.append(100*len(subset[subset.left_wins > (subset.total_votes/2)])/n)
            right_wins.append(100*len(subset[subset.right_wins > (subset.total_votes/2)])/n)      
    fig, ax1 = plt.subplots()
    ax1.plot(diffs, ns, color='pink', alpha=0.5)
    ax1.set_ylabel('# comparisons', color='pink')
    ax2 = ax1.twinx()
    ax2.plot(diffs, left_wins, color='blue', label='Left Wins')
    ax2.plot(diffs, right_wins, color='red', label = 'Right Wins')
    ax2.plot(diffs, draws, color='purple', label='Draws')
    ax2.set_ylabel('% of Events')
    ax1.set_xlabel('Left-Right Complexity difference')
    plt.legend(loc=0)
    
    subset = pc[pc.complexity_diff < diff]
    n = len(subset)
    draws = len(subset[subset.left_wins == (subset.total_votes/2)]) 
    left_wins = len(subset[subset.left_wins > (subset.total_votes/2)])
    right_wins = len(subset[subset.right_wins > (subset.total_votes/2)])
    print('% of total', n/len(pc))
    print('n', n)
    print('draws %', draws/n)
    print('left wins %', left_wins/n)
    print('right wins %', right_wins/n)
    

def plot_wins_per_year(dfpc):
    # remove comparisons where both are from the same year
    dfpc = dfpc[dfpc.left_year != dfpc.right_year]
    years = []
    per_a = []
    per_b = []
    for year in range(2008, 2017, 1):
        pc_year_left = dfpc[dfpc.left_year == year]
        wins_left = pc_year_left[pc_year_left.winner == 'left']
        wins_left_a = wins_left[wins_left.left_craft == 'sta']
        wins_left_b = wins_left[wins_left.left_craft == 'stb']
        
        pc_year_right = dfpc[dfpc.right_year == year]
        wins_right = pc_year_right[pc_year_right.winner == 'right']
        wins_right_a = wins_right[wins_right.right_craft == 'sta']
        wins_right_b = wins_right[wins_right.right_craft == 'stb']
        
        wins_a = len(wins_left_a) + len(wins_right_a)
        wins_b = len(wins_left_b) + len(wins_right_b)
        
        year_total = len(pc_year_left) + len(pc_year_right)
        
        percent_win_a = 100 * (wins_a / year_total)
        percent_win_b = 100 * (wins_b / year_total)
        years.append(year)
        per_a.append(percent_win_a)
        per_b.append(percent_win_b)
    
    plt.figure(figsize=[8, 5])
    plt.bar(years, per_a, color='pink', label='STEREO-A')
    plt.bar(years, per_b, color='lightskyblue', bottom=per_a, label='STEREO-B')
    plt.xlabel('Year')
    plt.ylabel('Percentage')
    plt.xticks(range(2008, 2017, 1))
    plt.legend(loc=0)
    # Save
    project_dirs = de.get_project_dirs()
    savename = os.path.join(project_dirs['figures'], 'wins_per_year')
    plt.savefig(savename) 

###############################################################################
def corset_boxplot(df):
    fig, ax = plt.subplots(figsize=(14, 10))
    data = []
    labels = []
    nums = []
    # add totals on the end
    for craft in ['sta', 'stb']:
        dfc = df[df.craft == craft]
        data.append(dfc.complexity)
        nums.append(str(len(dfc)))
        labels.append('All Events')
    for m in np.unique(df.morphology):
        dfs = df[df.morphology == m]
        for craft in ['sta', 'stb']:
            dfc = dfs[dfs.craft == craft]
            if len(dfc) > 1:
                data.append(dfc.complexity)
                nums.append(str(len(dfc)))
                labels.append(str(m))
    bp = ax.boxplot(data, medianprops=dict(color='k'), widths=0.6)
    plt.xlabel("Event type in COR2, identified by CORSET", fontsize=16)
    plt.ylabel("Relative Visual Complexity", fontsize=16)
    xticks = np.unique(labels)
    lowlim = ax.get_xlim()[0]
    hilim = ax.get_xlim()[1] 
    spacing = (hilim - lowlim) / len(xticks)
    ax.set_xticklabels(xticks, fontsize=16)
    ax.set_xticks(np.arange(lowlim + spacing/2, hilim, spacing))
    plt.yticks(fontsize=16)  
    spacing2 = (hilim - lowlim) / len(data)
    locs = np.arange(lowlim + spacing2/2, hilim, spacing2)
    box_colors = ['pink', 'lightskyblue']
    box_colors2 = ['crimson', 'navy']
    for n in range(len(data)):
        if n < 2:
            color = 'black'
        else:
            color = 'black'
        ax.text(locs[n], bp['medians'][n]._y[0]+0.1, nums[n], fontsize=16,
                ha='center', color=color)
        # if n < 2:
        #     # change colours
        #     for detail in ['boxes', 'medians']:
        #         plt.setp(bp[detail][n], color=box_colors2[n % 2])
        #     for detail in ['caps', 'whiskers', 'fliers']:
        #         plt.setp(bp[detail][2*n:2*n+2], color=box_colors2[n % 2])
    print(lowlim, spacing)
    ax.axvline(lowlim+spacing, color='k', ls='-', lw=1)
    # ax.axvline(lowlim+spacing, color='lightgrey', ls='--')
    # Now fill the boxes with desired color
    num_boxes = len(data)
    for i in range(num_boxes):
        box = bp['boxes'][i]
        boxX = []
        boxY = []
        for j in range(5):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
        box_coords = np.column_stack([boxX, boxY])
        # Alternate between colours
        # if i < 2:
        #     ax.add_patch(Polygon(box_coords, facecolor=box_colors2[i % 2]))
        # else:
        ax.add_patch(Polygon(box_coords, facecolor=box_colors[i % 2]))
    sta_patch = mpatches.Patch(facecolor='pink', label='STEREO-A',
                               edgecolor='k')
    stb_patch = mpatches.Patch(facecolor='lightskyblue', label='STEREO-B',
                               edgecolor='k')
    plt.legend(handles=[sta_patch, stb_patch], loc=0, frameon=False,
               fontsize=16)
    
    # Save
    project_dirs = de.get_project_dirs()
    savename = os.path.join(project_dirs['figures'], 'Complexity vs CORSET Morphology')
    plt.savefig(savename)     
    
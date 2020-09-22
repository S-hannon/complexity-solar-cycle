from __future__ import division
import os
import sys
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


sys.path.insert(0, r'C:\\Users\\shann\\OneDrive\\Documents\\Research\\Workspace\\Code\\useful_code')
import misc


class STEREOPlot:
    def __init__(self, data_loc, fig_loc):
        self.data_loc = data_loc
        self.fig_loc = fig_loc
        self.sta_name = 'STEREO-A'
        self.stb_name = 'STEREO-B'
        self.sta_colour = 'pink'
        self.stb_colour = 'lightskyblue'
        self.sta_means_colour = 'crimson'
        self.stb_means_colour = 'navy'
        self.figsize = [14, 9]
        self.small_font = 12
        self.big_font = 20
    

    def add_corr_text(self, ax, df, col_x, col_y, corr):
        dfa = df[df.craft.values == 'sta']
        dfb = df[df.craft.values == 'stb']
        new_line = (np.nanmax(df[col_y]) - np.nanmin(df[col_y]))/15
        a_corr, a_pval = sps.spearmanr(dfa[col_x].values, dfa[col_y].values,
                                       nan_policy='omit')
        b_corr, b_pval = sps.spearmanr(dfb[col_x].values, dfb[col_y].values,
                                       nan_policy='omit')
        ax.text(corr[0], corr[1], 'Spearman Rank Corr. Coeffs.',
                fontsize=16)
        ax.text(corr[0], corr[1]-new_line,
                '         STEREO-A: ' + str('{:+.2f}'.format(round(a_corr, 2))),
                fontsize=16)
        ax.text(corr[0], corr[1]-2*new_line,
                '         STEREO-B: ' + str('{:+.2f}'.format(round(b_corr, 2))),
                fontsize=16)
        return ax

    
    def scatter_x_vs_y(self, ax, df, col_x, col_y):
        for craft in ['sta', 'stb']:
            dfc = df[df.craft.values == craft]
            ax.scatter(dfc[col_x].values, dfc[col_y].values,
                       color=getattr(self, craft + '_colour'),
                       label=getattr(self, craft + '_name'))
        return ax
    
    
    def scatter_x_vs_y_with_z_markers(self, ax, df, col_x, col_y, col_z):
        markers = ['^', 'x', '*', 'o', 'd', '1', '2', '3']       
        for n, z in enumerate(np.unique(df[col_z])):
            if z != 'nan':
                for craft in ['sta', 'stb']:
                    dfc = df[df.craft.values == craft]
                    dfc_z = dfc[dfc[col_z] == z]
                    ax.scatter(dfc_z[col_x].values, dfc_z[col_y].values,
                               color=getattr(self, craft + '_colour'),
                               label=getattr(self, craft + '_name') + ' ' + z,
                               marker=markers[n])
        return ax


    def add_time_means(self, ax, df, col, means_type='independent', means_t=12,
                       err='std'):
        for craft in ['sta', 'stb']:
            dfc = df[df.craft.values == craft]
            if means_type == 'independent':
                dates, means, errs = misc.find_time_means(dfc[col].values,
                                                          dfc['time'],
                                                          t=means_t, err=err)
            elif means_type == 'running':
                # NB RUNNING MEANS DOES NOT WORK
                dates, means, errs = misc.find_time_running_means(dfc[col].values,
                                                                  dfc['time'],
                                                                  t=means_t,
                                                                  err=err)
            if means_t == 12:
                label = getattr(self, craft + '_name') + ' yearly means'
            else:
                label = getattr(self, craft + '_name') + ' ' + str(means[1]) + '-month means'   
            ax.plot(dates, means,
                    color=getattr(self, craft + '_means_colour'),
                    label=label, marker='s', linestyle='dashed')
            ax.errorbar(dates, means, yerr=errs,
                        color=getattr(self, craft + '_means_colour'),
                        capsize=3, linestyle='None')
        ax.legend(loc=0, fontsize=12)
        return ax
        
        
    def plot_x_vs_y_ax(self, ax, df, col_x, col_y, label_x, label_y, col_z=None,
                       lim_x=False, lim_y=False, oneone=False, corr=False):
        """
        :param: corr: prints correlation coefficients on the image. This should
        be [X,Y] coords for top left corner of text.
        add_discrete: changes the markers according to data
        """
        for col in ['craft', col_x, col_y]:
            if col not in df.columns:
                raise ValueError("no " + col + " column in df")
        if col_z != None:
            ax = self.scatter_x_vs_y_with_z_markers(ax, df, col_x, col_y,
                                                    col_z)
        else:
            ax = self.scatter_x_vs_y(ax, df, col_x, col_y)
        ax.tick_params(axis="x", labelsize=self.small_font)
        ax.tick_params(axis="y", labelsize=self.small_font)
        if lim_x:
            ax.set_xlim(lim_x)
        if lim_y:
            ax.set_ylim(lim_y)
        if oneone:
            ax.plot((plt.gca().get_xlim()), (plt.gca().get_ylim()), ls="--",
                    c=".3")
        if corr:
            ax = self.add_corr_text(ax, df, col_x, col_y, corr)
        ax.legend(loc=0, fontsize=self.small_font)
        # plt.savefig(os.path.join(self.fig_loc, col_x + ' vs ' + col_y))
        return ax


    def plot_x_vs_y(self, df, col_x, col_y, label_x, label_y, col_z=None,
                    lim_x=False, lim_y=False, oneone=False, corr=False):
        fig, ax = plt.subplots(figsize=self.figsize)
        ax = self.plot_x_vs_y_ax(ax, df, col_x, col_y, label_x, label_y,
                                 col_z=col_z, lim_x=lim_x, lim_y=lim_y,
                                 oneone=oneone, corr=corr)
        ax.set_xlabel(label_x, fontsize=self.big_font)
        ax.set_ylabel(label_y, fontsize=self.big_font)
        plt.savefig(os.path.join(self.fig_loc, col_x + ' vs ' + col_y))


    def __extract_boxplot_vals(self, df, col_x, col_y):
        data = []
        labels = []
        nums = []
        for craft in ['sta', 'stb']:
            dfc = df[df.craft == craft]
            # append average values
            data.append(dfc[col_y])
            nums.append(str(len(dfc)))
            labels.append('All Events')
            # append data for each col_x category
            for m in np.unique(df[col_x]):
                dfcm = dfc[dfc[col_x] == m]
                if len(dfcm) > 1:
                    data.append(dfc[col_y])
                    nums.append(str(len(dfcm)))
                    labels.append(str(m))
        return data, labels, nums        
        

    def boxplot(self, df, col_x, col_y, label_x, label_y, medians=False):
        """
        median: bool, if True plot median values onto median line
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        data, labels, nums = self.__extract_boxplot_vals(self, df, col_x, col_y)
        bp = ax.boxplot(data, medianprops=dict(color='k'), widths=0.6)
        # format y-axis
        ax.tick_params(axis="y", labelsize=self.small_font)
        plt.ylabel(label_y, fontsize=self.big_font)
        # sort x-axis
        plt.xlabel(label_x, fontsize=self.big_font)
        xticks = np.unique(labels)
        lowlim = ax.get_xlim()[0]
        hilim = ax.get_xlim()[1] 
        spacing = (hilim - lowlim) / len(xticks)
        ax.set_xticklabels(xticks, fontsize=16)
        ax.set_xticks(np.arange(lowlim + spacing/2, hilim, spacing))
        if medians:
            spacing2 = (hilim - lowlim) / len(data)
            locs = np.arange(lowlim + spacing2/2, hilim, spacing2)
            for n in range(len(data)):
                if n < 2:
                    color = 'black'
                else:
                    color = 'black'
                ax.text(locs[n], bp['medians'][n]._y[0]+0.1, nums[n],
                        fontsize=16, ha='center', color=color)
        # add line to separate average from individual columns
        ax.axvline(lowlim+spacing, color='k', ls='-', lw=1)
        # Now fill the boxes with desired color
        box_colors = [self.sta_colour, self.stb_colour]
        num_boxes = len(data)
        for i in range(num_boxes):
            box = bp['boxes'][i]
            boxX = []
            boxY = []
            for j in range(5):
                boxX.append(box.get_xdata()[j])
                boxY.append(box.get_ydata()[j])
            box_coords = np.column_stack([boxX, boxY])
            ax.add_patch(Polygon(box_coords, facecolor=box_colors[i % 2]))
        patches = []
        for craft in ['sta', 'stb']:
            patch = mpatches.Patch(facecolor=getattr(self, craft + '_colour'),
                                   label=getattr(self, craft + '_name'),
                                   edgecolor='k')
            patches.append(patch)
        plt.legend(handles=patches, loc=0, frameon=False,
                   fontsize=self.small_font)       
        plt.savefig(os.path.join(self.fig_loc,
                                 'boxplot complexity vs ' + col_x))  


    def histogram(self, ax, sta_vals, stb_vals, label):
        vals = dict({'sta' : sta_vals, 'stb' : stb_vals})
        for craft in ['sta', 'stb']:
            ax.hist(vals[craft], bins=np.arange(-5, 4, 0.5), alpha=0.5,
                    label=getattr(self, craft + '_name'),
                    color=getattr(self, craft + '_colour'))
        ax.set_xlabel(label, fontsize=16)
        ax.set_ylabel('Frequency', fontsize=16)
        ax.legend(loc=0, frameon=False, fontsize=12)
        return ax


    def plot_a_vs_b(self, ax, sta_vals, stb_vals, label):
        ax.scatter(sta_vals, stb_vals)
        ax.set_xlabel(self.sta_name + ' ' + label, fontsize=16)
        ax.set_ylabel(self.stb_name + ' ' + label, fontsize=16)
        # Add 1:1 line
        ax.plot((plt.gca().get_xlim()), (plt.gca().get_ylim()), ls="--", c=".3")
        # plt.savefig(os.path.join(self.fig_loc, label + 'A vs B'))
        return ax
        
    
    def __find_ratio(self, sta_vals, stb_vals):
        ratios = []
        for i in range(len(sta_vals)):
            ratios.append(sta_vals[i] / stb_vals[i])
        return ratios
    
    
    def plot_ab_ratio_dist(self, ax, sta_vals, stb_vals, label):
        ratios = self.__find_ratio(sta_vals, stb_vals)
        ax.hist(ratios)
        ax.set_xlabel(self.sta_name + ' / ' + self.stb_name + ' ' + label,
                      fontsize=self.fontsize)
        ax.set_ylabel('Frequency', fontsize=self.fontsize)
        # plt.savefig(os.path.join(self.fig_loc, label + 'AB ratio hist'))
        return ax
    
    
    def __sort_deg_diff(diffs):
        # convert to 0-180 not 0-360
        new_diffs = []
        for n, i in enumerate(diffs):
            if abs(i) > 180:
                new_diffs.append(360 - abs(i))
            else:
                new_diffs.append(i)
        return new_diffs
    
    
    def __find_ab_diff(self, sta_vals, stb_vals, abs_col=False, deg=False):
        diffs = np.array(sta_vals) - np.array(stb_vals)
        if abs_col == True:
            diffs = abs(diffs)
        if deg == True:
            diffs = self.__sort_deg_diff(diffs)
        return diffs
    
    
    def plot_ab_diff_dist(self, ax, sta_vals, stb_vals, label, abs_col=False,
                          deg=False, bins=70):
        diffs = self.__find_ab_diff(sta_vals, stb_vals, abs_col=abs_col,
                                    deg=deg)
        ax.hist(diffs, bins=bins)
        ax.set_xlabel(self.sta_name + ' - ' + self.stb_name + ' ' + label,
                      fontsize=self.fontsize)
        ax.set_ylabel('Frequency', fontsize=self.fontsize)
        # ax.savefig(os.path.join(self.fig_loc, label + 'A-B hist'))
        return ax
        
    
    def plot_ab_diff_vs_x(self, ax, sta_vals, stb_vals, x_vals,
                          label_x, label_y, abs_col=False, deg=False):
        diffs = self.__find_ab_diff(sta_vals, stb_vals, abs_col=abs_col,
                                    deg=deg)
        ax.scatter(diffs, x_vals)
        ax.set_xlabel(label_x, fontsize=self.fontsize)
        ax.set_ylabel(label_y + ' A-B', fontsize=self.fontsize)
        ax.axhline(y=0, ls='--', color='.3')
        # plt.savefig(os.path.join(self.fig_loc, label_x + 'A-B vs ' + label_y))
        return ax
        
        
    def plot_ab_diff_vs_ab_diff(self, ax,
                                sta_vals_x, stb_vals_x, label_x,
                                sta_vals_y, stb_vals_y, label_y,
                                abs_x=False, abs_y=False,
                                deg_x=False, deg_y=False):
        diffs_x = self.__find_ab_diff(sta_vals_x, stb_vals_x, abs_col=abs_x,
                                      deg=deg_x)
        diffs_y = self.__find_ab_diff(sta_vals_y, stb_vals_y, abs_col=abs_y,
                                      deg=deg_y)
        if abs_x == False:
            plt.axvline(x=0, ls='--', color='.3')
        if abs_y == False:
            plt.axhline(y=0, ls='--', color='.3')
        ax.scatter(diffs_x, diffs_y)
        ax.set_xlabel(label_x + ' A-B', fontsize=self.fontsize)
        ax.set_ylabel(label_y + ' A-B', fontsize=self.fontsize)
        # plt.savefig(os.path.join(self.fig_loc, label_x + ' A-B vs ' + label_y + ' A-B'))
        return ax
    
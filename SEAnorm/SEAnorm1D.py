# -*- coding: utf-8 -*-
"""

This module supports a 1D (e.g., time) and 2D (e.g., time and space)
normalized superposed epoch analysis of time series data stored in a 
Pandas DataFrame. 

For the time normalization to work the DataFrame index must
be a datetime index. 

Each event is seperated into two phases, phase 1 and phase 2,
seperated by 3 epochs, t0, t1, t2 (or start, epoch, end).

Each phase is normalized from 0-1 depending on the passed x_dimensions.

The normalized time is then used to bin the data (DataFrame columns) and
calculate typical statistics for each bin (median, mean, upper and lower
quartile, and counts) for each phase. 

If performing a 2D analysis, then one of the columns of the DataFrame must
be a second axis to bin along. The column name must be passed as y_col and
binning parameters must be passed as a list with the min and max value and
spacing to define the bin edges.

The data is then returned as a normalized superposed epoch time series along
with a dictionary of metadata that can be used for plotting and reference.

"""

from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy import stats
import gc

import matplotlib.pylab as plt
import timeit

def SEAnorm1D(data, events, x_dimensions, cols=False, seastats=False,
              y_col=False, y_dimensions=False):
    
    
    
    
    
    """
    Performs a normalized superposed epoch analysis of the time series
    contained in a DataFrame

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    events : TYPE
        DESCRIPTION.
    x_dimensions : TYPE
        DESCRIPTION.
    cols : TYPE, optional
        DESCRIPTION. The default is False.
    seastats : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    SEAdat : TYPE
        DESCRIPTION.
    cols : TYPE
        DESCRIPTION.


    
    Parameters:
    -----------
    data - Pandas DataFrame containing the time series to be used.
         - Must have a Pandas datetime index
    events - list of three arrays [t0, t1, t2] containing the start (t0), 
        epoch (t1) and end times (t2) of each event
           - times must either be timestamps or strings 
        in the format: 'YYYY-MM-DD HH:MM:SS'
           - phase 1 is defined from t0->t1 (start to epoch)
           - phase 2 is defined from t1->t2 (epoch to end)
    x_dimensions - list [x, y] containing two elements: the desired number of 
        normalised time bins in [phase 1, phase 2]
    cols - list of column names to run the superposed epoch analysis on
    seastats - dictionary ofstatistics to be used for superposed epoch analysis 
         via the scipy.binned_statistic function. 
             - default is mean, median, upper and lower quartile 
             - stat_function can be a string, e.g. as defined in 
         scipy.stats. binned_statistic, a callable, e.g., np.nanmean, 
         or a lambda defined callable e.g., the 90th percental
         p90 = lambda stat: np.nanpercentile(stat, 90)
         
         To call all three the dictionary could be organized as 
         
         stats = {'mean':'mean','namean':np.mean,'p90':p90}
         
         Recommended to use numpy function as they can handle NaN better then
         the builtin scipy.stats.binned_statistic built in statistics
        

    Returns:
    --------
    Pandas DataFrame containing the final time-normalised superposed epoch analysis.
    """

    # get the required epochs from the event list    
    starts, epochs, ends = events
    
    # determine the spacing in normalized time for both phases
    # each phase is normalized to 1 and then binned based on the
    # spacing and bin sizes defined by x_dimensions
    x1_spacing, x2_spacing = 1/x_dimensions[0], 1/x_dimensions[1]

    # if a series is passed convert it to a data frame for simplicity 
    if isinstance(data, pd.Series):
        se_data=data.to_frame('data')
        cols = 'data'
    elif cols:
    # determine what columns we're keeping for analysis
    # if y_col is defined then append the y data for 
    # 2D analysis to the data frame
        col_dat = list(cols)
        if y_col and y_dimensions:
            col_dat.append(y_col)

        
        # get the sea data from the 
        # passed DataFrame
        # convert to DataFrame if
        # on column is passed
        se_data=data[col_dat].copy()
        if isinstance(se_data, pd.Series):
            se_data=se_data.to_frame(col_dat)
    
    # if cols is False keep them all
    # no need to account for 2D data here        
    else:
        se_data=data.copy()
        cols = se_data.columns.values.tolist()
        y_col = False
        

    # define stats values
    # if stats parameter is passed use that
    # else use predfined stats
    # mean, median, upper and lower quartile
    if seastats and isinstance(seastats, dict):
        stat_vals = seastats
    else:
        lq_nan = lambda stat: np.nanpercentile(stat, 25)
        uq_nan = lambda stat: np.nanpercentile(stat, 75)
    
        stat_vals = {'mean':np.nanmean, 'median':np.nanmedian, 
                     'low_q':lq_nan, 'up_q':uq_nan, 'cnt':'count'}
    
    # number of events for reference later on
    gc.collect()
    
    # create empty data frames to store the normalized time
    # for each phase
    p1data = pd.DataFrame() 
    p2data = pd.DataFrame()
    

    # loop through events normalize time and collect data
    for event in tqdm(range(len(starts))):
        # get the epochs for the event
        start = str(starts.iloc[event])
        epoch = str(epochs.iloc[event])
        end = str(ends.iloc[event])

        # get phase 1 and phase 2 data
        # in seperate dataframes
        phase1 = se_data[start:epoch].copy()
        phase2 = se_data[epoch:end].copy()

        # normalise time axis of phase 1 for each phase from 0 to 1.
        try:
            # reset time for this event to 0
            phase1['t_norm'] = phase1.index - phase1.index[0]   
        except IndexError:
            # in case there is no data during a given event
            print('There is no data for event '+str(event))
            continue
        
        # get time in seconds only, ready to normalise
        phase1['t_norm'] = phase1['t_norm'].dt.total_seconds() 
        # find smallest and largest values (to become 0 and 1)
        # and normalize
        p1min = phase1['t_norm'][0]                            
        p1max = phase1['t_norm'][-1]
        # normalise the time values from 0 to 1
        phase1['t_norm'] = (phase1['t_norm'] - p1min) / (p1max - p1min)  

        # normalise the time axis for phase 2 (same as above)
        try:
            phase2['t_norm'] = phase2.index - phase2.index[0]
        except IndexError:
            continue
        phase2['t_norm'] = phase2['t_norm'].dt.total_seconds()
        p2min = phase2['t_norm'][0]
        p2max = phase2['t_norm'][-1]
        phase2['t_norm'] = ((phase2['t_norm'] - p2min) / (p2max - p2min))

        # append the phase 1 and phase 2 data frames 
        # to final DataFrames which contain all the 
        p1data = pd.concat([p1data,phase1])
        p2data = pd.concat([p2data,phase2])


    # calculate the normalized SEA
    # statistics
    
    # create bins and edges in normalized 
    # time for binning the data in both phases
    x1bins = np.arange(0, 1, x1_spacing)
    x1_edges = np.arange(0, 1 + x1_spacing, x1_spacing)
    x2bins = np.arange(0, 1, x2_spacing)
    x2_edges = np.arange(0, 1 + x2_spacing, x2_spacing)
    
    # if calculating 2D SEA then calculate the y bins
    if y_col and y_dimensions:
        ymin, ymax, y_spacing = y_dimensions
        y_edges = np.arange(ymin, ymax + y_spacing, y_spacing)
        sea2d = True
    else:
        sea2d = False

    
    # create normalized time axis
    t_norm = (x1bins-x1bins.max()-x1_spacing)/x1_spacing
    t_norm = np.concatenate([t_norm,x2bins/x2_spacing])
    
    # create return DataFrame
    SEAdat = pd.DataFrame()
    SEAdat['t_norm'] = t_norm
    
    # convert phase normalized data into
    # a list of lists that can be passed
    # as a single function call to 
    # stat_binned_statistic
    ph1list = [p1data[x] for x in cols]
    ph2list = [p2data[x] for x in cols]
    
    # loop over the stat values that 
    # need to be calculated and calculate
    # them for each column from ph1/ph2list
    for s_name, s_fun in stat_vals.items():
        # 2D superposed epoch analysis
        if sea2d:
            p1stat, _, y1_v, _ = stats.binned_statistic_2d(p1data['t_norm'], 
                                                        p1data[y_col], 
                                                        values=ph1list, 
                                                        bins=[x1_edges, y_edges], 
                                                        statistic=s_fun)
            p2stat, _, y2_v, _ = stats.binned_statistic_2d(p2data['t_norm'], 
                                                        p2data[y_col], 
                                                        values=ph2list, 
                                                        bins=[x2_edges, y_edges], 
                                                        statistic=s_fun)
            #loop over the columns and ybins
            #to fill DataFrame
            for i in np.arange(p1stat.shape[0]):
                for j in np.arange(p1stat.shape[2]):
                    
                    
                    SEAdat[cols[i]+'_'+s_name+f'_{j:03d}'] = \
                       np.concatenate([p1stat[i,:,j],p2stat[i,:,j]], axis=0)
                               
        # 1D superposed epoch analysis
        else:
            p1stat, _, _ = stats.binned_statistic(p1data['t_norm'],
                                              values=ph1list, 
                                              bins=x1_edges, statistic=s_fun)
            p2stat, _, _ = stats.binned_statistic(p2data['t_norm'], 
                                              values=ph2list,
                                              bins=x2_edges, statistic=s_fun)

            # loop over the columns and add the superposed
            # data to the returned DataFrame
            for p1_col, p2_col, c, in zip(p1stat, p2stat, cols):
                
                SEAdat[c+'_'+s_name] = np.concatenate([p1_col,p2_col], axis=0)
            
    
    #set t_norm as the index
    SEAdat = SEAdat.set_index('t_norm')
    
    if isinstance(cols,str):
        cols = [cols]
    
    # pass back y axis information
    if sea2d:
        y_rtn = {'min':ymin, 'max':ymax, 'bin':y_spacing, 'edges':y_edges}
        meta = {'sea_cols':cols, 'stats':stat_vals, 'y_meta':y_rtn}
    else:
        meta = {'sea_cols':cols, 'stats':stat_vals, 'y_meta':False}
    
    return SEAdat, meta

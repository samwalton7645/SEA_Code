# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 08:08:52 2022

@author: krmurph1
"""

from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy import stats
import gc

import matplotlib.pylab as plt

def SEAnorm1D(data, events, statistic, x_dimensions, cols=False):

    """Performs a normalized superposed epoch analysis of the time series
    contained in a a DataFrame
    
    Parameters:
    -----------
    data - Pandas DataFrame containing the time series to be used.
         - Must have a Pandas datetime index
    events - list of three arrays [t0, t1, t2] containing the start (t0), epoch (t1) and end times (t2) of each event
           - times must either be timestamps or strings in the format: 'YYYY-MM-DD HH:MM:SS'
           - phase 1 is defined from t0->t1 (start to epoch)
           - phase 2 is defined from t1->t2 (epoch to end)
    statistic - the statistical function to be used in the SEA (e.g. np.mean, np.median, np.sum, etc.)
              - for percentiles, input an integer only
    x_dimensions - list [x, y] containing two elements: the desired number of normalised time bins in [phase 1, phase 2]
    

    Returns:
    --------
    DataFrame containing the final time-normalised superposed epoch analysis.
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
    # if cols is False keep them all
        se_data=data[cols].copy()
    else:
        se_data=data.copy()
        cols = data.columns()
    
    # number of events for reference later on
    eventno = 0  
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

        # assign x, y and z variables for histogram
        p1data = pd.concat([p1data,phase1])
        p2data = pd.concat([p2data,phase2])


    # calculate the normalized superposed epoch analysis
    # statistics
    # create bins and edges for in normalized 
    # time for binning the data in both phases
    x1bins = np.arange(0, 1, x1_spacing)
    x1_edges = np.arange(0, 1 + x1_spacing, x1_spacing)
    x2bins = np.arange(0, 1, x2_spacing)
    x2_edges = np.arange(0, 1 + x2_spacing, x2_spacing)
    
    # create normalized time axis
    t_norm = (x1bins-x1bins.max()-x1_spacing)/x1_spacing
    t_norm = np.concatenate([t_norm,x2bins/x2_spacing])
    
    # create return DataFrame
    SEAdat = pd.DataFrame()
    SEAdat['t_norm'] = t_norm
    
    
    
    # loop through columns and calculate
    # mean, median, upper and lower quartile
    
    p1mean, _, _ = stats.binned_statistic(p1data['t_norm'], values=p1data['data'], bins=x1_edges, statistic=lambda stat: np.nanmean(stat))
    p1median, _, _ = stats.binned_statistic(p1data['t_norm'], values=p1data['data'], bins=x1_edges, statistic=lambda stat: np.nanpercentile(stat, 50))
    p1lq, _, _ = stats.binned_statistic(p1data['t_norm'], values=p1data['data'], bins=x1_edges, statistic=lambda stat: np.nanpercentile(stat, 25))
    p1uq, _, _ = stats.binned_statistic(p1data['t_norm'], values=p1data['data'], bins=x1_edges, statistic=lambda stat: np.nanpercentile(stat, 75))
    p1c, _, _ = stats.binned_statistic(p1data['t_norm'], values=p1data['data'], bins=x1_edges, statistic='count')
    
    p2mean, _, _ = stats.binned_statistic(p2data['t_norm'], values=p2data['data'], bins=x2_edges, statistic=lambda stat: np.nanmean(stat))
    p2median, _, _ = stats.binned_statistic(p2data['t_norm'], values=p2data['data'], bins=x2_edges, statistic=lambda stat: np.nanpercentile(stat, 50))
    p2lq, _, _ = stats.binned_statistic(p2data['t_norm'], values=p2data['data'], bins=x2_edges, statistic=lambda stat: np.nanpercentile(stat, 25))
    p2uq, _, _ = stats.binned_statistic(p2data['t_norm'], values=p2data['data'], bins=x2_edges, statistic=lambda stat: np.nanpercentile(stat, 75))
    p2c, _, _ = stats.binned_statistic(p2data['t_norm'], values=p2data['data'], bins=x2_edges, statistic='count')
    
    SEAdat['mean'] = np.concatenate([p1mean,p2mean], axis=0)
    SEAdat['median'] = np.concatenate([p1median,p2median], axis=0)
    SEAdat['low_q'] = np.concatenate([p1lq,p2lq], axis=0)
    SEAdat['up_q'] = np.concatenate([p1uq,p2uq],axis=0)
    #SEAdat['cnt'] = np.concatenate([p1c,p2c],axis=0)

    SEAdat = SEAdat.set_index('t_norm')

    return SEAdat


# load the data from local file
omnidata = pd.read_pickle('D:/data/SEAnorm/omnidata')

# select a single parameter for the SEA
data = omnidata['SymH']  # this MUST be a pandas Series for the 1D SEA to work

# load the event list
stormlist = pd.read_csv('D:/data/SEAnorm/StormList_short.txt', index_col=0, parse_dates=[1, 2, 3, 4])
stormlist = stormlist.reset_index(drop=True)

# place into correct format to be used in SEAnorm
starts = stormlist.IStart
epochs = stormlist.RStart
ends = stormlist.REnd
events=[starts, epochs, ends]

# choose the statistic to be used. (e.g. np.mean, np.sum, np.median,... etc.)
# For percentiles, just use an integer on it's own.
statistic=np.nanmean

# specify the number of bins in phase 1 and phase 2 as [nbins1, nbins2]
bins=[25, 100]

# call the function
SEAarray = SEAnorm1D(data, events, statistic, bins)

# plot the result
plt.plot(SEAarray['mean'])
plt.xlabel('Normalised Time Units')
plt.ylabel('Sym-H')
plt.title('New')
plt.show()

SEAarray.plot()





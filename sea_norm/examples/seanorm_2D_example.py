# -*- coding: utf-8 -*-
"""

Example usage of SEAnorm to perfrom 2D time normalized superposed epoch 
analysis on SAMPEX low (1.5-6.0 MeV) and high (2.5-14 MeV) electrons observed
by the PET instrument. The second binning dimension is L-shell from the 'L'
columnd of the loaded data frame.

Uses the WGStormList.txt to set epochs and define the
two phases.

Plots the median SEA for the low and high energy electron channels from the
loaded data frame.

The 2D analysis can be time intensive due to the extra binning. If you do not
need all the default stats (mean, median, upper and lower quartile) it is 
recommended that you define the `seastats` parameter so that only the required
statistics are calculated. 

This exmaple will only return the median. 

WGStormList.txt storm list is generated using the algorithm described in 
Walach and Grocott (2019):
    Walach, M.-T., & Grocott, A. (2019). SuperDARN observations during 
    geomagnetic storms,geomagnetically active times, and enhanced solar wind 
    driving. Journal of Geophysical Research: Space Physics, 124, 5828– 5847.
    https://doi.org/10.1029/2019JA026816

"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from sea_norm import sean


# file location for sampex data
s_dat = 'https://zenodo.org/record/6835641/files/sampexflux.csv.bz2'

# set the columns to run the analysis on
sea_cols = ['ELO','EHI']

# specify the number of bins in phase 1 and phase 2 as [nbins1, nbins2]
bins=[20, 120]

# specify the y parameters for the 2D SEA
y_col = 'L'
ymin = 2.5
ymax = 5.5
y_spacing = 0.2
y_dim = [ymin, ymax, y_spacing]

# define the statistics to use
seastats = {'median':np.nanmedian}

# load the sampex data to analyzed into a DataFrame
sampexdata = pd.read_csv(s_dat,parse_dates=True, 
                        infer_datetime_format=True, header=0, 
                        names=['t','ELO','EHI','L'],
                        index_col=0)

# log the sampex data before performing SEA
# replace infinity values with nan to properly 
# calculate statistics
logdata=sampexdata.copy()
logdata.iloc[:, 0:2]=np.log10(sampexdata.iloc[:, 0:2])
logdata.replace([np.inf, -np.inf], np.nan, inplace=True)

# load the event list and place the
# epoch times into the appropriate format
stormlist = pd.read_csv('D:/data/SEAnorm/StormList_short.txt', index_col=0, 
                        parse_dates=[1, 2, 3, 4])
stormlist = stormlist.reset_index(drop=True)

starts = stormlist.IStart
epochs = stormlist.RStart
ends = stormlist.REnd
events=[starts, epochs, ends]

# perform the 2D SEA analysis
sea2d, meta =  sean(logdata, events, bins, cols=sea_cols, 
                         seastats=seastats, 
                         y_col=y_col,y_dimensions=y_dim)

# get the columsn that the SEA was performed
# on from the returned metadata
cols = meta['sea_cols']  

# grab the y metadata for plotting
ymeta = meta['y_meta']

fig, axes = plt.subplots(nrows=len(cols), sharex=True, 
                         squeeze=True,figsize=(5,3.5))

# loop over columns that were analyzed
for c, ax in zip(cols, axes):
    # for each column identify the column titles which
    # have 'c' in the title
    # a more complex mask would need to be used if multiple
    # statistics where returned    
    mask = sea2d.columns.str.startswith(c) 
    
    # plot the data from the mask
    # transform to a 2D numpy array and transpopse for plotting
    hb = ax.imshow(sea2d.loc[:,mask].to_numpy().transpose(), cmap='inferno', 
              origin='lower', aspect='auto', 
              extent =[sea2d.index.min(),sea2d.index.max(),
                       min(ymeta['edges']),max(ymeta['edges'])])
    ax.set_ylabel(c)
    cb = fig.colorbar(hb, ax=ax, label='log(flux)')

axes[len(cols)-1].set_xlabel('Normalized Time')    
plt.tight_layout()
plt.show()

# -*- coding: utf-8 -*-
"""

Example usage of SEAnorm to perfrom 1D normalized superposed epoch analysis on 
solar wind omni data using the WGStormList.txt


WGStormList.txt storm list is generated using the algorithm described in 
Walach and Grocott (2019):
    Walach, M.-T., & Grocott, A. (2019). SuperDARN observations during 
    geomagnetic storms,geomagnetically active times, and enhanced solar wind 
    driving. Journal of Geophysical Research: Space Physics, 124, 5828– 5847.
    https://doi.org/10.1029/2019JA026816

"""


import pandas as pd
import matplotlib.pylab as plt

from SEAnorm import SEAnorm

# set the columns to run the analysis on
sea_cols = ['V','P','B_Z_GSE','SymH','AE']

# specify the number of bins in phase 1 and phase 2 as [nbins1, nbins2]
bins=[20, 120]

# load omni data and select 
# columns to run analysis on
omnidata = pd.read_hdf('D:/data/SEAnorm/omnidata.hdf')
 

# load the event list and place the
# epoch times into the appropriate format
stormlist = pd.read_csv('D:/data/SEAnorm/StormList_short.txt', index_col=0, 
                        parse_dates=[1, 2, 3, 4])
stormlist = stormlist.reset_index(drop=True)

starts = stormlist.IStart
epochs = stormlist.RStart
ends = stormlist.REnd
events=[starts, epochs, ends]

# perform the noramlized superposed epoch analysis
SEAarray, meta = SEAnorm(omnidata, events, bins, cols=sea_cols)

# get the columsn that the SEA was performed
# on from the returned metadata
cols = meta['sea_cols']                           

# plot the superposed epoch analysis for each variable
# plot the mean, median, upper and lower quartiles
# ignore the cnts column

fig, axes = plt.subplots(nrows=len(cols), sharex=True, 
                         squeeze=True,figsize=(5,8))

for c, ax in zip(cols, axes):
    print(c)
    mask = SEAarray.columns.str.startswith(c) & \
        ~SEAarray.columns.str.endswith('cnt')
    SEAarray.loc[:,mask].plot(ax=ax, style=['r-','b-','b--','b--'], 
                              xlabel='Normalized Time',
                              ylabel=c.replace('_',' '), 
                              legend=False, fontsize=8)






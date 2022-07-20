# -*- coding: utf-8 -*-
"""

Example usage of SEAnorm to perfrom 1D normalized superposed epoch analysis on 
solar wind velocity returning percentiles as defined by a user set of 
statistics vai the seastats parameter. In this example we return 10 equally
spaced percentiles. e.g., the deciles.


WGStormList.txt storm list is generated using the algorithm described in 
Walach and Grocott (2019):
    Walach, M.-T., & Grocott, A. (2019). SuperDARN observations during 
    geomagnetic storms,geomagnetically active times, and enhanced solar wind 
    driving. Journal of Geophysical Research: Space Physics, 124, 5828– 5847.
    https://doi.org/10.1029/2019JA026816
    
StormList_short.txt is the same stormlist but truncated to the time range
where there is data.

"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from sea_norm import sean

# define a function that will return a lambda
# percentile function
def makepercentile(x):
    """
    Parameters
    ----------
    x : int
        Percentile to calculate

    Returns
    -------
    TYPE
        Lambda function to calculate x'th percentile.

    """
    return lambda stat: np.nanpercentile(stat, x)

# get the percentiles 
# use the makepercentile() function
# to return a lambda function for
# different percentiles, required for
# how the lambdas are bound 
seastats = {}
for x in np.arange(10,100,10):
    seastats[f'{x}th_%tile'] = makepercentile(x)

# file location for omnidata
o_dat='https://zenodo.org/record/6835641/files/omnidata.csv.bz2'

# set the columns to run the analysis on
sea_cols = ['V']

# specify the number of bins in phase 1 and phase 2 as [nbins1, nbins2]
bins=[20, 120]

# define a set of lambda functions that will be used

# load omni data and select 
# columns to run analysis on
omnidata = pd.read_csv(o_dat,parse_dates=True, 
                        infer_datetime_format=True, header=0, 
                        names=['t','B_Z_GSE','V','P','AE','SymH'],
                        index_col=0)


 
# load the event list and place the
# epoch times into the appropriate format
stormlist = pd.read_csv('StormList_short.txt', index_col=0, 
                        parse_dates=[1, 2, 3, 4])
stormlist = stormlist.reset_index(drop=True)

starts = stormlist.IStart
epochs = stormlist.RStart
ends = stormlist.REnd
events=[starts, epochs, ends]

# perform the noramlized superposed epoch analysis
SEAarray, meta = sean(omnidata, events, bins, cols=sea_cols, 
                         seastats=seastats)

# set up plotting
fig = plt.figure()
ax = plt.subplot(111)

# get column names for labesl and for setting colors
cols = SEAarray.columns.values.tolist()

# setup some colors
colors = plt.cm.plasma(np.linspace(0,1,len(cols)))

# make better legend labels from the column names
lab_col = []
for cc in cols:
    lab_col.append(cc[2:].replace('_',' '))

# plot the DataFrame
SEAarray.plot(ax=ax, ylabel=sea_cols[0], xlabel='Normalized Time', 
              title='Storm-time Solar Wind Velocity Percentiles', 
              color=colors)                        

# shrink current axis by 20% for the new axis
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# place the legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
          labels=lab_col, facecolor='silver')

# change background color so it's easier to see
ax.set_facecolor('silver')

plt.show()


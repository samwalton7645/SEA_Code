# example use of SEAnorm.py to produce a 1D superposed epoch analysis using OMNIWeb data.

# WGStormList.txt storm list is generated using the algorithm described in Walach and Grocott (2019):
#    Walach, M.-T., & Grocott, A. (2019). SuperDARN observations during geomagnetic storms,
#    geomagnetically active times, and enhanced solar wind driving.
#    Journal of Geophysical Research: Space Physics, 124, 5828– 5847.
#    https://doi.org/10.1029/2019JA026816

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from SEAnorm import SEAnorm


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
bins=[5, 50]

# call the function
SEAarray = SEAnorm(data, events, statistic, bins)

# plot the result
plt.plot(SEAarray)
plt.xlabel('Normalised Time Units')
plt.ylabel('Sym-H')
plt.show()
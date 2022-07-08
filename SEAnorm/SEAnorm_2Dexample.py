# example use of SEAnorm.py to produce a 2D superposed epoch analysis using the SAMPEX mission flux data.

# WGStormList.txt storm list is generated using the algorithm described in Walach and Grocott (2019):
#    Walach, M.-T., & Grocott, A. (2019). SuperDARN observations during geomagnetic storms,
#    geomagnetically active times, and enhanced solar wind driving.
#    Journal of Geophysical Research: Space Physics, 124, 5828– 5847.
#    https://doi.org/10.1029/2019JA026816

from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pylab as plt
import gc
from SEAnorm import SEAnorm


# load the data from local file
data = pd.read_pickle('sampexflux')  # this MUST be a pandas DataFrame for the 2D SEA to work
                                     # first column must contain the data being analysed
                                     # second column must contain the y-axis data

# if the desired analysis is of logged data, create the logged data before calling the function
logdata=data.copy()
logdata.iloc[:, 0]=np.log10(data.iloc[:, 0])

# load the event list
stormlist = pd.read_csv('WGStormList.txt', index_col=0, parse_dates=[1, 2, 3, 4])
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

# specify the y-dimensions of the SEA
ymin = 2.5
ymax = 5.5
y_spacing = 0.2

y_dimensions = [ymin, ymax, y_spacing]

# call the function
SEAarray = SEAnorm(logdata, events, statistic, bins, y_dimensions)

# plot the result
im=plt.imshow(SEAarray, cmap='inferno', origin='lower', aspect='auto')
plt.xlabel('Normalised Time Units')
plt.ylabel('L-Shell')

# add a colour bar
cb=plt.colorbar(im)
cb.ax.set_ylabel('log(flux)')

plt.show()
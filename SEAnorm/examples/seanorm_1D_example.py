# -*- coding: utf-8 -*-
"""
Example usage of SEAnorm


"""

# load the data from local file
#omnidata = pd.read_pickle('D:/data/SEAnorm/omnidata')

# select a single parameter for the SEA
#data = omnidata[['V','P','B_Z_GSE','SymH','AE']]  

# load the event list
stormlist = pd.read_csv('D:/data/SEAnorm/StormList_short.txt', index_col=0, 
                        parse_dates=[1, 2, 3, 4])
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
bins=[20, 120]

# call the function
p90 = lambda stat: np.nanpercentile(stat, 90)
#SEAarray, cols = SEAnorm1D(data, events, bins,
#                           cols=['SymH','AE'],
#                           seastats={'test':np.nanmin, 'p90':p90})


# SEAarray, cols = SEAnorm1D(data, events, bins)
# fig, axes = plt.subplots(nrows=len(cols), sharex=True, 
#                          squeeze=True,figsize=(5,8))

# for c, ax in zip(cols, axes):
#     print(c)
#     mask = SEAarray.columns.str.startswith(c) & \
#         ~SEAarray.columns.str.endswith('cnt')
#     SEAarray.loc[:,mask].plot(ax=ax, style=['r-','b-','b--','b--'], 
#                               xlabel='Normalized Time',
#                               ylabel=c.replace('_',' '), 
#                               legend=False, fontsize=8)


#2D testing

data = pd.read_pickle('D:/data/SEAnorm/sampexflux')  # this MUST be a pandas DataFrame for the 2D SEA to work
                                     # first column must contain the data being analysed
                                     # second column must contain the y-axis data

# if the desired analysis is of logged data, create the logged data before calling the function
logdata=data.copy()
logdata.iloc[:, 0]=np.log10(data.iloc[:, 0])

# specify the number of bins in phase 1 and phase 2 as [nbins1, nbins2]
bins=[5, 50]

# specify the y-dimensions of the SEA
ymin = 2.5
ymax = 5.5
y_spacing = 0.2
y_dim = [ymin, ymax, y_spacing]

sea2d, meta =  SEAnorm1D(logdata, events, bins, cols=['ELO'], 
                         seastats={'mean':np.nanmean}, 
                         y_col='L',y_dimensions=y_dim)
    
yy = meta['y_meta']

# plot the result
im=plt.imshow(sea2d.to_numpy().transpose(), cmap='inferno', 
              origin='lower', aspect='auto', 
              extent =[sea2d.index.min(),sea2d.index.max(),min(yy['edges']),max(yy['edges'])] )
plt.xlabel('Normalised Time Units')
plt.ylabel('L-Shell')

# add a colour bar
cb=plt.colorbar(im)
cb.ax.set_ylabel('log(flux)')
plt.show()
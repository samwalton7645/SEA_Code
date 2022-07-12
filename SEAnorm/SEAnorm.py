from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy import stats
import gc

def SEAnorm(data, events, statistic, x_dimensions, y_dimensions='none'):

    """
    Arguments:
    data - Pandas time series containing time series to be used.
         - Must have a Pandas datetime index
         - For 2D analysis, a DataFrame column must be used with values in column 0 and y-data in column 1.
    events - list of three arrays [x, y, z] containing the start, epoch and end times of each event
           - times must either be timestamps or strings in the format: 'YYYY-MM-DD HH:MM:SS'
    statistic - the statistical function to be used in the SEA (e.g. np.mean, np.median, np.sum, etc.)
              - for percentiles, input an integer only
    x_dimensions - list [x, y] containing two elements: the desired number of normalised time bins in [phase 1, phase 2]
    y_dimensions - required only for 2D histograms
                 - list [x, y, z] containing the min, max and desired bin spacing in the y-dimension

    Returns:
    1D or 2D numpy array containing the final time-normalised superposed epoch analysis.
    """

    if isinstance(data, pd.core.frame.DataFrame)==True: # 2D SEA
        starts, epochs, ends = events
        x1_spacing, x2_spacing = 1/x_dimensions[0], 1/x_dimensions[1]
        ymin, ymax, y_spacing = y_dimensions

        eventno = 0  # for reference later on
        gc.collect()

        for event in tqdm(range(len(starts))):
            start = str(starts.iloc[event])
            epoch = str(epochs.iloc[event])
            end = str(ends.iloc[event])

            phase1 = data[start:epoch].copy()
            phase2 = data[epoch:end].copy()

            # normalise time axis of phase 1 for each phase from 0 to 1.
            try:
                phase1['t_norm'] = phase1.index - phase1.index[0]   # reset time for this event to 0
            except IndexError:
                print('There is no data for event '+str(event))   # in case there is no data during a given event
                continue
            phase1['t_norm'] = phase1['t_norm'].dt.total_seconds()  # get time in seconds only, ready to normalise
            p1min = phase1['t_norm'][0]                            # find smallest and largest values (to become 0 and 1)
            p1max = phase1['t_norm'][-1]
            phase1['t_norm'] = (phase1['t_norm'] - p1min) / (p1max - p1min)  # normalise the time values from 0 to 1

            # normalise the time axis for phase 2
            try:
                phase2['t_norm'] = phase2.index - phase2.index[0]
            except IndexError:
                continue
            phase2['t_norm'] = phase2['t_norm'].dt.total_seconds()
            p2min = phase2['t_norm'][0]
            p2max = phase2['t_norm'][-1]
            phase2['t_norm'] = ((phase2['t_norm'] - p2min) / (p2max - p2min))

            # assign x, y and z variables for histogram
            p1xdata = phase1.iloc[:, 2]
            p1ydata = phase1.iloc[:, 1]
            p1zdata = phase1.iloc[:, 0]

            p2xdata = phase2.iloc[:, 2]
            p2ydata = phase2.iloc[:, 1]
            p2zdata = phase2.iloc[:, 0]

            # create bins and edges for histogram
            x1bins = np.arange(0, 1, x1_spacing)
            x2bins = np.arange(0, 1, x2_spacing)
            ybins = np.arange(ymin, ymax, y_spacing)
            x1_edges = np.arange(0, 1 + x1_spacing, x1_spacing)
            x2_edges = np.arange(0, 1 + x2_spacing, x2_spacing)
            y_edges = np.arange(ymin, ymax + y_spacing, y_spacing)

            try:
                p1array, _, _, _ = stats.binned_statistic_2d(p1xdata, p1ydata, values=p1zdata, bins=[x1_edges, y_edges], statistic=lambda stat: np.nanpercentile(stat, 50))
                p2array, _, _, _ = stats.binned_statistic_2d(p2xdata, p2ydata, values=p2zdata, bins=[x2_edges, y_edges], statistic=lambda stat: np.nanpercentile(stat, 50))
            except ValueError:
                p1array = np.zeros((x1bins.shape[0], ybins.shape[0]))  # in case there is no data in any of the defined bins
                p1array[:] = np.nan
                p2array = np.zeros((x2bins.shape[0], ybins.shape[0]))
                p2array[:] = np.nan

            if eventno == 0:
                p1all = p1array
                p2all = p2array
                eventno = 'done'
            else:
                p1all = np.dstack((p1all, p1array))
                p2all = np.dstack((p2all, p2array))

        # calculate statistic for phase 1 and 2 histograms
        if isinstance(statistic, int) == True:
            p1 = np.nanpercentile(p1all, statistic, axis=2)
            p2 = np.nanpercentile(p2all, statistic, axis=2)
        else:
            p1 = statistic(p1all, axis=2)
            p2 = statistic(p2all, axis=2)

        # concatenate phases to make final superposed epoch analysis
        SEAarray = np.concatenate((p1, p2), axis=0)
        SEAarray = np.swapaxes(SEAarray, 0, 1)
    elif isinstance(data, pd.core.series.Series)==True:  # 1D SEA
        starts, epochs, ends = events
        x1_spacing, x2_spacing = 1/x_dimensions[0], 1/x_dimensions[1]

        data=data.to_frame('data')  # changes time series into a data frame

        eventno = 0  # for reference later on
        gc.collect()

        for event in tqdm(range(len(starts))):
            start = str(starts.iloc[event])
            epoch = str(epochs.iloc[event])
            end = str(ends.iloc[event])

            phase1 = data[start:epoch].copy()
            phase2 = data[epoch:end].copy()

            # normalise time axis of phase 1 for each phase from 0 to 1.
            try:
                phase1['t_norm'] = phase1.index - phase1.index[0]   # reset time for this event to 0
            except IndexError:
                print('There is no data for event '+str(event))   # in case there is no data during a given event
                continue
            phase1['t_norm'] = phase1['t_norm'].dt.total_seconds()  # get time in seconds only, ready to normalise
            p1min = phase1['t_norm'][0]                            # find smallest and largest values (to become 0 and 1)
            p1max = phase1['t_norm'][-1]
            phase1['t_norm'] = (phase1['t_norm'] - p1min) / (p1max - p1min)  # normalise the time values from 0 to 1

            # normalise the time axis for phase 2
            try:
                phase2['t_norm'] = phase2.index - phase2.index[0]
            except IndexError:
                continue
            phase2['t_norm'] = phase2['t_norm'].dt.total_seconds()
            p2min = phase2['t_norm'][0]
            p2max = phase2['t_norm'][-1]
            phase2['t_norm'] = ((phase2['t_norm'] - p2min) / (p2max - p2min))

            # assign x, y and z variables for histogram
            p1xdata = phase1['t_norm']
            p1ydata = phase1['data']

            p2xdata = phase2['t_norm']
            p2ydata = phase2['data']

            # create bins and edges for histogram
            x1bins = np.arange(0, 1, x1_spacing)
            x2bins = np.arange(0, 1, x2_spacing)
            x1_edges = np.arange(0, 1 + x1_spacing, x1_spacing)
            x2_edges = np.arange(0, 1 + x2_spacing, x2_spacing)

            try:
                p1array, _, _ = stats.binned_statistic(p1xdata, values=p1ydata, bins=x1_edges, statistic=lambda stat: np.nanpercentile(stat, 50))
                p2array, _, _ = stats.binned_statistic(p2xdata, values=p2ydata, bins=x2_edges, statistic=lambda stat: np.nanpercentile(stat, 50))
            except ValueError:
                p1array = np.zeros(len(x1bins))
                p1array[:] = np.nan
                p2array = np.zeros(len(x2bins))
                p2array[:] = np.nan

            if eventno == 0:
                p1all = p1array
                p2all = p2array
                eventno = 'done'
            else:
                p1all = np.vstack((p1all, p1array))
                p2all = np.vstack((p2all, p2array))

        # calculate statistic for phase 1 and 2 histograms
        if isinstance(statistic, int) == True:
            p1 = np.nanpercentile(p1all, statistic, axis=0)
            p2 = np.nanpercentile(p2all, statistic, axis=0)
        else:
            p1 = statistic(p1all, axis=0)
            p2 = statistic(p2all, axis=0)

        # concatenate phases to make final superposed epoch analysis
        SEAarray = np.concatenate((p1, p2), axis=0)
    else:
        print('Incorrect data input.\nFor 1D SEA, input pandas Series.\nFor 2D SEA, input pandas DataFrame with one data column and one y-data column.\nAny input must have a pandas datetime index.')
        return
    return SEAarray


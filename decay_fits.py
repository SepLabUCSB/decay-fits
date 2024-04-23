import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import optimize, signal
import os
import warnings
plt.ion()


data_folder = ''
plt.style.use('scientific.mplstyle')



# FUNC = 'linear'
# FUNC = 'monoexponential'
# FUNC = 'monoexp-linear'
# FUNC = 'monoexponential-inflection'
# FUNC = 'biexponential'
FUNC = 'biexponential-inflection'


BASELINE_CORRECT = False
I_SCALE = 1e-3        # Conversion to amps. i.e. data in mA, I_SCALE = 1e-3
START_AFTER = 10      # cut off first (n) seconds
min_s_to_fit = 5      # Requires n seconds of data to accept the fit
DELAY = 0             # Points after "fast" spike to skip fitting on
thresh = 0.5          # Used to determine acceptable baseline "flatness"
                      # Smaller = more picky, need flatter baseline to accept spike
apply_filter     = False
filter_freq      = 25



def load_indices_from_file(data_file):
    
    def get_vals(path):
        df = pd.read_csv(path)
        return df['Index'].to_list()
    
    # Automatically look for previously-generated file
    path = None
    old_file = data_file.replace('.txt', '_indices.csv')
    if os.path.exists(old_file):
        check = input(f'Found index file: {old_file}. Load? (y/n) >>')
        if check == 'y':
            return get_vals(old_file)
            
    # Otherwise prompt user for file        
    if not path:
        indice_file = input('File >>')
        
        # Look for file with exact input path
        if os.path.exists(indice_file):
            return get_vals(indice_file)
        
        # Try to build path from user input and known data file
        path = data_folder + '/' + indice_file + '.csv'
        while not os.path.exists(path):
            print("Invalid input. Please enter indice file name")
            indice_file = input('>>')
            path = data_folder + '/' + indice_file + '.csv'
    
    return get_vals(path)


def thresholding_algo(y, lag, threshold, influence):
    '''
    Taken from 
    https://stackoverflow.com/questions/22583391/peak-signal-detection-in-
    realtime-timeseries-data/43512887#43512887
    
    
    Returns:
        signals: array of [-1, 0, 1]. 0 = no spike at this point,
                 +- 1 = positive/ negative spike at this point
        avgFilter: array, filtered data
        stdFilter: array, standard devation of filtered data
    '''
    
    signals = np.zeros(len(y))
    filteredY = np.array(y)
    avgFilter = [0]*len(y)
    stdFilter = [0]*len(y)
    avgFilter[lag - 1] = np.mean(y[0:lag])
    stdFilter[lag - 1] = np.std(y[0:lag])
    for i in range(lag, len(y)):
        if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter [i-1]:
            if y[i] > avgFilter[i-1]:
                signals[i] = 1
            else:
                signals[i] = -1

            filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
            stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])
        else:
            signals[i] = 0
            filteredY[i] = y[i]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
            stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])
    
    return signals, np.array(avgFilter), np.array(stdFilter)


def find_inflection(ts, data, len_max_time):
    '''
    Calculate the slope for the first 'n' points, where n = 50.

    Parameters
    ----------
    slopes : list
        List of slopes after each spike.
    inflection_pt : value
        Index where slope is 10% of the first slope in slopes.

    Returns
    -------
    inflection_pt : index value
        Index after spike where the slope changes by 5%.
    '''

    count = 0
    slopes = []
    # Calculate the slope for the first 30 points
    while count < min(30, len(data)-1):
        m = (data[count+1]-data[count])/(ts[count+1]-ts[count])
        slopes.append(m)
        count += 1
    
    # Calculate IP by comparing slope to first until < 10%
    inflection_pt = 0
    for slope in slopes:
          if slope < 0.05*slopes[0]:
              #print(slopes.index(slope))
              inflection_pt = slopes.index(slope)
              # print(inflection_pt)
              break
    # print(inflection_pt)
    return inflection_pt


def shift_axes(ax, direction):
    # Pan axes if arrow keys are pressed
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xdelta = 0.2*(xlim[1] - xlim[0])
    ydelta = 0.2*(ylim[1] - ylim[0])
    if direction == 'right':
        xlim = [
            xlim[0] + xdelta,
            xlim[1] + xdelta,
            ]
    if direction == 'left':
        xlim = [
            xlim[0] - xdelta,
            xlim[1] - xdelta,
            ]
    if direction == 'up':
        ylim = [
            ylim[0] + ydelta,
            ylim[1] + ydelta,
            ]
    if direction == 'down':
        ylim = [
            ylim[0] - ydelta,
            ylim[1] - ydelta,
            ]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return
    
    
def zoom_axes(ax, direction, center):
    # Zoom axes by mouse scroll wheel
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Shift (almost) to new center
    x, y = center
    shift_x = x - (xlim[0] + xlim[1])/2 
    shift_y = y - (ylim[0] + ylim[1])/2 
    
    if direction == 'down':
        shift_x = -shift_x
        shift_y = -shift_y
    
    xlim = [
        xlim[0] + 0.7*shift_x,
        xlim[1] + 0.7*shift_x,
        ]  
    ylim = [
        ylim[0] + 0.7*shift_y,
        ylim[1] + 0.7*shift_y,
        ]
        
    # Zoom in or out
    xdelta = 0.2*(xlim[1] - xlim[0])
    ydelta = 0.2*(ylim[1] - ylim[0])
    
    if direction == 'down':
        xdelta = -xdelta
        ydelta = -ydelta
    
    xlim = [
        xlim[0] + xdelta,
        xlim[1] - xdelta,
        ]
    ylim = [
        ylim[0] + ydelta,
        ylim[1] - ydelta,
        ]
        
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return    

class ExpFunc():
    def _linear(x, a, b):
        return a*x + b

    def _monoexponential(x, a,b,c):
        return a * np.exp(-b * x) + c

    def _monoexplinear(x, a, b, m, c):
        return a*np.exp(-b * x) + m*x + c
  
    def _monoexpinflection(x, a,b,c):
        return a * np.exp(-b * x) + c

    def _biexponential(x, a,b,c,d,e):
        return a * np.exp(-b * x) + c * np.exp(-d * x) + e
 
    def _biexpinflection(x, a,b,c,d,e):
        return a * np.exp(-b * x) + c * np.exp(-d * x) + e
    
    
    _func_mapping = {'linear': _linear,
                     'monoexponential': _monoexponential,
                     'monoexp-linear': _monoexplinear,
                     'monoexponential-inflection': _monoexpinflection,
                     'biexponential': _biexponential,
                     'biexponential-inflection': _biexpinflection,}
    
    _bound_mapping = {'linear': (-np.inf, [1., np.inf]),
                      'monoexponential': (-np.inf, [1., np.inf, np.inf]),
                      'monoexp-linear': (-np.inf, [1., np.inf, np.inf]),
                      'monoexponential-inflection': (-np.inf, [1., np.inf, np.inf]),
                      'biexponential': (-np.inf, [1., np.inf, np.inf, np.inf, np.inf]),
                      'biexponential-inflection': (-np.inf, [1., np.inf, np.inf, np.inf, np.inf]),}
    
    _param_mapping = {'linear': ['a/ A s-1', 'b/ A'],
                      'monoexponential': ['a/ A', 'b/ s-1', 'c/ A'],
                      'monoexp-linear': ['a/ A', 'b/ s-1', 'm/ A s-1', 'c/ A'],
                      'monoexponential-inflection': ['a/ A', 'b/ s-1', 'c/ A'],
                      'biexponential': ['a/ A', 'b/ s-1', 'c/ A', 'd/ s-1', 'e/ A'],
                      'biexponential-inflection': ['a/ A', 'b/ s-1', 'c/ A', 'd/ s-1', 'e/ A'],}
    
    
    
    def func(self):
        return self._func_mapping[FUNC]
    
    def bounds(self):
        return self._bound_mapping[FUNC]
    
    def param_names(self):
        return self._param_mapping[FUNC]
    
    
    
        
   

class Spike:
    
    def __init__(self, DataFile, idx):
        self.DataFile = DataFile # Local ref to DataFile which contains this Point
        self.REMOVE = False
        
        
        self.idx         = idx  # int, index corresponding to peak current
        self.left_bound  = None # int, set by self.integrate_Hads(). For baseline
        self.right_bound = None # int, set by self.get_right_bound(). For integration/ fitting
        
        self.H_integral     = None    # float
        self.cat_integral   = None    # float
        self.fit_params     = None    # array (popt output by curve_fit)
        self.chi_sq         = None    # float
        
        self.artists = []   # List of matplotlib artists to draw
        
    
    
    def get_right_bound(self):
        # Examine previous and next Point
        all_idxs = self.DataFile.spike_idxs()
        
        this_point_idx = [i for i, idx in enumerate(all_idxs) if idx == self.idx][0]
        
        # This point is the last one in the file
        if self.idx == all_idxs[-1]:
            self.right_bound = len(self.DataFile.i) - 1 
            return
        
        # Otherwise, use start of next spike to end this point's bounds
        next_spike = self.DataFile.spikes[this_point_idx + 1]
        self.right_bound = next_spike.left_bound
        return
        
    
    
    def integrate_Hads(self):
        if self.REMOVE:
            # Already flagged this point for removal. Skip this analysis.
            return
                
        t = self.DataFile.t
        y = self.DataFile.i
        avg = self.DataFile.avg
        
        count = 0
        slopes = []
        # Calculate the slope for the first 20 points backwards
        while count < 30:
            m = (avg[self.idx-count-1]-avg[self.idx-count])/(t[self.idx-count-1]-t[self.idx-count])
            slopes.append(m)
            count += 1
        # print(slopes)
        
        pos_slopes = []
        # Calculate IP by comparing slope to first until < 10%
        for slope in slopes:
              if slope > 0:
                  cliff_pt = slopes.index(slope)
                  # print(idx, cliff_pt, idx-cliff_pt)
                  pos_slopes.append(cliff_pt)
        
        left_bound = self.idx - pos_slopes[0]
        right_bound = self.idx + pos_slopes[0]
        
        xs = t[left_bound:self.idx]
        ys = y[left_bound:self.idx] - y[left_bound]
        integral = np.trapz(ys, xs)
        # print(t[idx])
        
        if integral > 0:
            self.REMOVE = True
            print(f'Positive integral: {self.idx}')
        
        self.H_integral = integral
        self.left_bound = left_bound
        return
    
    
    
    def _baseline(self, t, y):
        bas_idx = max(0, self.left_bound-10)
        
        m, b = np.polyfit(t[bas_idx:self.left_bound],
                          y[bas_idx:self.left_bound], deg=1)
        
        baseline = 0*t
        if BASELINE_CORRECT:
            baseline = m*t + b
        return baseline
        
    
    
    def integrate_cat(self):
        if self.REMOVE:
            # Already flagged this point for removal. Skip this analysis.
            return
        
        t = self.DataFile.t
        y = self.DataFile.i
        
        baseline = self._baseline(t, y)
        ts = t[self.idx:self.right_bound] - t[self.idx]
        data = y[self.idx:self.right_bound] - baseline[self.idx:self.right_bound]
        inflection_pt = find_inflection(ts, data, None)
            
        
        # Fit the line connecting Left_idx to Next_idx
        (a,b) = np.polyfit(np.array([t[self.left_bound], t[self.right_bound]]),
                            np.array([y[self.left_bound], y[self.right_bound]]), 1)
        x1 = t[self.idx - inflection_pt:self.right_bound]
        y1 = a*x1 + b
        # ax.plot(x1, y1, '-', color = 'r')
        
        ### Calculate integral (excludes initial sharp spike)      
        # area to draw
        verts = [(t[self.idx - inflection_pt], y[self.idx - inflection_pt]),
                 *zip(t[self.idx - inflection_pt:self.right_bound],
                      y[self.idx - inflection_pt:self.right_bound]),
                 (t[self.right_bound], y[self.right_bound])]
        poly = Polygon(verts, color= 'y', alpha = 0.5, ec = 'k')
        
        # actual calculation
        xs = t[self.idx - inflection_pt:self.right_bound]
        ys = y[self.idx - inflection_pt:self.right_bound] - y1
        integ = np.trapz(ys, xs)
        
        # self.artists.append(poly)
        self.cat_integral = integ
        self.artists.append(poly)
        return
    
    
    
    def fit_decay(self):
        if self.REMOVE:
            # Already flagged this point for removal. Skip this analysis.
            return
        
        t = self.DataFile.t
        y = self.DataFile.i
        
        # Subtract out linear baseline
        baseline = self._baseline(t, y)
        ts = t[self.idx:self.right_bound] - t[self.idx]
        data = y[self.idx:self.right_bound] - baseline[self.idx:self.right_bound]
        inflection_pt = find_inflection(ts, data, None)
        
        
        exp_func = ExpFunc().func()
        bounds   = ExpFunc().bounds()
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                popt, pcov = optimize.curve_fit(exp_func, ts, data, 
                                maxfev=100000,
                                #bounds=bounds
                                )
        except: 
            # Failed to fit at this point
            print(f'Chi^2 could not compute (divide by zero) at {t[self.idx]}')
            self.REMOVE = True
            return
                
        fit_y = exp_func(ts, *popt)
        residuals = abs((data - fit_y)/fit_y)
 
        # Update self.artists with fitted curve and marker point
        ln = Line2D(ts+t[self.idx], 
                    fit_y+baseline[self.idx:self.right_bound],
                    marker='o', color='gold', ms=3)
        pt = Line2D([t[self.idx]], [y[self.idx]], marker='o', color='red')
        
        self.chi_sq = np.sum(residuals**2)/len(residuals)
        self.fit_params = popt
        self.artists.extend([ln, pt])
        return
    
    
    
    def get_results(self)->pd.DataFrame:
        '''
        Collect all integration and fitting results and return a DataFrame
        '''
        ## For debugging
        # print('')
        # print(f'Exporting {self.DataFile.t[self.idx]} s. IDX = {self.idx}')
        # print(f'Catalytic integral: {self.cat_integral}')
        # print(f'Hads integral: {self.H_integral}')
        # print(f'Fit parameters: {self.fit_params}')
        # print(f'Fit chi squared: {self.chi_sq}')
        # print('')
        
        d = {
            'Index': [self.idx],
            'Number': [0],
            'Time/s': [self.DataFile.t[self.idx]],
            **{name:[val] for name,val in zip(ExpFunc().param_names(),
                                              self.fit_params)},
            'Catalytic area/ C': [self.cat_integral],
            'Hads integral/ C': [self.H_integral],
            'Reduced Chi^2': [self.chi_sq],
            }
        
        return pd.DataFrame(d)
    








class DataFile():
    def __init__(self, file):
        self.file = file
        self.spikes = []
        
        self.t = None    # Time, array
        self.i = None    # Current, array
        self.sample_rate = None    # float
        self.avg = None    # Filtered current, array
        
        self.analyze_file()
        
    
    def spike_idxs(self):
        return [spike.idx for spike in self.spikes]
    
    
    def get_data(self):
        df = pd.read_csv(self.file, names=('t', 'v', 'i'), skiprows=1, sep='\t')
        df = df[df['t'] > START_AFTER]
        df = df[df['t'] < 60]
        
        t = np.array(df['t'])
        i = np.array(df['i'])*I_SCALE
        
        # Calculate sampling rate
        self.sample_rate = np.average([t[i] - t[i-1] for i in range(1, len(t))])
        self.t = t
        self.i = i
    
    
    def refine_peaks(self):
        '''
        Finds local maxima to refine peak locations    
        '''
                
        signals, self.avg, stdFilter = thresholding_algo(self.i, lag=50, 
                                                          threshold=5, influence=0.6)
        
        
        # ii = input('Import indices? (y/n)>>')
        # if ii == 'y':
        #     idxs = load_indices_from_file(self.file)
        #     if idxs:
        #         return idxs
            
        print('Could not load from file. Finding new indices.')
        
        
        # Determine index of this peak and next peak
        idxs = list(np.where(signals != 0)[0])
        
        # Remove double labelled spikes
        for idx in idxs:
            for i in range(idx-10, idx+10):
                if i in idxs and i != idx:
                    #print(f'Removing {i} because of duplicate')
                    idxs.remove(i)
            
            
        # Refine peak location            
        for idx in idxs[:]:
            if any(abs(self.i[idx-10:idx+10]) > abs(self.i[idx])):
                
                i = np.where(abs(self.i) ==
                              max(abs(self.i[idx-10:idx+10])))[0][0]
                
                #print(f'Moving {idx} to {i}')
                idxs.remove(idx)
                idxs.append(i)
                idxs.sort()
        return idxs
    
    
    
    def find_points(self):
        idxs = self.refine_peaks()
        for idx in idxs:
            self.spikes.append( Spike(DataFile=self, idx=idx) )
    
        
    
    def remove_at_t(self, clickX, xtol):
        'Remove closest point to user click'
        for spike in self.spikes:
            x = self.t[spike.idx]
            if (clickX-xtol < x < clickX+xtol):
                # Remove point
                print(f'Removing closest point at {x}s')
                self.remove_spike(spike)
    
                    
    
    def remove_spike(self, spike):
        if spike.artists:
            for artist in spike.artists:
                try:
                    artist.remove()
                except Exception as e:
                    print(f'Error removing artist {artist}')
        self.spikes.remove(spike)
                
    
    
    def process_points(self):
        # Do H integration for everyone first. Sets left bounds of each 
        # spike for later funcs
        for spike in self.spikes:
            spike.integrate_Hads()
            
        for spike in self.spikes:
            spike.get_right_bound()   # Determine right border to integrate/fit to
            spike.integrate_cat()     # Get area under catalytic curve
            spike.fit_decay()         # Fit decay transient
            
        for spike in self.spikes:
            if spike.REMOVE:
                self.remove_spike(spike)
                
    
    
    def analyze_file(self):
        self.get_data()
        self.find_points()
        self.process_points()
        
        
        
    def get_artists(self):
        l = []
        for spike in self.spikes:
            l.extend(spike.artists)
        return l
        
    
    
    def get_results(self)->pd.DataFrame:
        df = self.spikes[0].get_results()
        file_col = [f'File: {self.file}']
        for spike in self.spikes[1:]:
            df = df.append(spike.get_results())
            file_col.append('')
        file_col[1] = f'Fit: {FUNC}'
        file_col[2] = f'Baseline correct: {BASELINE_CORRECT}'
        file_col[3] = f'Delay: {DELAY} pts'
        
        df['File'] = file_col
        df['Number'] = [i+1 for i in range(len(file_col))]
        return df
    
    
    
    def save_output(self):
        df = self.get_results()
        path = f'{self.file.replace(".txt", "_output.xlsx")}'
        writer = pd.ExcelWriter(path, engine='xlsxwriter')
        df.to_excel(writer, index=False, header=True, startcol=0)
        print(f'Saved results to {path}')
        
    
    
    def save_indices(self):
        ind_df = self.get_results()['Index']
        ind_path = f'{self.file.replace(".txt", "_indices.csv")}'
        ind_df.to_csv(ind_path, index=False)
        print(f'Saved indices to {ind_path}')
            
   

class InteractivePicker:
    
    def __init__(self, file, fig, ax, plot=True):
        self.file = file
        self.fig = fig
        self.ax = ax
        self.plot = plot
        
        print(f'\n===== Loading {self.filename()} =====')
        
        self.DataFile = DataFile(self.file)
        self.draw()
    

    def __call__(self, event):
        if not self.plot:
            return
        # Set xtol based on current axis limits 
        upper = self.ax.get_xlim()[1]
        lower = self.ax.get_xlim()[0]
        diff = upper-lower
        xtol = 0.01*diff
        
        if event.inaxes and self.fig.canvas.manager.toolbar.mode == "":
            # print(xtol)
            clickX = event.xdata
            if (self.ax is None) or (self.ax is event.inaxes):
                # Prioritizing removing marked point
                self.DataFile.remove_at_t(clickX, xtol)
                self.fig.canvas.draw_idle()
                
                
    
    def filename(self):
        return os.path.basename(os.path.normpath(self.file))
                
                
                
    def keypress_event_handler(self, event):
        # Pressing arrow keys pans around graph
        key = event.key
        if key in ['left', 'right', 'up', 'down']:
            shift_axes(self.ax, key)
        self.fig.canvas.draw_idle()
    
    
    
    def scroll_event_handler(self, event):
        # Scrolling mousewheel zooms in/out
        key = event.button
        x, y = event.xdata, event.ydata
        if key in ['down', 'up']:
            zoom_axes(self.ax, key, (x,y))
        self.fig.canvas.draw_idle() 
        
        
        
    def draw(self):
        self.ax.cla()
        self.ax.plot(self.DataFile.t, self.DataFile.i, '.', color='k')
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Current (A)')
        self.ax.set_title(self.filename(), fontsize=7, )
        self.ax.set_facecolor("0.9")
        
        for artist in self.DataFile.get_artists():
            self.ax.add_artist(artist)
            self.ax.draw_artist(artist)
            # artist.set_figure(self.fig)
            # artist.draw()
            
        self.fig.canvas.draw_idle()
            
    
 
class Index():
    def __init__(self, folder, fig, ax):
        self.fig = fig
        self.ax  = ax
        
        self.folder = folder
        self.files = [os.path.join(folder, file)
                      for file in os.listdir(folder)
                      if file.endswith('.txt')]
        
        self.ind = 0
        
        self.Picker  = None
        self.Pickers = []
        self.get_picker()
        
        self.load_cids()
        
        
        
    def load_cids(self):
        if hasattr(self, 'cid'):
            for cid in [self.cid, self.cid2, self.cid3]:
                self.fig.canvas.mpl_disconnect(cid)
        self.cid  = self.fig.canvas.mpl_connect('button_press_event', self.Picker)
        self.cid2 = self.fig.canvas.mpl_connect('key_press_event', self.Picker.keypress_event_handler)
        self.cid3 = self.fig.canvas.mpl_connect('scroll_event', self.Picker.scroll_event_handler)
    
        
    
    def get_picker(self):
        if len(self.Pickers) >= self.ind+1:
            p = self.Pickers[self.ind]
            p.draw() # Redraw this file
            print(f'\n===== Reloaded {p.filename()} =====')
            return p
        p = InteractivePicker(self.files[self.ind], self.fig, self.ax)
        self.Picker = p
        self.Pickers.append(p)
        return p
      


    def _next(self, _):
        self.ind += 1
        if self.ind > len(self.files) - 1:
            self.ind = 0
        self.get_picker()
        self.load_cids()
        
    
    
    def _prev(self, _):
        self.ind -= 1
        if self.ind < 0:
            self.ind = len(self.files) - 1
        self.get_picker()
        self.load_cids()
        
    
    
    def save(self, _):
        
        # Save each individual file
        df = self.Pickers[0].DataFile.get_results()
        self.Pickers[0].DataFile.save_indices()
        self.Pickers[0].DataFile.save_output()
        
        for p in self.Pickers[1:]:
            df = df.append(p.DataFile.get_results())
            p.DataFile.save_indices()
            p.DataFile.save_output()
        
        # Save net output from multiple files
        writer = pd.ExcelWriter(self.folder + '/output.xlsx', engine='xlsxwriter')
        df.to_excel(writer, index=False, header=True, startcol=0)
        writer.save()
        print(f'Saved as {self.folder}/output.xlsx')
        print('')
        
        
        
        
        
        
    def reset(self, _):
        p = InteractivePicker(self.files[self.ind], self.fig, self.ax)
        self.Picker = p
        self.Pickers[self.ind] = p
        self.load_cids()






if __name__ == '__main__':

    fig, ax = plt.subplots(figsize=(5,6), dpi=100)    
    plt.subplots_adjust(bottom=0.3)
    
    index = Index(data_folder, fig, ax)
    
    
    axcalc = plt.axes([0.4, 0.1, 0.25, 0.05])
    bcalc = Button(axcalc, 'Reset')
    bcalc.on_clicked(index.reset)   
    
    axprev = plt.axes([0.15, 0.1, 0.25, 0.05])
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(index._prev) 
    
    axnext = plt.axes([0.65, 0.1, 0.25, 0.05])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(index._next) 
    
    axsave = plt.axes([0.4, 0.025, 0.25, 0.05])
    bsave = Button(axsave, 'Save')
    bsave.on_clicked(index.save)
    
    plt.show(block=True)



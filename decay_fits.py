import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import optimize, signal
import os
import warnings
import sys
plt.ion()


data_folder = r'C:\Users\orozc\Google Drive (miguelorozco@ucsb.edu)\Research\Spyder\Run'
plt.style.use('C:/Users/orozc/Google Drive (miguelorozco@ucsb.edu)/Research/Spyder/style.mplstyle')


FUNC_FIT = True      # Option to turn off fitting (True = Fit, False = Don't fit)
# FUNC = 'linear'
FUNC = 'monoexponential'
# FUNC = 'monoexp-linear'
# FUNC = 'monoexponential-inflection'
# FUNC = 'biexponential'
# FUNC = 'biexponential-inflection'
# FUNC = 'x-reciprocal'
# FUNC = 'Custom'

CHECK_FIT = False     # Plot individual fits for FUNC determination
SECOND_FUNC = 'monoexponential-inflection'

BASELINE_CORRECT = False
I_SCALE = 1e-3        # Conversion to amps. i.e. data in mA, I_SCALE = 1e-3
START_AFTER = 10      # cut off first (n) seconds
END_BEFORE = False       # cut off (n) seconds from data set or False
min_s_to_fit = 40      # Requires n seconds of data to accept the fit
FIT_T_MAX = 40        # Fit at most x seconds of data for each spike
DELAY = 0             # Points after "fast" spike to skip fitting on
thresh = 0.5          # Used to determine acceptable baseline "flatness"
                      # Smaller = more picky, need flatter baseline to accept spike

APPLY_FILTER     = False  # Apply a low-pass filter before drawing data
FILTER_FREQ      = 25     # All analysis still performed on raw data



def load_indices_from_file(data_file):
    
    def get_vals(path):
        if path.endswith('.csv'):
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path)
        indices = df['Index'].to_list()
        bounds = [None]*len(indices)
        if 'Right bound' in df:
            bounds = df['Right bound'].to_list()
        return indices, bounds
    
    # Automatically look for previously-generated file
    path = None
    old_file = data_file.replace('.txt', '_output.xlsx')
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
            print("Invalid input. Please enter output file name (q to exit)")
            indice_file = input('>>')
            if indice_file == 'q':
                print("Exiting the program...")
                sys.exit(0)
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


def lowpass(y, t, filter_freq):
    fs = 1/np.mean([t[i] - t[i-1] for i in range(1, len(t))])
    fc = filter_freq/(fs/2)
    try:
        b, a = signal.butter(8, fc)
        filt_y = signal.filtfilt(b, a, y, padlen=150)
        return filt_y
    except ValueError:
        print('Bad filter_freq, not filtering.')
        print(f'Valid filter_freq from 0 < f < {fs/2}')
        return y


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
    
    def _xreciprocal(x, a,b,c):
        return a * x / (b + a * x) + c
    
    def _custom(x, a,b,c,d,e):
        return a * np.exp(-b * x) + c * np.exp(-d * x) + e
    
    _func_mapping = {'linear': _linear,
                     'monoexponential': _monoexponential,
                     'monoexp-linear': _monoexplinear,
                     'monoexponential-inflection': _monoexpinflection,
                     'biexponential': _biexponential,
                     'biexponential-inflection': _biexpinflection,
                     'x-reciprocal': _xreciprocal,
                     'Custom': _custom,}
    
    _bound_mapping = {'linear': (-np.inf, [1., np.inf]),
                      'monoexponential': ([-np.inf, 0], [0, np.inf], [-np.inf,np.inf]),
                      'monoexp-linear': (-np.inf, [1., np.inf, np.inf]),
                      'monoexponential-inflection': (-np.inf, [1., np.inf, np.inf]),
                      'biexponential': (-np.inf, [0, np.inf, 0, np.inf, np.inf]),
                      'biexponential-inflection': (-np.inf, [0, np.inf, 0, np.inf, np.inf]),
                      'x-reciprocal': (-np.inf, [1., np.inf]),
                      'Custom': (-np.inf, [0, np.inf, 0, np.inf, np.inf]),}
    
    _param_mapping = {'linear': ['a/ A s-1', 'b/ A'],
                      'monoexponential': ['a/ A', 'b/ s-1', 'c/ A'],
                      'monoexp-linear': ['a/ A', 'b/ s-1', 'm/ A s-1', 'c/ A'],
                      'monoexponential-inflection': ['a/ A', 'b/ s-1', 'c/ A'],
                      'biexponential': ['a/ A', 'b/ s-1', 'c/ A', 'd/ s-1', 'e/ A'],
                      'biexponential-inflection': ['a/ A', 'b/ s-1', 'c/ A', 'd/ s-1', 'e/ A'],
                      'x-reciprocal': ['a', 'b', 'c'],
                      'Custom': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'],}
    
    def func(self):
        return self._func_mapping[FUNC]
    
    def bounds(self):
        return self._bound_mapping[FUNC]
    
    def param_names(self):
        return self._param_mapping[FUNC]
    
    
class Spike:
    
    def __init__(self, DataFile, idx, right_bound=None):
        self.DataFile = DataFile # Local ref to DataFile which contains this Point
        self.REMOVE = False
        
        
        self.idx         = idx  # int, index corresponding to peak current
        self.left_bound  = None # int, set by self.integrate_Hads(). For baseline
        self.right_bound = right_bound # int, set by self.get_right_bound() or 
                                       # loaded from previous result. 
                                       # For integration/ fitting
        
        self.H_integral     = None    # float
        self.cat_integral   = None    # float
        self.fit_params     = None    # array (popt output by curve_fit)
        self.chi_sq         = None    # float
        self.half_life      = None    # float
        
        self.artists = []   # List of matplotlib artists to draw
        
    
    
    def get_right_bound(self):
        if self.right_bound:
            return
        # Examine previous and next Point
        all_idxs = self.DataFile.spike_idxs()
        
        this_point_idx = [i for i, idx in enumerate(all_idxs) if idx == self.idx][0]
        
        # Maximum sample time for a single spike (FIT_T_MAX)
        max_sample_pts = int(FIT_T_MAX/self.DataFile.sample_rate)

        t = self.DataFile.t
        y = self.DataFile.i
        count = 0
        subtractions = []
        try:
            if self.idx < all_idxs[-1]-1:
                # Length in pts that corresponds to t_max in seconds
                len_max_time = len(t[self.idx:self.idx+max_sample_pts])
                
                # Load in the next spike data
                next_spike = self.DataFile.spikes[this_point_idx + 1]
                next_left_bound = next_spike.left_bound
                # print(this_point_idx, self.idx, len_max_time, next_left_bound)
                
                # Calculate the difference between y[left_bound] and y values after idx
                while count < min(len_max_time, len(t[self.idx:next_left_bound - 1])):
                    sub = abs(y[self.left_bound] - y[self.idx+count])
                    subtractions.append(sub)
                    count += 1
                    
                # Index where the absolute value of the subtractions is min
                thresh_max = 1*min(subtractions)
                for val in subtractions:
                    if val <= thresh_max:
                        min_sub = subtractions.index(val)
                        break
                self.right_bound = min(next_left_bound-1, self.idx + min_sub)                      
            
            # This point is the last one in the file
            if self.idx == all_idxs[-1]:
                # Length in pts that corresponds to t_max in seconds
                len_max_time = len(t[self.idx:self.idx+max_sample_pts])
                
                # Calculate the difference between y[left_bound] and y values after idx
                while count < min(len_max_time, len(t[self.idx:len(self.DataFile.i) - 1])):
                    sub = abs(y[self.left_bound] - y[self.idx+count])
                    subtractions.append(sub)
                    count += 1
                # Index where the absolute value of the subtractions is min
                thresh_max = 1*min(subtractions)
                for val in subtractions:
                    if val <= thresh_max:
                        min_sub = subtractions.index(val)
                        break
                self.right_bound = min(len(self.DataFile.i) - 1, self.idx + min_sub)
                
        except:
             print(f'Failed to find right bound at: {t[self.idx]} s')
             self.REMOVE = True                   

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
        # Calculate the slope for the first 30 points backwards
        while count < 30:
            m = (avg[self.idx-count-1]-avg[self.idx-count])/(t[self.idx-count-1]-t[self.idx-count])
            slopes.append(m)
            count += 1
        # print(slopes)
        
        # Calculate IP by comparing slope to first until < 10%
        cliff_pt = 0
        for slope in slopes:
              if slope > 0:
                  cliff_pt = slopes.index(slope)
                  break
                  # print(idx, cliff_pt, idx-cliff_pt)
                  # pos_slopes.append(cliff_pt)
        
        left_bound = self.idx - cliff_pt
        
        ## Plot the left bound point
        pt = Line2D([t[left_bound]], [y[left_bound]], marker='o', color='blue')
        verts = [(t[left_bound], y[left_bound]),
                  *zip(t[left_bound:self.idx + 1],
                       y[left_bound:self.idx + 1]),
                  (t[self.idx], y[left_bound])]
        poly = Polygon(verts, color= 'red', alpha = 0.5, ec = 'k')
        self.artists.extend([pt, poly])
        
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
        
        # Fit the line connecting Left_idx to Next_idx
        with warnings.catch_warnings():
            # Fit may fail at some points, remove those spikes manually later.
            warnings.simplefilter('ignore')
            (a,b) = np.polyfit(np.array([t[self.left_bound], t[self.right_bound]]),
                               np.array([y[self.left_bound], y[self.right_bound]]), 1)
        x1 = t[self.idx:self.right_bound]
        y1 = a*x1 + b
        # ax.plot(x1, y1, '-', color = 'r')
        
        ### Calculate integral (excludes initial sharp spike)      
        # area to draw
        verts = [(t[self.idx], y[self.left_bound]),
                 *zip(t[self.idx:self.right_bound],
                      y[self.idx:self.right_bound]),
                 (t[self.right_bound], y[self.right_bound])]
        poly = Polygon(verts, color= 'y', alpha = 0.5, ec = 'k')
        
        # actual calculation
        xs = t[self.idx:self.right_bound]
        ys = y[self.idx:self.right_bound] - y1
        integ = np.trapz(ys, xs)
        
        # self.artists.append(poly)
        self.cat_integral = integ
        self.artists.extend([poly,])
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
        
        if FUNC in ['monoexponential-inflection',
                    'biexponential-inflection',
                    'Custom',]:
            infl_pt = self.find_inflection(ts, data)
            
        if FUNC in ['linear',
                    'monoexponential',
                    'monoexp-linear',
                    'biexponential',
                    'x-reciprocal',]:
                infl_pt = DELAY
        exp_func = ExpFunc().func()
        # bounds   = ExpFunc().bounds()
       
        if FUNC == 'Custom':
            exp_func2 = ExpFunc()._func_mapping['monoexponential']
            
            #Function before inflection point
            exp_func3 = ExpFunc()._func_mapping['x-reciprocal']
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    popt, pcov = optimize.curve_fit(exp_func2, 
                                                    ts[infl_pt:],
                                                    data[infl_pt:], 
                                                    maxfev=100000,)
                    popt2, pcov2 = optimize.curve_fit(exp_func3, 
                                                    ts[:infl_pt],
                                                    data[:infl_pt], 
                                                    maxfev=100000,)
            except:
                # Failed to fit at this point
                print(f'Failed to fit {FUNC} at {t[self.idx]}')
                self.REMOVE = True
                return
            
            print(f'Params 1: {popt}')
            print(f'Params 2: {popt2}')
            print('')
            
            fit_y = exp_func2(ts[infl_pt:], *popt)
            residuals = abs((data[infl_pt:] - fit_y)/fit_y)
            
            fit_y2 = exp_func3(ts[:infl_pt], *popt2)
            residuals2 = abs((data[:infl_pt] - fit_y2)/fit_y2)
            
            # Update self.artists with fitted curve and marker point
    
            ln = Line2D(ts[infl_pt:]+t[self.idx], 
                        fit_y+baseline[self.idx+infl_pt:self.right_bound],
                        marker='o', color='gold', ms=3)
            pt = Line2D([t[self.idx]], [y[self.idx]], marker='o', color='red')
            
            ln2 = Line2D(ts[:infl_pt]+t[self.idx], 
                        fit_y2[:],
                        marker='o', color='green', ms=3)
            
            self.chi_sq = np.sum(residuals**2)/len(residuals)
            self.chi_sq2 = np.sum(residuals2**2)/len(residuals2)
            self.fit_params = (*popt,
                                *popt2
                               )
            self.artists.extend([ln, pt, ln2])
            
            #Finding half-life of impact
            delta_y = self.fit_params[2] + data[infl_pt]
            half_c = delta_y/2
            #print(half_c)
            #print(ts[np.where(data == half_c)])
            for i in range(infl_pt + 1, len(data)-1):
                # Check if a crossing occurs between consecutive points
                if (data[i-1] < half_c and data[i] > half_c):
                    half_life_idx = i
                    self.half_life = ts[half_life_idx]
                    break
                else:
                    self.half_life = None
            
        else:    
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    popt, pcov = optimize.curve_fit(exp_func, 
                                                    ts[infl_pt:],
                                                    data[infl_pt:], 
                                                    maxfev=100000,
                                                    # bounds=bounds
                                                    )
            except:
                # Failed to fit at this point
                print(f'Failed to fit {FUNC} at {t[self.idx]}')
                self.REMOVE = True
                return
                    
            fit_y = exp_func(ts[infl_pt:], *popt)
            residuals = abs((data[infl_pt:] - fit_y)/fit_y)
     
            # Update self.artists with fitted curve and marker point
    
            ln = Line2D(ts[infl_pt:]+t[self.idx], 
                        fit_y+baseline[self.idx+infl_pt:self.right_bound],
                        marker='o', color='gold', ms=3)
            pt = Line2D([t[self.idx]], [y[self.idx]], marker='o', color='red')
            
            self.chi_sq = np.sum(residuals**2)/len(residuals)
            self.fit_params = popt
            self.artists.extend([ln, pt])
            
            #Finding half-life of impact
            delta_y = self.fit_params[2] + data[infl_pt]
            half_c = delta_y/2
            #print(half_c)
            #print(ts[np.where(data == half_c)])
            for i in range(infl_pt + 1, len(data)-1):
                # Check if a crossing occurs between consecutive points
                if (data[i-1] < half_c and data[i] > half_c):
                    half_life_idx = i
                    self.half_life = ts[half_life_idx]
                    break
                else:
                    self.half_life = None
            
        if CHECK_FIT == True:
            self.analyze_fits(ts, data, baseline,
                              infl_pt, fit_y, self.chi_sq, self.fit_params)
        return
    
    def find_inflection(self, ts, data):
        '''
        Calculate the slope for the first 'n' points, where n = 30.

        Parameters
        ----------
        slopes : list
            List of slopes after each spike.

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
    
    def analyze_fits(self, ts, data, baseline, infl_pt, fit_y, chi_sq, fit_params):
        if self.REMOVE:
            # Already flagged this point for removal. Skip this analysis.
            return
        '''
        Plot individual fits on log scale and 
        compare Chi^2 for a second fitting function
        '''
        exp_func = ExpFunc()._func_mapping[SECOND_FUNC]

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            popt, pcov = optimize.curve_fit(exp_func, ts[infl_pt:],
                                            data[infl_pt:], maxfev=100000)
        mono_fit_y = exp_func(ts[infl_pt:], *popt)
        residuals = abs((data[infl_pt:] - mono_fit_y)/mono_fit_y)
        mono_chi_sq = np.sum(residuals**2)/len(residuals)
        diff = chi_sq < mono_chi_sq
        
        fig, axs = plt.subplots(3,1, sharex=True, dpi=100)    
        fig.subplots_adjust(hspace=0)
        axs[0].plot(ts, data, '.', color='k')
        axs[0].plot(ts[infl_pt:], fit_y, label = 'Normal', color='y')
        axs[0].set_box_aspect(0.33)
        axs[0].legend(loc='upper right', fontsize='x-small',
                      labelcolor='y', handlelength=0)
        
        # axs[1].plot(ts, np.log(abs(data)), '.', color='k')
        axs[1].plot(ts[infl_pt:], np.log(abs(mono_fit_y)),
                    label = f'Log scale\n{SECOND_FUNC}\nChi$^{2}$ = {round(mono_chi_sq, 7)}',
                    color = 'dodgerblue')
        axs[1].set_box_aspect(0.33)
        axs[1].legend(loc='upper right', fontsize='x-small',
                      labelcolor='dodgerblue', handlelength=0)
        
        # axs[2].plot(ts, np.log(abs(data)), '.', color='k')
        axs[2].plot(ts[infl_pt:], np.log(abs(fit_y)),
                    label = f'Log scale\n{FUNC}\nChi$^{2}$ = {round(chi_sq, 7)}\nMin = {diff}',
                    color='violet')
        axs[2].set_box_aspect(0.33)
        axs[2].legend(loc='upper right', fontsize='x-small',
                      labelcolor='violet', handlelength=0)
       
        fig.supylabel('Current (A)',)
        fig.supxlabel('Time (s)',)
        
        plt.show()
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
        
        if FUNC_FIT == False:
            d = {
                'Index': [self.idx],
                'Right bound': [self.right_bound],
                'Number': [0],
                'Time/s': [self.DataFile.t[self.idx]],
                'Catalytic area/ C': [self.cat_integral],
                'Hads integral/ C': [self.H_integral],
                }
        
        if FUNC_FIT != False:
            if FUNC != 'Custom':
                d = {
                    'Index': [self.idx],
                    'Right bound': [self.right_bound],
                    'Number': [0],
                    'Time/s': [self.DataFile.t[self.idx]],
                    **{name:[val] for name,val in zip(ExpFunc().param_names(),
                                                      self.fit_params)},
                    'Catalytic area/ C': [self.cat_integral],
                    'Hads integral/ C': [self.H_integral],
                    'Reduced Chi^2': [self.chi_sq],
                    'Half Life': [self.half_life],
                    }
            
            if FUNC == 'Custom':
                d = {
                    'Index': [self.idx],
                    'Right bound': [self.right_bound],
                    'Number': [0],
                    'Time/s': [self.DataFile.t[self.idx]],
                    **{name:[val] for name,val in zip(ExpFunc().param_names(),
                                                      self.fit_params)},
                    'Catalytic area/ C': [self.cat_integral],
                    'Hads integral/ C': [self.H_integral],
                    'Reduced Chi^2': [self.chi_sq],
                    '2nd Reduced Chi^2': [self.chi_sq2],
                    'Half Life': [self.half_life]
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
        df = pd.read_csv(self.file, names=('t', 'v', 'i', 'cycle number'), skiprows=1, sep='\t')
        df = df[df['t'] > START_AFTER]
        
        # Option to import partical data set
        if type(END_BEFORE) == int:
            df = df[df['t'] < END_BEFORE] 
        
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
                                                          threshold=10, influence=0.6)
        
        
        ii = input('Import indices? (y/n)>>')
        if ii == 'y':
            idxs, right_bounds = load_indices_from_file(self.file)
            if idxs:
                return idxs, right_bounds
            
        print('File not loaded. Finding new indices.')
        
        
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
            if any(abs(self.i[idx-20:idx+20]) > abs(self.i[idx])):
                
                i = np.where(abs(self.i) ==
                              max(abs(self.i[idx-20:idx+20])))[0][0]
                
                #print(f'Moving {idx} to {i}')
                idxs.remove(idx)
                idxs.append(i)
                idxs.sort()
        return idxs, [None]*len(idxs)
    
    
    
    def find_points(self):
        idxs, right_bounds = self.refine_peaks()
        for idx, bound in zip(idxs, right_bounds):
            self.spikes.append( Spike(DataFile=self, idx=idx,
                                      right_bound=bound) )
    
        
    
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
                    continue
                    # print(f'Error removing artist {artist}')
                    # print(e)
        self.spikes.remove(spike)
                
    
    
    def process_points(self):
        # Do H integration for everyone first. Sets left bounds of each 
        # spike for later funcs
        for spike in self.spikes:
            spike.integrate_Hads()
            
        for spike in self.spikes:
            spike.get_right_bound()   # Determine right border to integrate/fit to
            spike.integrate_cat()     # Get area under catalytic curve
            if FUNC_FIT == True:     # Skip fitting if not wanted
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
            df = pd.concat([df, spike.get_results()])
            file_col.append('')
        if len(df) > 2:
            file_col[1] = f'Fit: {FUNC}'
        if len(df) > 3:
            file_col[2] = f'Baseline correct: {BASELINE_CORRECT}'
        if len(df) > 4:
            file_col[3] = f'Fit_t_max: {FIT_T_MAX} s'
        
        df['File'] = file_col
        df['Number'] = [i+1 for i in range(len(file_col))]
        return df
    
    
    
    def save_output(self):
        df = self.get_results()
        path = f'{self.file.replace(".txt", "_output.xlsx")}'
        with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
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
        key = {'w':'up',
               'a':'left',
               's':'down',
               'd':'right'}.get(key, key)
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
        if APPLY_FILTER:
            self.ax.plot(self.DataFile.t, 
                         lowpass(self.DataFile.i, self.DataFile.t, FILTER_FREQ),
                         '.', color='k')
        else:
            self.ax.plot(self.DataFile.t, 
                         self.DataFile.i, '.', color='k')
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
            self.Picker = p
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
        # self.Pickers[0].DataFile.save_indices()
        self.Pickers[0].DataFile.save_output()
        
        for p in self.Pickers[1:]:
            df = pd.concat([df, p.DataFile.get_results()])
            # p.DataFile.save_indices()
            p.DataFile.save_output()
        
        # Save net output from multiple files
        with pd.ExcelWriter(self.folder + '/output.xlsx', engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, header=True, startcol=0)
        print(f'Saved as {self.folder}/output.xlsx')
        print('')
        
        
        
        
        
        
    def reset(self, _):
        p = InteractivePicker(self.files[self.ind], self.fig, self.ax)
        self.Picker = p
        self.Pickers[self.ind] = p
        self.load_cids()



def unbind_matplotlib_hotkeys():
    for param, vals in plt.rcParams.items():
        if 'keymap' in param:
            for item in vals:
                plt.rcParams[param].remove(item)



if __name__ == '__main__':
    unbind_matplotlib_hotkeys()

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



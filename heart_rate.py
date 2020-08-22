'''
The data collected from pulse sensor is stored in a file sig.xls in the coloumn named Raw_PPG_Green, the code below removes
noise from the signal and calculate heart rate of the person and also determine heart rate variability analysis
'''
import numpy as np
import peakutils as pk
from peakutils.plot import plot as pplot
from scipy.signal import butter, lfilter, freqz
from scipy import signal
from scipy.integrate import simps
from matplotlib import pyplot as plt
from matplotlib import style
from matplotlib.patches import Ellipse
import pandas as pd
from pandas import Series
import array
from hrvanalysis import get_time_domain_features
from hrvanalysis import plot_psd
import pyhrv.frequency_domain as fd
import seaborn as sns
from typing import List
import heartpy as hp
from hrvanalysis.extract_features import _get_freq_psd_from_nn_intervals
from collections import namedtuple

# to get data from file and store it in an array
Sampling_rate=50
df=pd.read_excel(r'sig.xls')
y = np.array(df.Raw_PPG_Green)
a=np.mean(y)
y=y-a
raw=y
n=y.size


#to plot raw data
t=np.linspace(1,n,n)
plt.plot(t,y)
plt.title ("Raw data")
plt.show()

#applying moving average filter to smoothen signals
w=20
mask=np.ones((1,w))/w
mask=mask[0,:]
convolved_data=np.convolve(y,mask,'same')
series=Series.to_frame(df.Raw_PPG_Green)
series['convolved_data']=convolved_data
plt.plot(series)
plt.title ("data when passed through moving average filter")
plt.show()
y=convolved_data/200


#apply band-pass butterworth filter to remove noise due to motion artifacts breathing rate
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

y = butter_bandpass_filter(y,0.6,4.0,Sampling_rate , 3)
plt.plot(t,y)
plt.title('estimate after passing through butterworth filter')
plt.show()

#to count the number of peaks
indexes = pk.indexes(y, thres=0.1, min_dist=20)
arr=indexes
num=arr
mun=array.array('f',arr)
rms=array.array('f',arr)
p=(len(t[indexes]))
pplot(t, y, indexes)
plt.title('estimating peaks')
plt.show()

#finding heart rate using heartpy and to calculate instantaneous BPM
working_data, measures = hp.process(y,Sampling_rate)
hp.plotter(working_data, measures)
i=0
while (i<p-1):
	num[i] = (Sampling_rate/(arr[i+1]-arr[i])*60)
	print (num[i])
	i=i+1

working_data, measures = hp.process(y, Sampling_rate, report_time=True)
print('breathing rate is: %s Hz' %measures['breathingrate'])

qwe=rms
i=len(rms)
rms[i-1]=rms[i-2]
i=0
while (i<p):
	rms[i]=rms[i]*1000
	i=i+1
time_domain_features = get_time_domain_features(rms)
print (time_domain_features)


#to calculate LF/HF ratio
#LF band is the frequency band between 0.04-0.15 Hz
#HF band is the frequency band between 0.15-0.4 Hz
# Named Tuple for different frequency bands
VlfBand = namedtuple("Vlf_band", ["low", "high"])
LfBand = namedtuple("Lf_band", ["low", "high"])
HfBand = namedtuple("Hf_band", ["low", "high"])
def plot_psd(nn_intervals: List[float], method: str = "welch", sampling_frequency: int = 7,
             interpolation_method: str = "linear", vlf_band: namedtuple = VlfBand(0.003, 0.04),
             lf_band: namedtuple = LfBand(0.04, 0.15), hf_band: namedtuple = HfBand(0.15, 0.40)):
    

    freq, psd = _get_freq_psd_from_nn_intervals(nn_intervals=nn_intervals, method=method,
                                                sampling_frequency=sampling_frequency,
                                                interpolation_method=interpolation_method)
    low, high = 0.04, 0.15
    idx_delta = np.logical_and(freq >= low, freq <= high)
    freq_res = 0.00001
    delta_power_1 = simps(psd[idx_delta],x=freq[idx_delta], dx=freq_res)
    
    low, high = 0.15, 0.4
    idx_delta = np.logical_and(freq >= low, freq <= high)
    freq_res = 0.00001
    delta_power_2 = simps(psd[idx_delta],x=freq[idx_delta], dx=freq_res)
    ratio = (delta_power_1/delta_power_2)
    print('LF/HF ratio: %.3f' % ratio)

    
    # Calculate of indices between desired frequency bands
    vlf_indexes = np.logical_and(freq >= vlf_band[0], freq < vlf_band[1])
    lf_indexes = np.logical_and(freq >= lf_band[0], freq < lf_band[1])
    hf_indexes = np.logical_and(freq >= hf_band[0], freq < hf_band[1])

    frequency_band_index = [vlf_indexes, lf_indexes, hf_indexes]
    label_list = ["VLF component", "LF component", "HF component"]

    
    if method == "welch":
        plt.title("FFT Spectrum : Welch's periodogram", fontsize=20)
        for band_index, label in zip(frequency_band_index, label_list):
            plt.fill_between(freq[band_index], 0, psd[band_index] / (1000 * len(psd[band_index])), label=label)
        plt.legend(prop={"size": 15}, loc="best")
        plt.xlim(0, hf_band[1])
        plt.show()
    else:
        raise ValueError("Not a valid method. Choose between 'lomb' and 'welch'")

plot_psd(rms, method="welch",sampling_frequency = Sampling_rate)



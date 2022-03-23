"""
This script shows how to calculate the Z-score health indicator as explained in https://www.mdpi.com/2218-6581/10/2/80
based on sample data.
"""
import pandas as pd
import numpy as np

from scipy import signal



def calc_stft_single(y, fs=10000, window='hamming', nperseg=50, noverlap=25):
    """
    Calculates the absolute amplitudes of the stft for a single measurement
    """
    f, t ,Sxx = signal.stft(y, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)
    return abs(Sxx), t, f

def mean_std_stft(dic_stfts):
    """
    Calc mean and std of stft dict
    """
    mean_stft = np.mean([value for value in dic_stfts.values()], axis=0)
    std_stft = np.std([value for value in dic_stfts.values()], axis=0)
    return mean_stft, std_stft

def calc_zscore_single(stft_new, mean_ref, std_ref):
    """
    Calc z-score for single stft
    """
    z_stft = (stft_new - mean_ref) / std_ref
    z_score = np.mean(np.abs(z_stft))
    return abs(z_stft), z_score


hdf = pd.HDFStore("Data/exampledata.hdf")
keys = hdf.keys()

stfts = {key:calc_stft_single(hdf[key].T.values[0])[0] for key in keys}
mean, std = mean_std_stft(stfts)

zscores = {key:calc_zscore_single(stfts[key], mean, std)[1] for key in keys}



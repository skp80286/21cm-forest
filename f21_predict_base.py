'''
Predict parameters fX and xHI from the 21cm forest data.
'''

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

import scipy.fftpack as fftpack
from scipy.stats import binned_statistic

def power_spectrum_1d(data, bins=10):
    """
    Calculate the 1D binned power spectrum of an array.

    Parameters:
    data: 1D array of data
    bins: Number of bins, or array of bin edges

    Returns:
    k: Array of wavenumbers (bin centers)
    power: Array of power spectrum values
    """

    # Calculate the Fourier transform of the data
    fft_data = fftpack.fft(data)

    # Calculate the power spectrum
    power = np.abs(fft_data)**2

    # Calculate the wavenumbers
    k = fftpack.fftfreq(len(data))

    # Bin the power spectrum
    power, bin_edges, _ = binned_statistic(np.abs(k), power, statistic='mean', bins=bins)
    k = 0.5 * (bin_edges[1:] + bin_edges[:-1])  # Bin centers

    return k, power

def plot_power_spectra(ps, ks, params):
    #print(f'shapes ps:{ps.shape} ks:{ks.shape}')
    print(params[0:2])
    logfxs = params[:,1]
    minfx = min(logfxs)
    maxfx = max(logfxs)
    plt.rcParams['figure.figsize'] = [15, 9]
    plt.title('power spectra.')
    for i, (row_ps, row_ks, row_fx) in enumerate(zip(ps, ks, logfxs)):
        plt.loglog(row_ks[1:]*1e6, row_ps[1:], linewidth=0.5, color=plt.cm.coolwarm((row_fx-minfx)/(maxfx-minfx)))
        break
    plt.xlabel('k (MHz$^{-1}$)')
    plt.ylabel('P$_{21}$(k)')
    plt.show()

def plot_los(los, freq_axis):
    plt.rcParams['figure.figsize'] = [15, 9]
    for f in los:
        plt.plot(freq_axis/1e6, f)
        break
    plt.xlabel('frequency[MHz]'), plt.ylabel('flux/S147')
    plt.show()

def summarize_test(y_pred, y_test):
    print(f"y_pred: {y_pred}")
    print(f"y_test: {y_test}")
    # Calculate R2 scores
    r2 = [r2_score(y_test[:, i], y_pred[:, i]) for i in range(2)]
    print("R2 Score: " + str(r2))

    # Calculate rmse scores
    rmse = np.sqrt((y_test - y_pred) ** 2)
    rmse_comb = rmse[:,0]+rmse[:,1]/5 # Weighted as per the range of params
    print(f"rmse_comb : {rmse_comb.shape}\n{rmse_comb[:10]}")
    df_y = pd.DataFrame()
    df_y = df_y.assign(actual_xHI=y_test[:,0])
    df_y = df_y.assign(actual_logfX=y_test[:,1])
    df_y = df_y.assign(pred_xHI=y_pred[:,0])
    df_y = df_y.assign(pred_logfX=y_pred[:,1])
    df_y = df_y.assign(rmse_xHI=rmse[:,0])
    df_y = df_y.assign(rmse_logfx=rmse[:,1])
    df_y = df_y.assign(rmse=rmse_comb)
    print(f"Describing test data with rmse: {df_y.describe()}")

    df_y_agg = df_y.groupby(["actual_xHI", "actual_logfX"])['rmse'].mean()
    df_y_agg.rename('agg_rmse', inplace=True)
    df_y = df_y.merge(df_y_agg, on=['actual_xHI', 'actual_logfX'], validate='many_to_one')
    print(f"Describing data with rmse: \n{df_y.describe()}\n{df_y.head()}")

    rmse_summary = df_y.groupby(["actual_xHI", "actual_logfX"]).agg({'agg_rmse':'mean','rmse_xHI':'mean','rmse_logfx': 'mean'})
    print(f"rmse Summary: \n{rmse_summary}")

    cmap = plt.get_cmap('viridis')
    rmse = df_y['agg_rmse']
    rmse_min = rmse.min()
    rmse_max = rmse.max()
    norm = plt.Normalize(rmse.min(), rmse.max())
    colors = cmap(norm(rmse))    

    plt.rcParams['figure.figsize'] = [15, 9]
    plt.scatter(df_y['pred_xHI'], df_y['pred_logfX'], marker="o", s=25, label='Predicted', c=colors)
    plt.plot([df_y['pred_xHI'], df_y['actual_xHI']], [df_y['pred_logfX'], df_y['actual_logfX']], 'r--', alpha=0.2)
    plt.scatter(df_y['actual_xHI'], df_y['actual_logfX'], marker="X", s=100, label='Actual', c=colors)
    plt.xlim(0, 1)
    plt.ylim(-4, 1)
    plt.xlabel('xHI')
    plt.ylabel('logfX')
    plt.title('Predictions')
    plt.legend()
    plt.colorbar(label=f'RMS Error ({rmse_min:.2f} to {rmse_max:.2f})')
    plt.savefig(f'{output_dir}/f21_prediction.png')
    plt.show()

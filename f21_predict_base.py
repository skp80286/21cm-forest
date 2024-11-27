'''
Predict parameters fX and xHI from the 21cm forest data.
'''

import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

import scipy.fftpack as fftpack
from scipy.stats import binned_statistic
import logging

logger = logging.getLogger(__name__)

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

def plot_single_power_spectrum(ps, ks, showplots=False, label=""):
    plt.rcParams['figure.figsize'] = [15, 9]
    plt.title(f'{label} - power spectrum')
    plt.loglog(ks[1:]*1e6, ps[1:], linewidth=0.5)
    plt.xlabel('k (MHz$^{-1}$)')
    plt.ylabel('P$_{21}$(k)')
    if showplots: plt.show()
    plt.clf()

def plot_power_spectra(ps, ks, params, output_dir=".", showplots=False, saveplots=True, label=""):
    #logger.info(f'shapes ps:{ps.shape} ks:{ks.shape}')
    logger.info(params[0:2])
    logfxs = params[:,1]
    minfx = min(logfxs)
    maxfx = max(logfxs)
    plt.rcParams['figure.figsize'] = [15, 9]
    plt.title(f'{label} - power spectra')
    for i, (row_ps, row_ks, row_fx) in enumerate(zip(ps, ks, logfxs)):
        color = None
        if maxfx > minfx: color=plt.cm.coolwarm((row_fx-minfx)/(maxfx-minfx))
        plt.loglog(row_ks[1:]*1e6, row_ps[1:], linewidth=0.5, color=color)
        if i> 10: break
    plt.xlabel('k (MHz$^{-1}$)')
    plt.ylabel('P$_{21}$(k)')
    if showplots: plt.show()
    if saveplots: plt.savefig(f"{output_dir}/power_spectra.png")
    plt.clf()

def plot_single_los(los, freq_axis, output_dir=".", showplots=False, saveplots=True, label=""):
    plt.rcParams['figure.figsize'] = [15, 9]
    plt.title(f'{label} - LoS')
    plt.plot(freq_axis/1e6, los)
    plt.xlabel('frequency[MHz]'), plt.ylabel('flux/S147')
    if showplots: plt.show()
    if saveplots: plt.savefig(f"{output_dir}/los.png")
    plt.clf()

def plot_los(los, freq_axis, output_dir=".", showplots=False, saveplots=True, label=""):
    plt.rcParams['figure.figsize'] = [15, 9]
    plt.title(f'{label} - LoS')
    for i, f in enumerate(los):
        plt.plot(freq_axis/1e6, f)
        if i> 10: break
    plt.xlabel('frequency[MHz]'), plt.ylabel('flux/S147')
    if showplots: plt.show()
    if saveplots: plt.savefig(f"{output_dir}/los.png")
    plt.clf()

def plot_predictions(df_y, colors):
    """
    Plot actual points with their corresponding predicted points and contours.
    
    Parameters:
    df_y: DataFrame containing columns: actual_xHI, actual_logfX, pred_xHI, pred_logfX
    """
    df_y['color'] = colors
    plt.rcParams['figure.figsize'] = [15, 9]
    
    # Group by actual points to get corresponding predictions
    for (actual_xHI, actual_logfX), group in df_y.groupby(['actual_xHI', 'actual_logfX']):
        # Get predicted points for this actual point
        pred_points = group[['pred_xHI', 'pred_logfX', 'color']].values
        
        # Create contour of predictions
        if len(pred_points) > 2:  # Need at least 3 points for ConvexHull
            try:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(pred_points)
                # Plot the convex hull boundary
                for simplex in hull.simplices:
                    plt.plot(pred_points[simplex, 0], pred_points[simplex, 1], 
                            'c--', alpha=0.5, linewidth=0.5, c=pred_points[2])
            except:
                pass  # Skip contour if ConvexHull fails
        
        # Plot predicted points
        plt.scatter(pred_points[:, 0], pred_points[:, 1], 
                   c=pred_points[:, 2], alpha=0.5, s=30)
        
        # Plot actual point
        plt.scatter(actual_xHI, actual_logfX, 
                marker='*', s=200)
    
    plt.xlim(0, 1)
    plt.ylim(-4, 1)
    plt.xlabel('xHI')
    plt.ylabel('logfX')
    plt.title('Actual Points (red stars) with Corresponding Predictions (blue dots)')
    
    # Add legend
    plt.scatter([], [], color='lightblue', alpha=0.5, label='Predictions')
    plt.scatter([], [], color='red', marker='*', s=200, label='Actual Points')
    plt.plot([], [], 'c--', alpha=0.5, label='Prediction Boundary')
    plt.legend()
    plt.show()

def summarize_test(y_pred, y_test, output_dir=".", showplots=False, saveplots=True):
    logger.info(f"y_pred: {y_pred}")
    logger.info(f"y_test: {y_test}")

    # Calculate R2 scores
    r2 = [r2_score(y_test[:, i], y_pred[:, i]) for i in range(2)]
    logger.info("R2 Score: " + str(r2))
    mean_r2_score = 0.5*(r2[0]+r2[1])
    logger.info("Mean R2 Score: " + str(mean_r2_score))

    # Calculate RMSE scores
    rmse = np.sqrt((y_test - y_pred) ** 2)
    # Weighted as per the range of params (xHI and logfX)
    rmse_comb = rmse[:,0] + rmse[:,1]/5 
    logger.info(f"rmse_comb : {rmse_comb.shape}\n{rmse_comb[:10]}")

    # Create arrays for actual and predicted values
    actual_xHI = y_test[:,0]
    actual_logfX = y_test[:,1]
    pred_xHI = y_pred[:,0]
    pred_logfX = y_pred[:,1]

    # Calculate aggregate RMSE for unique actual points
    unique_points = np.unique(y_test, axis=0)
    agg_rmse = []
    for point in unique_points:
        mask = np.all(y_test == point, axis=1)
        agg_rmse.append(np.mean(rmse_comb[mask]))
    
    logger.info("RMSE Summary:")
    logger.info(f"Mean RMSE: {np.mean(rmse_comb)}")
    logger.info(f"Min RMSE: {np.min(rmse_comb)}")
    logger.info(f"Max RMSE: {np.max(rmse_comb)}")

    if showplots or saveplots:
        # Calculate colors for plotting
        cmap = plt.get_cmap('viridis')
        rmse_min = np.min(rmse_comb)
        rmse_max = np.max(rmse_comb)
        norm = plt.Normalize(rmse_min, rmse_max)
        colors = cmap(norm(rmse_comb))

        # Plot predictions with colored points
        #plot_predictions(np.column_stack((actual_xHI, actual_logfX, pred_xHI, pred_logfX)), colors)
        
        # Create scatter plot
        plt.rcParams['figure.figsize'] = [15, 9]
        fig, ax = plt.subplots()
        plt.scatter(pred_xHI, pred_logfX, marker="o", s=25, label='Predicted', c=colors)
        plt.plot([pred_xHI, actual_xHI], [pred_logfX, actual_logfX], 'r--', alpha=0.2)
        plt.scatter(actual_xHI, actual_logfX, marker="x", s=100, label='Actual', c=colors)
        plt.xlim(0, 1)
        plt.ylim(-4, 1)
        plt.xlabel('xHI')
        plt.ylabel('logfX')
        plt.title('Predictions')
        plt.legend()
        plt.colorbar(label=f'RMS Error ({rmse_min:.2f} to {rmse_max:.2f})')
        if showplots: plt.show()
        if saveplots: plt.savefig(f'{output_dir}/f21_prediction.png')
        plt.clf()
    return mean_r2_score
    
    """
def summarize_test(y_pred, y_test, output_dir=".", showplots=False):
    logger.info(f"y_pred: {y_pred}")
    logger.info(f"y_test: {y_test}")

    # Calculate R2 scores
    r2 = [r2_score(y_test[:, i], y_pred[:, i]) for i in range(2)]
    logger.info("R2 Score: " + str(r2))

    # Calculate rmse scores
    rmse = np.sqrt((y_test - y_pred) ** 2)
    rmse_comb = rmse[:,0]+rmse[:,1]/5 # Weighted as per the range of params
    logger.info(f"rmse_comb : {rmse_comb.shape}\n{rmse_comb[:10]}")
    df_y = pd.DataFrame()
    df_y = df_y.assign(actual_xHI=y_test[:,0])
    df_y = df_y.assign(actual_logfX=y_test[:,1])
    df_y = df_y.assign(pred_xHI=y_pred[:,0])
    df_y = df_y.assign(pred_logfX=y_pred[:,1])
    df_y = df_y.assign(rmse_xHI=rmse[:,0])
    df_y = df_y.assign(rmse_logfx=rmse[:,1])
    df_y = df_y.assign(rmse=rmse_comb)
    logger.info(f"Describing test data with rmse: {df_y.describe()}")

    df_y_agg = df_y.groupby(["actual_xHI", "actual_logfX"])['rmse'].mean()
    df_y_agg.rename('agg_rmse', inplace=True)
    df_y = df_y.merge(df_y_agg, on=['actual_xHI', 'actual_logfX'], validate='many_to_one')
    logger.info(f"Describing data with rmse: \n{df_y.describe()}\n{df_y.head()}")


    rmse_summary = df_y.groupby(["actual_xHI", "actual_logfX"]).agg({'agg_rmse':'mean','rmse_xHI':'mean','rmse_logfx': 'mean'})
    logger.info(f"rmse Summary: \n{rmse_summary}")

    cmap = plt.get_cmap('viridis')
    rmse = df_y['agg_rmse']
    rmse_min = rmse.min()
    rmse_max = rmse.max()
    norm = plt.Normalize(rmse.min(), rmse.max())
    colors = cmap(norm(rmse))    

    plot_predictions(df_y[["actual_xHI", "actual_logfX", "pred_xHI", "pred_logfX"]], colors)
    plt.rcParams['figure.figsize'] = [15, 9]
    fig, ax = plt.subplots()
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
    if showplots: plt.show()
    plt.savefig(f'{output_dir}/f21_prediction.png')
    plt.clf()
"""

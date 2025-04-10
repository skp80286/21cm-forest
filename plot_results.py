
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as clr
from sklearn.metrics import r2_score

import argparse
import logging
from datetime import datetime
import f21_predict_base as base
import PS1D
import F21Stats as f21stats

logger = logging.Logger("main")


def split_key(key):
    # Split the key into two float values
    xHI, logfX = map(float, key.split('_'))
    return xHI, logfX

def create_key(xHI, logfX):
    return f"{xHI:.2f}_{logfX:.2f}" 


selected_keys = ["0.80_-3.00","0.11_-3.00","0.52_-2.00","0.80_-1.00","0.11_-1.00"]
def calc_squared_error(predictions, y_test):
    # Create keys for each row
    keys = [create_key(y_test[i][0], y_test[i][1]) for i in range(len(y_test))]
    
    # Calculate mean predictions for each unique key
    unique_keys = set(keys)
    mean_predictions = {key: [] for key in unique_keys}
    
    for i, key in enumerate(keys):
        mean_predictions[key].append(predictions[i])
    
    mean_values = {key: np.mean(values, axis=0) for key, values in mean_predictions.items()}
    
    # Calculate squared error
    total_squared_error = 0
    for i, key in enumerate(selected_keys):
        xHI, logfX = split_key(key)
        squared_error = (xHI - mean_values[key][0])**2 + (logfX - mean_values[key][1])**2 
        logger.info(f"key: {key}, mean_values: {mean_values[key]}, squared_error: {squared_error}")
        total_squared_error += squared_error
    
    logger.info(f"Total Squared Error (Means): {total_squared_error}")
    mse_means = total_squared_error/len(selected_keys)
    logger.info(f"MSE (Means): {mse_means}")
    rmse_means = np.sqrt(mse_means)
    logger.info(f"RMSE (Means): {rmse_means}")
    return total_squared_error, rmse_means

def mean_squared_error(predictions, y_test):
    mse = calc_squared_error(predictions, y_test)/5.0
    return mse

def rmse_all(y_pred, y_test):
    rmse_all = np.sqrt(np.mean((y_test - y_pred) ** 2))
    logger.info(f"RMSE (all predictions): {rmse_all}")
    return rmse_all

def summarize_test_1000(y_pred, y_test, output_dir=".", showplots=False, saveplots=True, label=""):
    """
    Analyze predictions by grouping them for each unique test point.
    
    Parameters:
    y_pred: numpy array of predictions (n, 2)
    y_test: numpy array of test values (n, 2)
    """
    logger.info(f"summarize_test_1000: Summarizing results pred shape:{y_pred.shape} actual shape: {y_test.shape}")

    # Create unique identifier for each test point
    unique_test_points = np.unique(y_test[:,:2], axis=0)
    logger.info(f"Number of unique test points: {len(unique_test_points)}")
    logger.info(f"Unique test points: {unique_test_points}")
    
    # Calculate mean predictions for each unique test point
    mean_predictions = []
    std_predictions = []
    
    for test_point in unique_test_points:
        # Find all predictions corresponding to this test point
        mask = np.all(y_test == test_point, axis=1)
        corresponding_preds = y_pred[mask]
        logger.info(f"Test point: {test_point}, Number of preds: {len(corresponding_preds)}")

        # Calculate mean and std of predictions
        mean_pred = np.mean(corresponding_preds, axis=0)
        std_pred = np.std(corresponding_preds, axis=0)
        logger.info(f"Test point: {test_point}, Number of preds: {len(corresponding_preds)}, Mean: {mean_pred}, Std: {std_pred}")
        
        mean_predictions.append(mean_pred)
        std_predictions.append(std_pred)
    
    mean_predictions = np.array(mean_predictions)
    std_predictions = np.array(std_predictions)
    
    # Calculate R2 score using mean predictions
    r2_means = [r2_score(unique_test_points[:, i], mean_predictions[:, i]) for i in range(2)]
    r2_means_combined = np.mean(r2_means)
    logger.info(f"R2 Score (using means): {r2_means}, combined score: {r2_means_combined}")
    r2_total = r2_score(y_test, y_pred)
    logger.info(f"R2 Score for All Predictions: {r2_total}")
    tse, rmse_means =  calc_squared_error(y_pred, y_test)
    rmse_total = rmse_all(y_pred, y_test)
    
    # Plotting
    if showplots or saveplots:
        plt.rcParams['figure.figsize'] = [8, 8]
        fig, ax = plt.subplots()
        
        num_points = len(unique_test_points)
        colors = plt.cm.rainbow(np.linspace(0, 1, num_points))
        
        # For each unique test point, create contours of predictions
        for i, test_point in enumerate(unique_test_points):
            # Find all predictions corresponding to this test point
            mask = np.all(y_test == test_point, axis=1)
            corresponding_preds = y_pred[mask]

            x, y = corresponding_preds[:, 0], corresponding_preds[:, 1]
            # Step 2: Normalize the histogram to get the probability density
            # Get the bin counts and edges
            counts, xedges, yedges = np.histogram2d(x, y, bins=18)
            # Normalize the counts to create a probability density function
            pdf = counts / np.sum(counts)
            # Find the levels that correspond to 68% and 95% confidence intervals
            level_68 = np.percentile(pdf, 68)
            level_95 = np.percentile(pdf, 95)
            # Step 4: Create the contour plot
            plt.contourf(xedges[:-1], yedges[:-1], pdf.T, levels=[level_68,level_95,np.max(pdf)],colors=[colors[i],colors[i]],alpha=[0.3,0.6])
            plt.contour(xedges[:-1], yedges[:-1], pdf.T, levels=[level_68,level_95,np.max(pdf)],colors=[colors[i],colors[i]],linewidths=0.5)
            plt.xlim(0,1)
            plt.ylim(-4,0)
            plt.tick_params(axis='both', direction='in', length=10)  # Inward ticks with length 10
            plt.xlabel(r"$\langle x_{HI}\rangle$")
            plt.ylabel(r"$log_{10}(f_X)$")


        # Plot mean predictions
        plt.scatter(mean_predictions[:, 0], mean_predictions[:, 1], 
                   marker="x", s=200, label='Mean Predicted', alpha=1, c=colors)

        # Plot actual points
        plt.scatter(unique_test_points[:, 0], unique_test_points[:, 1], 
                   marker="*", s=200, label='Actual', c=colors)
        
        plt.xlim(0, 1)
        plt.ylim(-4, 1)
        
        plt.xlabel(r'$\langle x_{HI}\rangle$', fontsize=18)
        plt.ylabel(r'$log_{10}(f_X)$', fontsize=18)
        plt.yticks(fontsize=18)
        plt.xticks(fontsize=18)
        plt.title('Mean Predictions with  ±1σ and ±2σ Contours', fontsize=18)

        # Overlay RMSE, R2 means, and R2 total on the graph
        textstr = f'RMSE: {rmse_means:.4f}\nR²: {r2_means_combined:.4f}'
        props = dict(facecolor='white', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=18,
                verticalalignment='top', bbox=props)

        plt.legend()
        
        if saveplots: plt.savefig(f'{output_dir}/f21_prediction_means{label}.png')
        if showplots: plt.show()
        plt.close()

        # Make a scatter plot
        plt.rcParams['figure.figsize'] = [8, 8]
        fig, ax = plt.subplots()
        colors = plt.cm.rainbow(np.linspace(0, 1, num_points))
        # Plot all 10000 predctions
        for i, test_point in enumerate(unique_test_points):
            # Find all predictions corresponding to this test point
            mask = np.all(y_test == test_point, axis=1)
            corresponding_preds = y_pred[mask]
            plt.scatter(corresponding_preds[:, 0], corresponding_preds[:, 1], 
                marker="o", s=25, alpha=0.01, c=colors[i])
            
        # Plot mean predictions
        plt.scatter(mean_predictions[:, 0], mean_predictions[:, 1], 
                marker="o", edgecolor='b', s=100, label='Mean Predicted', alpha=1, c=colors)
        # Plot actual points
        plt.scatter(unique_test_points[:, 0], unique_test_points[:, 1], 
                marker="*", edgecolor='b', s=200, label='Actual', c=colors)

        plt.xlim(0, 1)
        plt.ylim(-4, 1)

        plt.xlabel(r'$\langle x_{HI}\rangle$', fontsize=18)
        plt.ylabel(r'$log_{10}(f_X)$', fontsize=18)
        plt.yticks(fontsize=18)
        plt.xticks(fontsize=18)
        plt.title('Parameter Predictions', fontsize=18)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=18,
                verticalalignment='top', bbox=props)
        plt.legend()

        if saveplots: plt.savefig(f'{output_dir}/f21_prediction_means_scatter_{label}.png')
        if showplots: plt.show()
        plt.close()
    # Log statistics
    logger.info("\nPrediction Statistics:")
    logger.info(f"Mean xHI std: {np.mean(std_predictions[:, 0]):.4f}")
    logger.info(f"Mean logfX std: {np.mean(std_predictions[:, 1]):.4f}")
    
    return r2_means_combined

markers=['o', 'x', '*']
def plot_power_spectra(ps_set, ks, title, labels, xscale='log', yscale='log', showplots=False, saveplots=True, output_dir='tmp_output'):
    #print(f"plot_power_spectra: shapes: {ps_set.shape},{ks.shape}")

    base.initplt()
    plt.title(f'{title}')
    if len(ps_set.shape) > 1:
        for i, ps in enumerate(ps_set):

            if labels is not None: label = labels[i]
            row_ks = None
            if ks is not None:
                if len(ks.shape) > 1: row_ks = ks[i]
                else: row_ks = ks
            plt.plot(row_ks*1e6, ps, label=label, marker=markers[i% len(markers)], alpha=0.5)
    else:
        row_ks = None
        if ks is not None:
                if len(ks.shape) > 1: row_ks = ks[0]
                else: row_ks = ks
        plt.plot(ks*1e6, ps, label=label, marker='o')
        #plt.scatter(ks[1:]*1e6, ps[1:], label=label)
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.xlabel(r'k (Hz$^{-1}$)')
    plt.ylabel(r'$kP_{21}$')
    plt.legend()
    if showplots: plt.show()
    if saveplots: plt.savefig(f"{output_dir}/reconstructed_ps_{title}.png")
    plt.close()

def plot_denoised_ps(los_test, y_test_so, y_pred_so, samples=1, showplots=False, saveplots=True, label='', signal_bandwidth=20473830.8, output_dir='tmp_output'):
    ks_noisy, ps_noisy = PS1D.get_P_set(los_test, signal_bandwidth, scaled=True)
    #logger.info(f'get_P_set: {ks_noisy.shape}, {ps_noisy.shape},')
    ks_noisy, ps_noisy = f21stats.logbin_power_spectrum_by_k(ks_noisy, ps_noisy)
    #logger.info(f'get_P_set: {ks_noisy.shape}, {ps_noisy.shape},')
    ps_noisy_mean = np.mean(ps_noisy, axis=0)
    ks_so, ps_so = PS1D.get_P_set(y_test_so, signal_bandwidth, scaled=True)
    ks_so, ps_so = f21stats.logbin_power_spectrum_by_k(ks_so, ps_so)
    ps_so_mean = np.mean(ps_so, axis=0)
    ks_pred, ps_pred = PS1D.get_P_set(y_pred_so, signal_bandwidth, scaled=True)
    ks_pred, ps_pred = f21stats.logbin_power_spectrum_by_k(ks_pred, ps_pred)
    ps_pred_mean = np.mean(ps_pred, axis=0)

    plot_power_spectra(np.vstack((ps_so_mean,ps_noisy_mean,ps_pred_mean)), ks_noisy[0,:], title=label, labels=["signal-only", "noisy-signal", "reconstructed"])

def plot_denoised_los(los_test, y_test_so, y_pred_so, samples=1, showplots=False, saveplots=True, label='', output_dir='tmp_output'):
    for i, (noisy, test, pred) in enumerate(zip(los_test[:samples], y_test_so[:samples], y_pred_so[:samples])):
        plt.figure()
        plt.rcParams['figure.figsize'] = [15, 9]
        plt.title(f'Reconstructed LoS vs Actual Noiseless LoS {label}')
        plt.plot(noisy-0.01, label='Signal with Noise')
        plt.plot(test, label='Actual signal')
        plt.plot(pred+0.01, label='Reconstructed')
        plt.plot(test-pred+0.98, label='Reconstructed - Signal')
        plt.xlabel('frequency'), 
        plt.ylabel('flux/S147')
        plt.legend()
        if showplots: plt.show()
        if saveplots: plt.savefig(f"{output_dir}/reconstructed_los_{label}.png")
        if i> 5: break
        plt.close()

"""Sample plotting code"""
if __name__ == "__main__":
    all_results = np.loadtxt("saved_output/unet_inference/test_results.csv", delimiter=",", skiprows=1)
    print(f"loaded data shape: {all_results.shape}")
    y_pred = all_results[:,:2]
    y_test = all_results[:,2:4]
    summarize_test_1000(y_pred, y_test, output_dir="./tmp_out", showplots=True, saveplots=True, label="test")

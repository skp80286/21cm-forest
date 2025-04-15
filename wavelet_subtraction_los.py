import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import mean_squared_error
import argparse
import F21DataLoader as dl
import glob
import f21_predict_base as base
import plot_results as pltr
import F21Stats as f21stats
import pywt 

def denoise_with_wavelet(all_los):
    # Perform wavelet transform
    coeffs = pywt.wavedec(all_los, 'haar', level=2)  # Using Haar wavelet for example
    # Thresholding
    threshold = 0.4  # Set a threshold value
    coeffs_thresholded = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    # Inverse wavelet transform
    denoised_los = pywt.waverec(coeffs_thresholded, 'haar')
    return denoised_los

def load_dataset(datafiles, psbatchsize=100, ps_bins=None, limitsamplesize=None, logbinning=False, logpower=False):
    # Create processor with desired number of worker threads
    processor = dl.F21DataLoader(max_workers=1, psbatchsize=psbatchsize, skip_ps=False, ps_bins=None, limitsamplesize=limitsamplesize, scale_ps=True)#, limitsamplesize=4)
        
    # Process all files and get results 
    results = processor.process_all_files(datafiles)
        
    # Access results
    #print(f'{results.keys()}')
    all_ks = results['ks']
    all_los = results['los']
    all_freq = results['freq_axis']
    all_ps = results['ps']
    #ps_std = results['ps_std']
    #ps_plus_std = all_ps + ps_std
    #ps_minus_std = all_ps - ps_std
    all_params = results['params']
    #plot_los(all_ps[0], freq_axis)
    """
    print(f"sample ks:{all_ks[0]}")
    print(f"sample ps:{all_ps[0,:]}")
    print(f"sample params:{all_params[0]}")
    print(f"sample los:{all_los[0]}")
    print(f"sample freq:{all_freq}")
    """
    #base.plot_los(all_los[:1], all_freq[0,:], showplots=True, saveplots = False, label=f"{all_params[0]}")
    #base.plot_power_spectra(all_ps[:1], all_ks[:1], all_params[:1], showplots=True, saveplots = False, label=f"{all_params[0]}")
    if logbinning:
        all_ks, all_ps = f21stats.logbin_power_spectrum_by_k(all_ks, all_ps, silent=False)
    if logpower:
        all_ps = np.log10(np.clip(all_ps, 1e-20, None))
    print(f"\nCombined data shape: {all_los.shape}")
    print(f"Combined parameters shape: {all_params.shape}")
    all_ps = np.nan_to_num(all_ps)
    return (all_params, all_freq, all_los, all_ps, all_ks)

def bootstrap(ps):
    if ps.shape[0] >= ps.shape[1]: return ps
    dim = ps.shape[1]
    print(f"bootstrapping. {ps.shape}")
    reps = dim//ps.shape[0]
    r = np.tile(ps, (reps, 1))
    print(r.shape)
    r = np.vstack((r, ps[:dim-r.shape[0]]))
    print(r.shape)
    return r

# main code start here
np.random.seed(42)

parser = argparse.ArgumentParser(description='Predict reionization parameters from 21cm forest')
parser.add_argument('-p', '--path', type=str, default='../data/21cmFAST_los/F21_noisy/', help='filepath')
parser.add_argument('-z', '--redshift', type=float, default=6, help='redshift')
parser.add_argument('-d', '--dvH', type=float, default=0.0, help='rebinning width in km/s')
parser.add_argument('-r', '--spec_res', type=int, default=8, help='spectral resolution of telescope (i.e. frequency channel width) in kHz')
parser.add_argument('-t', '--telescope', type=str, default='uGMRT', help='telescope')
parser.add_argument('-s', '--s147', type=float, default=64.2, help='intrinsic flux of QSO at 147Hz in mJy')
parser.add_argument('-a', '--alpha_r', type=float, default=-0.44, help='radio spectral index of QSO')
parser.add_argument('-i', '--t_int', type=float, default=500, help='integration time of obsevation in hours')
parser.add_argument('-f', '--log_fx', type=str, default='-1.00', help='log10(f_X)')
parser.add_argument('-x', '--xHI', type=str, default='0.25', help='mean neutral hydrogen fraction')
parser.add_argument('-m', '--runmode', type=str, default='', help='')
parser.add_argument('--use_log_binning', action='store_true', help='log binning of PS')
parser.add_argument('--use_log_ps', action='store_true', help='log PS values')
parser.add_argument('--create_noisy_ps', action='store_true', help='create noisy PS using signal and noise')
parser.add_argument('--model', type=str, default='PCA', help='')

args = parser.parse_args()
output_dir = base.create_output_dir(args)
logger = base.setup_logging(output_dir)
#logger = base.setup_logging(output_dir)
# File path
so_file_path = str('%sF21_signalonly_21cmFAST_200Mpc_z%.1f_fX%s_xHI%s_%dkHz.dat' % 
               (args.path, args.redshift,args.log_fx, args.xHI, args.spec_res))
no_file_path = str('%sF21_noiseonly_21cmFAST_200Mpc_z%.1f_%s_%dkHz_t%dh_Smin%.1fmJy_alphaR%.2f.dat' % 
               (args.path, args.redshift, args.telescope, args.spec_res, args.t_int, args.s147, args.alpha_r))
file_path = str('%sF21_noisy_21cmFAST_200Mpc_z%.1f_fX%s_xHI%s_%s_%dkHz_t%dh_Smin%.1fmJy_alphaR%.2f.dat' % 
               (args.path, args.redshift,args.log_fx, args.xHI, args.telescope, args.spec_res, args.t_int, args.s147, args.alpha_r))

print(file_path)
# Read the file and extract data
so_psbatchsize = 10
if args.create_noisy_ps: so_psbatchsize = 1
params, freq, los, ps, ks = load_dataset([file_path], psbatchsize=1, limitsamplesize=1000, ps_bins=None, logbinning=args.use_log_binning, logpower=args.use_log_ps)
_, _, losso, psso, _ = load_dataset([so_file_path], psbatchsize=so_psbatchsize, limitsamplesize=1000, ps_bins=None, logbinning=args.use_log_binning, logpower=args.use_log_ps)
_, _, losno, psno, _ = load_dataset([no_file_path], psbatchsize=1000, limitsamplesize=1000, ps_bins=None, logbinning=args.use_log_binning, logpower=args.use_log_ps)
if args.create_noisy_ps: ps = psso + psno
ps_mean = np.mean(ps, axis=0)
psso_mean = np.mean(psso, axis=0)
#ps = bootstrap(ps)
denoised_los = denoise_with_wavelet(los) 

print(f"Shape of los: {los.shape}")
print(f"Shape of denoised los: {denoised_los.shape}")

sightline_index=0
freq_axis = freq[sightline_index]
base.initplt()
plt.plot(freq_axis, los[sightline_index], label=f"Noisy", color="blue", alpha=0.4)
plt.plot(freq_axis, losso[sightline_index]+0.1, label=f"Pure Signal", color="green", alpha=0.4)
plt.plot(freq_axis, denoised_los[sightline_index]+0.1, label=f"Denoised", color="orange", alpha=0.4)
plt.xlabel("Hz")
plt.ylabel("Normalized Flux")
plt.title(f"Original Signal LoS vs De-Noised\n(z={args.redshift}, xHI_mean={args.xHI}, logfX={args.log_fx})")
plt.legend()
plt.grid(True)
plt.savefig(f"{output_dir}/denoised_los.pdf", format="pdf")
plt.show()
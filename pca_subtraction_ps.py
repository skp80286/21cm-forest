import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import argparse
import F21DataLoader as dl
import glob
import f21_predict_base as base
import plot_results as pltr
import F21Stats as f21stats

def load_dataset(datafiles, psbatchsize=10, ps_bins=None, limitsamplesize=None, logbinning=False):
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
    print(f"\nCombined data shape: {all_los.shape}")
    print(f"Combined parameters shape: {all_params.shape}")
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
parser.add_argument('-l', '--use_log_binning', action='store_true', help='log binning of PS')

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
params, freq, los, ps, ks = load_dataset([file_path], psbatchsize=1, limitsamplesize=1000, ps_bins=None, logbinning=args.use_log_binning)
_, _, losso, psso, _ = load_dataset([so_file_path], psbatchsize=1000, limitsamplesize=1000, ps_bins=None, logbinning=args.use_log_binning)
_, _, losno, psno, _ = load_dataset([no_file_path], psbatchsize=1000, limitsamplesize=1000, ps_bins=None, logbinning=args.use_log_binning)
ps_mean = np.mean(ps, axis=0)
ps = bootstrap(ps)

print(f"Shape of ps: {ps.shape}")
print(f"Shape of psso: {psso.shape}")

base.initplt()
plt.loglog(ks[0], psso[0], label=f"Signalonly", color="green", alpha=0.4)
plt.loglog(ks[0], psno[0], "r--", linewidth=0.5, label="Noise")
plt.xlabel("k (Hz^-1)")
plt.ylabel("kP(k)")
plt.title(f"Original Signal PS without Noise\n(z={args.redshift}, xHI_mean={args.xHI}, logfX={args.log_fx})")
plt.legend()
plt.grid(True)
plt.savefig(f"{output_dir}/sig_only_ps.png")
plt.show()

min_mse = 6e23
best_component = -1
best_reconstruction = None
best_reconstruction_type = None
# Perform PCA on los_arr (1000 sightlines, 2762 features each)
num_components = min(ps.shape[0], ps.shape[1])
pca = PCA(n_components=num_components)
transformed_data = pca.fit_transform(ps)
print(f"Shape of transformed data: {transformed_data.shape}")
for c in range(num_components):

    if c%100 == 0: 
        print(f"num_components={num_components}")
        print(f"min_mse={min_mse}, best_component={best_component}")

    #print(f"Shape of transformed_data {transformed_data.shape}")
    component = transformed_data[c].reshape((1,ps.shape[1]))
    #print(f"Shape of component: {component.shape}")
    #reconstructed_data = pca.inverse_transform(component)
    mse = mean_squared_error(psso, component)
    if mse < min_mse: 
        min_mse = mse
        best_component = c
        best_reconstruction = component
        best_reconstruction_type = "transformed"
    
    subtracted_data = ps_mean - component
    mse = mean_squared_error(psso, subtracted_data)
    if mse < min_mse: 
        min_mse = mse
        best_component = c
        best_reconstruction = subtracted_data
        best_reconstruction_type = "subtracted"

# Compare original and reconstructed sightlines
sightline_index = 0  # Choose a sightline to compare
noisy_ps = ps_mean
ks = ks[sightline_index]
reconstructed_ps = best_reconstruction
print(f"min_mse={min_mse}, best_component={best_component}, best_reconstruction_type={best_reconstruction_type}")
# Plot the original and reconstructed sightlines
base.initplt()
plt.loglog(ks, noisy_ps, label="Original PS Mean", color="blue", alpha=0.4)
plt.loglog(ks, reconstructed_ps[0], label=f"Reconstructed (using mode # {best_component})", color="red", alpha=0.4)
plt.loglog(ks, psso[0], label=f"Signalonly", color="green", alpha=0.4)
plt.loglog(ks, psno[0], "r--", linewidth=0.5, label="Noise")
plt.xlabel("k (Hz^-1)")
plt.ylabel("kP(k)")
plt.title(f"Comparison of Original and Best Reconstructed PS\n(z={args.redshift}, xHI_mean={args.xHI}, logfX={args.log_fx})")
plt.legend()
plt.grid(True)
plt.savefig(f"{output_dir}/orig_vs_best_reconstr.png")
plt.show()

# # Plot explained variance
# explained_variance = pca.explained_variance_ratio_
# plt.figure(figsize=(8, 5))
# plt.bar(range(1, len(explained_variance) + 1), explained_variance * 100)
# plt.xlabel("Principal Component")
# plt.ylabel("Explained Variance (%)")
# plt.title("Explained Variance by Principal Components")
# plt.show()
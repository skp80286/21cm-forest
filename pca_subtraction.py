import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import argparse

# Function to read the 21cm LOS file
def get_los(datafile: str):
    data = np.fromfile(datafile, dtype=np.float32)
    # Extract parameters
    i = 0
    z, xHI_mean, logfX = -999, -999, -999
    if "noiseonly" not in datafile:
        z = data[i]
        i += 1        # redshift
        xHI_mean = data[i] # mean neutral hydrogen fraction
        i += 1
        logfX = data[i]    # log10(f_X)
        i += 1
    Nlos = int(data[i])  # Number of lines-of-sight
    i += 1
    Nbins = int(data[i])  # Number of pixels/cells/bins in one line-of-sight
    x_initial = i + 1
    if len(data) != x_initial + (1 + Nlos) * Nbins:
        error_msg = f"Error: Found {len(data)} fields, expected {x_initial + (1 + Nlos) * Nbins}. x_initial={x_initial}, Nlos={Nlos}, Nbins={Nbins}. File may be corrupted: {datafile}"
        raise ValueError(error_msg)
    # Extract frequency axis and F21 data
    freq_axis = data[(x_initial + 0 * Nbins):(x_initial + 1 * Nbins)]
    los_arr = np.reshape(data[(x_initial + 1 * Nbins):(x_initial + 1 * Nbins + Nlos * Nbins)], (Nlos, Nbins))
    return z, xHI_mean, logfX, freq_axis, los_arr

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

args = parser.parse_args()
# File path
so_file_path = str('%sF21_signalonly_21cmFAST_200Mpc_z%.1f_fX%s_xHI%s_%dkHz.dat' % 
               (args.path, args.redshift,args.log_fx, args.xHI, args.spec_res))
file_path = str('%sF21_noisy_21cmFAST_200Mpc_z%.1f_fX%s_xHI%s_%s_%dkHz_t%dh_Smin%.1fmJy_alphaR%.2f.dat' % 
               (args.path, args.redshift,args.log_fx, args.xHI, args.telescope, args.spec_res, args.t_int, args.s147, args.alpha_r))

# Read the file and extract data
z, xHI_mean, logfX, freq_axis, los_arr = get_los(file_path)
_, _, _, _, so_los_arr = get_los(so_file_path)

min_mse = 6e23
best_component = -1
best_reconstruction = None
best_reconstruction_type = None
# Perform PCA on los_arr (1000 sightlines, 2762 features each)
num_components = min(los_arr.shape[0], los_arr.shape[1])-1
pca = PCA(n_components=num_components)
transformed_data = pca.fit_transform(los_arr)
print(f"Shape of transformed data: {transformed_data.shape}")
for c in range(1):#num_components):

    if c%100 == 0: 
        print(f"num_components={num_components}")
        print(f"min_mse={min_mse}, best_component={best_component}")

    #print(f"Shape of transformed_data {transformed_data.shape}")
    component = transformed_data[:,c].reshape((1000,1))
    print(f"Shape of componenet: {component.shape}")
    #reconstructed_data = pca.inverse_transform(component)
    selected_components = [0, 1, 2]  # Example: using the first three components
    reconstructed_data = pca.inverse_transform(transformed_data[:, selected_components])

    print(f"Shape of reconstructed_data: {reconstructed_data.shape}")
    mse = mean_squared_error(so_los_arr, reconstructed_data)
    if mse < min_mse: 
        min_mse = mse
        best_component = c
        best_reconstruction = reconstructed_data
        best_reconstruction_type = "transformed"
    
    subtracted_data = los_arr - reconstructed_data
    mse = mean_squared_error(so_los_arr, subtracted_data)
    if mse < min_mse: 
        min_mse = mse
        best_component = c
        best_reconstruction = subtracted_data
        best_reconstruction_type = "subtracted"

# Compare original and reconstructed sightlines
sightline_index = 0  # Choose a sightline to compare
original_sightline = los_arr[sightline_index]
reconstructed_sightline = best_reconstruction[sightline_index]-0.01
signalonly_sightline = so_los_arr[sightline_index]-0.02
print(f"min_mse={min_mse}, best_num_components={best_component}, best_reconstruction_type={best_reconstruction_type}")
# Plot the original and reconstructed sightlines
plt.figure(figsize=(10, 6))
plt.plot(freq_axis, original_sightline, label="Original Sightline", color="blue", alpha=0.1)
plt.plot(freq_axis, reconstructed_sightline, label=f"Reconstructed (using {best_num_components} modes)", color="red", alpha=0.5)
plt.plot(freq_axis, signalonly_sightline, label=f"Signalonly", color="green", alpha=0.5)
plt.xlabel("Frequency (MHz)")
plt.ylabel("Signal (arbitrary units)")
plt.title(f"Comparison of Original and Best Reconstructed Sightlines\n(z={z}, xHI_mean={xHI_mean}, logfX={logfX})")
plt.legend()
plt.grid(True)
plt.show()

# # Plot explained variance
# explained_variance = pca.explained_variance_ratio_
# plt.figure(figsize=(8, 5))
# plt.bar(range(1, len(explained_variance) + 1), explained_variance * 100)
# plt.xlabel("Principal Component")
# plt.ylabel("Explained Variance (%)")
# plt.title("Explained Variance by Principal Components")
# plt.show()
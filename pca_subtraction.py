import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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

# File path
file_path = '../data/21cmFAST_los/F21_noisy/F21_noisy_21cmFAST_200Mpc_z6.0_fX-4.00_xHI0.94_uGMRT_8kHz_t500h_Smin64.2mJy_alphaR-0.44.dat'

# Read the file and extract data
z, xHI_mean, logfX, freq_axis, los_arr = get_los(file_path)

# Perform PCA on los_arr (1000 sightlines, 2762 features each)
num_components = min(los_arr.shape[0], los_arr.shape[1])  # Number of principal components to retain
pca = PCA(n_components=num_components)
transformed_data = pca.fit_transform(los_arr)
print(f"Shape of transformed_data {transformed_data.shape}")
reconstructed_data = pca.inverse_transform(transformed_data)

# Compare original and reconstructed sightlines
sightline_index = 0  # Choose a sightline to compare
original_sightline = los_arr[sightline_index]
reconstructed_sightline = reconstructed_data[sightline_index]

# Plot the original and reconstructed sightlines
plt.figure(figsize=(10, 6))
plt.plot(freq_axis, original_sightline, label="Original Sightline", color="blue")
plt.plot(freq_axis, reconstructed_sightline, label=f"Reconstructed (using {num_components} modes)", color="red")
plt.xlabel("Frequency (MHz)")
plt.ylabel("Signal (arbitrary units)")
plt.title(f"Comparison of Original and Reconstructed Sightlines\n(z={z}, xHI_mean={xHI_mean}, logfX={logfX})")
plt.legend()
plt.grid(True)
plt.show()


# Plot the original and reconstructed sightlines
plt.figure(figsize=(10, 6))
plt.plot(freq_axis, original_sightline-reconstructed_sightline+1.0, label="Residual", color="blue")
plt.xlabel("Frequency (MHz)")
plt.ylabel("Signal (arbitrary units)")
plt.ylim(0.85,1.04)
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
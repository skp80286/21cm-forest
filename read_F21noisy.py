'''
Read 21-cm forest transmitted flux with noise data.

Version 29.10.2024
'''

import sys
import numpy as np
import argparse
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

parser = argparse.ArgumentParser(description='Read 21cm forest data')
parser.add_argument('-p', '--path', type=str, default='../data/21cmFAST_los/F21_noisy/', help='filepath')
parser.add_argument('-z', '--redshift', type=float, default=6, help='redshift')
parser.add_argument('-d', '--dvH', type=float, default=0.0, help='rebinning width in km/s')
parser.add_argument('-r', '--spec_res', type=float, default='8', help='spectral resolution of telescope (i.e. frequency channel width) in kHz')
parser.add_argument('-t', '--telescope', type=str, default='uGMRT', help='telescope')
parser.add_argument('-s', '--s147', type=float, default=64.2, help='intrinsic flux of QSO at 147Hz in mJy')
parser.add_argument('-a', '--alpha_r', type=float, default=-0.44, help='radio spectral index of QSO')
parser.add_argument('-i', '--t_int', type=float, default=500, help='integration time of obsevation in hours')
parser.add_argument('-f', '--log_fx', type=float, default=-0.6, help='log10(f_X)')
parser.add_argument('-x', '--xHI', type=float, default=0.24, help='mean neutral hydrogen fraction')
parser.add_argument('-n', '--nlos', type=int, default=1000, help='num lines of sight')

args = parser.parse_args()


#Input parameters
#Read LOS data from 21cmFAST 50cMpc box
datafile = str('%sF21_noisy_21cmFAST_200Mpc_z%.1f_fX%.2f_xHI%.2f_%s_%dkHz_t%dh_Smin%.1fmJy_alphaR%.2f.dat' % 
               (args.path, args.redshift,args.log_fx, args.xHI, args.telescope, args.spec_res, args.t_int, args.s147, args.alpha_r))
print (f"Reading file: {datafile}")
data  = np.fromfile(str(datafile),dtype=np.float32)
z        = data[0]		#redshift
xHI_mean = data[1]		#mean neutral hydrogen fraction
logfX    = data[2]		#log10(f_X)
Nlos     = int(data[3])	#Number of lines-of-sight
Nbins    = int(data[4])	#Number of pixels/cells/bins in one line-of-sight
x_initial = 5

print(data.shape)
print(data[:x_initial])
skipcount = 0
for d in data[x_initial:]:
    if d > 1e7:
        skipcount += 1
        continue
    else:
        break

print(f"Found {skipcount} fields > 1e7 after x_initial")

freq_axis   = data[(x_initial+0*Nbins):(x_initial+1*Nbins)]	                                    #frequency along LoS in Hz
#F21         = np.reshape(data[(x_initial+8*Nbins):(x_initial+8*Nbins+Nlos*Nbins)],(Nlos,Nbins))	#transmitted flux including noise
F21         = np.reshape(data[(x_initial+skipcount):(x_initial+skipcount+Nlos*Nbins)],(Nlos,Nbins))	#transmitted flux including noise

print('z=%.2f, <x_HI>=%.6f, log10(f_X)=%.2f, %d LOS, %d pixels' % (z,xHI_mean,logfX,Nlos,Nbins))
print('Frequency range: %.2f-%.2fMHz' % (freq_axis[0]/1e6,freq_axis[-1]/1e6))
print(F21.shape)
print(freq_axis)
print(F21[9:14])

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [15, 9]
skipcount = 0
for f in F21[:1]:
    if f[0] > 1e7:
        skipcount += 1
        print(f)
        continue
    plt.plot(freq_axis/1e6, f)
plt.xlabel('frequency[MHz]'), plt.ylabel('flux/S147')
plt.show()

print (f"Skipped {skipcount} lines")


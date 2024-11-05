'''
Read 21-cm forest 1D power spectrum data.

Version 21.10.2024
'''

import sys
import numpy as np

#Input parameters
path = '../data/21cmFAST_los/F21_noisy'
z_name = float(sys.argv[1])		#redshift
dvH = float(sys.argv[2])		#rebinning width in km/s
fX_name = float(sys.argv[3])	#log10(f_X)
xHI_mean = float(sys.argv[4])	#mean neutral hydrogen fraction


#Read LOS data from 21cmFAST 50cMpc box
#datafile = str('%slos/los_50Mpc_256_n1000_z%.3f_fX%.1f_xHI%.2f.dat' % (path,z_name,fX_name,xHI_mean))
datafile = '../data/21cmFAST_los/F21_noisy/F21_noisy_21cmFAST_200Mpc_z6.0_fX-0.20_xHI0.00_uGMRT_8kHz_t50h_Smin64.2mJy_alphaR-0.44.dat'
data  = np.fromfile(str(datafile),dtype=np.float32)
print(data.shape)
print(data[:50])
z        = data[0]		#redshift
omega_0  = data[1]		#Omega_matter
omega_L  = data[2]		#Omega_lambda
omega_b  = data[3]		#Omega_baryon
h        = data[4]		#Hubble constant
Lbox     = data[5]		#box size in cMpc
X_H      = data[6]		#hydrogen fraction
Nbins    = int(data[7])	#Number of pixels/cells/bins in one line-of-sight
Nlos     = int(data[8])	#Number of lines-of-sight
logfX    = data[9]		#log10(f_X)
ioneff   = data[10]		#ionization efficiency
xHI_mean = data[11]		#mean neutral hydrogen fraction

x_initial = 12

pos_axis = data[(x_initial+0*Nbins):(x_initial+1*Nbins)]	#position along LoS in ckpc
vel_axis = data[(x_initial+1*Nbins):(x_initial+2*Nbins)]	#Hubble velocity along LoS in km/s

Delta   = np.reshape(data[(x_initial+2*Nbins+0*Nlos*Nbins):(x_initial+2*Nbins+1*Nlos*Nbins)],(Nlos,Nbins))	#density contrast
xHI     = np.reshape(data[(x_initial+2*Nbins+1*Nlos*Nbins):(x_initial+2*Nbins+2*Nlos*Nbins)],(Nlos,Nbins))	#neutral hydrogen fraction
TK      = np.reshape(data[(x_initial+2*Nbins+2*Nlos*Nbins):(x_initial+2*Nbins+3*Nlos*Nbins)],(Nlos,Nbins))	#kinetic temperature of the gas in K
vpec	= np.reshape(data[(x_initial+2*Nbins+3*Nlos*Nbins):(x_initial+2*Nbins+4*Nlos*Nbins)],(Nlos,Nbins))	#peculiar velocity of the gas in km/s
Gamma_HI= np.reshape(data[(x_initial+2*Nbins+4*Nlos*Nbins):(x_initial+2*Nbins+5*Nlos*Nbins)],(Nlos,Nbins))	#photoionization rate of HI in s^-1
J21     = np.reshape(data[(x_initial+2*Nbins+5*Nlos*Nbins):(x_initial+2*Nbins+6*Nlos*Nbins)],(Nlos,Nbins))	#Ionization radiation background in s^-1



#Read LOS data from 21cmFAST 50cMpc box that have been rebinned for the tau_21 calculation to be convolved properly
datafile = str('%slos_regrid/los_50Mpc_n200_z%.3f_fX%.1f_xHI%.2f_dv%d_file%d.dat' % (path,z_name,fX_name,xHI_mean,dvH,0))
data  = np.fromfile(str(datafile),dtype=np.float32)
z        = data[0]		#redshift
omega_0  = data[1]		#Omega_matter
omega_L  = data[2]		#Omega_lambda
omega_b  = data[3]		#Omega_baryon
h        = data[4]		#Hubble constant
Lbox     = data[5]		#box size in cMpc
X_H      = data[6]		#hydrogen fraction
Nbins    = int(data[7])	#Number of pixels/cells/bins in one line-of-sight
Nlos     = int(data[8])	#Number of lines-of-sight
logfX    = data[9]		#log10(f_X)
ioneff   = data[10]		#ionization efficiency
xHI_mean = data[11]		#mean neutral hydrogen fraction

x_initial = 12

pos_axis = data[(x_initial+0*Nbins):(x_initial+1*Nbins)]	#position along LoS in ckpc
vel_axis = data[(x_initial+1*Nbins):(x_initial+2*Nbins)]	#Hubble velocity along LoS in km/s

Delta   = np.reshape(data[(x_initial+2*Nbins+0*Nlos*Nbins):(x_initial+2*Nbins+1*Nlos*Nbins)],(Nlos,Nbins))	#density contrast
xHI     = np.reshape(data[(x_initial+2*Nbins+1*Nlos*Nbins):(x_initial+2*Nbins+2*Nlos*Nbins)],(Nlos,Nbins))	#neutral hydrogen fraction
TK      = np.reshape(data[(x_initial+2*Nbins+2*Nlos*Nbins):(x_initial+2*Nbins+3*Nlos*Nbins)],(Nlos,Nbins))	#kinetic temperature of the gas in K
vpec	= np.reshape(data[(x_initial+2*Nbins+3*Nlos*Nbins):(x_initial+2*Nbins+4*Nlos*Nbins)],(Nlos,Nbins))	#peculiar velocity of the gas in km/s
Gamma_HI= np.reshape(data[(x_initial+2*Nbins+4*Nlos*Nbins):(x_initial+2*Nbins+5*Nlos*Nbins)],(Nlos,Nbins))	#photoionization rate of HI in s^-1



#Load data for tau_21 (optical depth) from 50cMpc box
datafile = str('%stau/tau_50Mpc_n%d_z%.3f_fX%.1f_xHI%.2f_dv%d_file%d.dat' %(path,200,z_name,fX_name,xHI_mean,dvH,0))
data = np.fromfile(str(datafile),dtype=np.float32)
tau = np.reshape(data,(Nlos,Nbins))



#Load data for x-axis from 200cMpc box
datafile = str('%stau_long/los_200Mpc_n%d_z%.3f_dv%d.dat' %(path,1000,z_name,dvH))
data  = np.fromfile(str(datafile),dtype=np.float32)
z        = data[0]		#redshift
box_size = data[1]/1000	#box size in cMpc
Nbins    = int(data[2])	#Number of pixels/cells/bins in one line-of-sight
Nlos     = int(data[3])	#Number of lines-of-sight
x_initial = 4
pos_axis = data[(x_initial+0*Nbins):(x_initial+1*Nbins)]	#position along LoS in ckpc
vel_axis = data[(x_initial+1*Nbins):(x_initial+2*Nbins)]	#Hubble velocity along LoS in km/s


#Load data for tau_21 (optical depth) from 200cMpc box
datafile = str('%stau_long/tau_200Mpc_n%d_z%.3f_fX%.1f_xHI%.2f_dv%d.dat' %(path,Nlos,z_name,fX_name,xHI_mean,dvH))
data = np.fromfile(str(datafile),dtype=np.float32)
tau = np.reshape(data,(Nlos,Nbins))

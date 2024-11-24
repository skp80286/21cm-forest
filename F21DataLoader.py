from concurrent.futures import ThreadPoolExecutor
import threading
from typing import List, Dict
import numpy as np
import PS1D
from scipy.stats import binned_statistic

class F21DataLoader:
    def __init__(self, max_workers: int = 4, psbatchsize: int = 1000, limitsamplesize: int = None, skip_ps: bool = False, ps_bins = None):
        self.max_workers = max_workers
        self.collector = ThreadSafeArrayCollector()
        self.psbatchsize = psbatchsize
        self.limitsamplesize = limitsamplesize
        self.skip_ps = skip_ps
        self.ps_bins = ps_bins

    def get_los(self, datafile: str) -> None:
        data = np.fromfile(str(datafile), dtype=np.float32)
    
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
        Nlos = int(data[i])# Number of lines-of-sight
        i += 1
        Nbins = int(data[i])# Number of pixels/cells/bins in one line-of-sight
        x_initial = i + 1

        if len(data) != x_initial + (1+Nlos)*Nbins:
            error_msg = f"Error: Found {len(data)} fields, expected {x_initial + (1+Nlos)*Nbins:}. x_initial={x_initial}, Nlos={Nlos}, Nbins={Nbins}. File may be corrupted: {datafile}"
            raise ValueError(error_msg)
        # Find skipcount
        skipcount = 0
        for d in data[x_initial:]:
            if d > 1e7:
                skipcount += 1
                continue
            else:
                break

        if skipcount != Nbins:
            error_msg = f"Error: Found {skipcount} fields > 1e7 after x_initial, expected {Nbins}. File may be corrupted: {datafile}"
            raise ValueError(error_msg)

        # Extract frequency axis and F21 data
        freq_axis = data[(x_initial+0*Nbins):(x_initial+1*Nbins)]
        """
        if freq_axis is not None and not np.array_equal(freq_axis, freq_axis_cur):
            error_msg = f"Error: Frequency axis mismatch in file: {datafile}\n"
            error_msg += f"Expected: {freq_axis[0]}-{freq_axis[-1]} MHz\n"
            error_msg += f"Found: {freq_axis_cur[0]}-{freq_axis_cur[-1]} MHz"
            raise ValueError(error_msg)
        """
        los_arr = np.reshape(data[(x_initial+1*Nbins):(x_initial+1*Nbins+Nlos*Nbins)],(Nlos,Nbins))
        """
        if Nlos > 100:
            Nlos = 100
        """
        return (z, xHI_mean, logfX, freq_axis, los_arr)


    def aggregate(self, dataseries):
        # Calculate mean and standard deviation across all samples
        mean = np.mean(dataseries, axis=0)
        std = np.std(dataseries, axis=0)
        return (mean, std)

    def process_file(self, datafile: str) -> None:
        try:
            print(f"Reading file: {datafile}")
            (z, xHI_mean, logfX, freq_axis, los_arr) = self.get_los(datafile)
            # Store the data
            #all_F21.append(F21_current)
            bandwidth = freq_axis[-1]-freq_axis[0]
            power_spectrum = []
            cumulative_los = []
            ks = None
            psbatchnum = 0
            samplenum = 0
            if self.limitsamplesize is not None and len(los_arr) > self.limitsamplesize:
                los_arr = los_arr[np.random.randint(len(los_arr), size=self.limitsamplesize)]
            Nlos = len(los_arr)
 
 
            print('z=%.2f, <x_HI>=%.6f, log10(f_X)=%.2f, %d LOS, %d pixels' % 
                (z, xHI_mean, logfX, Nlos, len(los_arr[0])))
            
            for los in los_arr:
                if self.skip_ps:
                    params = np.array([xHI_mean, logfX])
                    self.collector.add_data(None, None, None, los, None, freq_axis, params)
                else:
                    psbatchnum += 1
                    samplenum += 1
                    # Calculate the power spectrum
                    #ks, power_spectrum = power_spectrum_1d(los, bins=160)
                    ks, ps = PS1D.get_P(los, bandwidth)
                    if self.ps_bins is not None:
                        ps, bin_edges, _ = binned_statistic(np.abs(ks), ps, statistic='mean', bins=self.ps_bins)
                        ks = 0.5 * (bin_edges[1:] + bin_edges[:-1])  # Bin centers

                    #print(f"ks: {ks}")
                    #print(f"ps: {ps}")
                    power_spectrum.append(ps)
                    cumulative_los.append(los)
                    
                    if samplenum > Nlos or psbatchnum >= self.psbatchsize:
                        # Collect this batch
                        params = np.array([xHI_mean, logfX])
                        (ps_mean, ps_std) = self.aggregate(np.array(power_spectrum))
                        (los_mean, los_std) = self.aggregate(np.array(cumulative_los))
                        self.collector.add_data(ks, ps_mean, ps_std, los_mean, los_std, freq_axis, params)
                        psbatchnum = 0
                        power_spectrum = []
                        cumulative_los = []

        except Exception as e:
            print(f"Error processing {datafile}: {str(e)}")
            
    def process_all_files(self, file_list: List[str]) -> Dict[str, np.ndarray]:
        # Process files in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all files for processing
            futures = [executor.submit(self.process_file, filepath) 
                      for filepath in file_list]
            
            # Wait for all tasks to complete
            for future in futures:
                future.result()  # This will raise any exceptions that occurred
                
        # Return the collected results
        return self.collector.get_arrays()

class ThreadSafeArrayCollector:
    def __init__(self):
        self._data = {
            'ks': [],
            'ps': [],
            'ps_std': [],
            'los': [],
            'los_std': [],
            'freq_axis': [],
            'params': []
        }
        self._lock = threading.Lock()
        
    def add_data(self, ks, ps, ps_std, los, los_std, freq_axis, params):
        with self._lock:
            self._data['ks'].append(ks)
            self._data['ps'].append(ps)
            self._data['ps_std'].append(ps_std)
            self._data['los'].append(los)
            self._data['los_std'].append(los_std)
            self._data['freq_axis'].append(freq_axis)
            self._data['params'].append(params)
            if len(los) % 10 == 0: print(f"DataLoader: {len(los)} records added.")
            
    def get_arrays(self):
        with self._lock:
            return {
                'ks': np.array(self._data['ks']),
                'ps': np.array(self._data['ps']),
                'ps_std': np.array(self._data['ps_std']),
                'los': np.array(self._data['los']),
                'los_std': np.array(self._data['los_std']),
                'freq_axis': np.array(self._data['freq_axis']),
                'params': np.array(self._data['params'])
            }
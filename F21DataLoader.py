from concurrent.futures import ThreadPoolExecutor
import threading
from typing import List, Dict
import numpy as np
import PS1D

class F21DataLoader:
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.collector = ThreadSafeArrayCollector()
        
    def process_file(self, datafile: str) -> None:
        try:
            print(f"Reading file: {datafile}")
            data = np.fromfile(str(datafile), dtype=np.float32)
        
            # Extract parameters
            z = data[0]        # redshift
            xHI_mean = data[1] # mean neutral hydrogen fraction
            logfX = data[2]    # log10(f_X)
            Nlos = int(data[3])# Number of lines-of-sight
            Nbins = int(data[4])# Number of pixels/cells/bins in one line-of-sight
            x_initial = 5

            if len(data) != x_initial + (1+Nlos)*Nbins:
                error_msg = f"Error: Found {len(data)} fields, expected {x_initial + (1+Nlos)*Nbins:}. File may be corrupted: {datafile}"
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
            F21_current = np.reshape(data[(x_initial+1*Nbins):(x_initial+1*Nbins+Nlos*Nbins)],(Nlos,Nbins))

            print('z=%.2f, <x_HI>=%.6f, log10(f_X)=%.2f, %d LOS, %d pixels' % 
                (z, xHI_mean, logfX, Nlos, Nbins))
            # Store the data
            #all_F21.append(F21_current)
            bandwidth = freq_axis[-1]-freq_axis[0]
            power_spectrum = None
            ks = None
            for los in F21_current:
                # Calculate the power spectrum
                #ks, power_spectrum = power_spectrum_1d(los, bins=160)
                ks, ps = PS1D.get_P(los, bandwidth)
                if power_spectrum is None:
                    power_spectrum = ps
                else:
                    power_spectrum += ps

            params = np.array([xHI_mean, logfX])
            # Store results thread-safely
            self.collector.add_data(ks, power_spectrum/Nlos, params)
                
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
            'F21': [],
            'params': []
        }
        self._lock = threading.Lock()
        
    def add_data(self, ks, F21, params):
        with self._lock:
            self._data['ks'].append(ks)
            self._data['F21'].append(F21)
            self._data['params'].append(params)
            
    def get_arrays(self):
        with self._lock:
            return {
                'ks': np.array(self._data['ks']),
                'F21': np.array(self._data['F21']),
                'params': np.array(self._data['params'])
            }
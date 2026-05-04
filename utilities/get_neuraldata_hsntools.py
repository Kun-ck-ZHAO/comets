import numpy as np
import pandas as pd
import os
import sys

# NEW (hsntools)
# from hsntools.io import open_h5file, get_files, load_config, save_json, load_json, make_session_name
# from hsntools.paths import Paths
# from hsntools.utils.log import print_status
import hsntools
from hsntools.io.utils import get_files

# sys.path.append('../scripts')
# from settings import RUN, PATHS, UNITS,METHODS, SURROGATES


def get_spike_indx(file_path,verbose = True):
    res_files = get_files(file_path, select='.res')
    clu_files = get_files(file_path, select='.clu')
    num_elec = len(res_files)
    session_data = {}  # Dictionary to store data for all electrodes
    total_unit = 0 
    for elec in range(num_elec):
        res_path = os.path.join(file_path, res_files[elec])
        clu_path = os.path.join(file_path, clu_files[elec])
        
        # Load the spike indices and cluster files
        res_df = pd.read_csv(res_path, header=None, names=['spike_ind'])
        clu_df = pd.read_csv(clu_path, header=None, names=['clusters'])
        
        # Get the number of clusters for current electrode
        num_clu = clu_df['clusters'].iloc[0]
        total_unit = total_unit+num_clu
#         print_status(RUN['VERBOSE'], 'running electrode: \t\t\tE{:02d}'.format(elec), 1)
#         print_status(RUN['VERBOSE'], 'Number of clusters: \t\t\tE{:02d}'.format(num_clun), 1)
        if verbose:
            print(f'Elec {elec}: {num_clu} Clusters')
        
        # Drop the first row from clu_df and reset its index
        clu_df = clu_df.drop(clu_df.index[0]).reset_index(drop=True)
        
        # Combine the spike indices and cluster dataframes
        combined_df = pd.concat([res_df, clu_df], axis=1)
        
        # Group by clusters to separate spikes by cluster
        grouped = combined_df.groupby('clusters')
        
        # Prepare a dictionary for this electrode's spike indices, grouped by cluster
        elec_data = {}
        for cluster, data in grouped:
            elec_data[cluster] = data['spike_ind'].values
        
        # Store this electrode's data in the all_data dictionary
        session_data[f'{elec}'] = elec_data

    return session_data,total_unit


def get_wave_data(file_path):
    wave_files = get_files(file_path, select='.spk')
    clu_files = get_files(file_path, select='.clu')
    num_elec = len(wave_files)
    session_wave = {}
    
    for elec in range(num_elec):
        wave_path = os.path.join(file_path, wave_files[elec])
        clu_path = os.path.join(file_path, clu_files[elec])
        
        clu_df = pd.read_csv(clu_path, header=None, names=['clusters'])
        wave_data = np.fromfile(wave_path, dtype=np.int16)
        
        num_clu = clu_df['clusters'].iloc[0]
        clu_df = clu_df.drop(clu_df.index[0]).reset_index(drop=True)
        
        if len(clu_df) != int(len(wave_data)/256):
            session_wave[f'{elec}'] = {}
        else:
            all_waveforms = wave_data.reshape(len(clu_df), 32, 8)

            wave_df = pd.DataFrame({'waves': list(all_waveforms)})
            combined_df = pd.concat([wave_df, clu_df], axis=1)

            grouped = combined_df.groupby('clusters')

            wave_unit = {}
            for cluster, data in grouped:
                wave_unit[cluster] = data['waves'].values
                session_wave[f'{elec}'] = wave_unit
    return session_wave
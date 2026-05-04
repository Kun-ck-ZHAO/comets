"""Helper functions for creating reports."""

import numpy as np

from spiketools.measures.spikes import compute_firing_rate, compute_presence_ratio

from spiketools.utils.base import count_elements
from spiketools.utils.extract import drop_range
from spiketools.utils.timestamps import convert_sec_to_min

def create_group_str(group_info):
    """Create a string representation of the group info."""
    
    # Start with introductory information if it exists in group_info
    string = ''

    # Add each analysis's information
    for analysis, info in group_info.items():
        # Only include if it's an analysis entry (in dictionary format)
        if isinstance(info, dict):
            string += ' {} cells:    count = {:3d},  percentage = {:5.2f}%,  stat_value = {:.4f}\n'.format(
                analysis, info['count'], info['percentage'], info['stat_value']
            )

    return string


def create_sess_str(summary):
    """Create a string representation of the group info."""

    string = '\n'.join([
        '\n',
        'subject:  {}'.format(summary['subj']),
        'day:  {}'.format(summary['day']),
        'sessions:  {}'.format(summary['session']),
        'num_elec:  {}'.format(summary['num_elec']),
        'total clusters: {}'.format(summary['total_cluster'])
    ])

    return string



def create_pos_str(summary):
    """Create a string representation of the group info."""

    string = '\n'.join([
        '\n',
        'Session Length (s):  {}'.format(summary['Session Length']),
        'Track Length (cm):  {}'.format(summary['Track Length']),
#         'AP x_corr {}'.format(summary['AP x_corr']),
#         'AP y_corr {}'.format(summary['AP y_corr']),
        'Forward trials: {}'.format(summary['f_num_trials']),
        'Backward trials: {}'.format(summary['b_num_trials']),
        'Total trials: {}'.format(summary['trial_num'])
        
    ])

    return string


def create_elec_str(summary):
    """Create a string representation of the group info."""

    string = '\n'.join([
        '\n',
        'subjects:  {}'.format(summary['subj']),
        'day:  {}'.format(summary['day']),
        #'sessions:  {}'.format(summary['session']),
        'num_elec:  {}'.format(summary['num_elec']),
        'total clusters:  {}'.format(summary['total_cluster']),
        #'total units: {}'.format(summary['total units'])
    ])

    return string

# def create_unit_str(summary):
#     """Create a string representation of the group info."""

#     string = '\n'.join([
#         '\n',
#         'subject:  {}'.format(summary['subj']),
#         'day:  {}'.format(summary['day']),
#         'session:  {}'.format(summary['session']),
#         'E0{}'.format(summary['elec']),
#         'U0{}'.format(summary['unit']),
#         'region:  {}'.format(summary['region']),
#         'neuron_type:  {}'.format(summary['neuron_type']),
#         'familiarity level:  {}'.format(summary['fam'])
#     ])

#     return string

## UNIT

def create_unit_info(unit,spikes):
    """Create a dictionary of unit information."""

    #spikes = unit['spike_times'].values[0]

    unit_info = {}

    unit_info['n_spikes'] = len(spikes)
    unit_info['firing_rate'] = compute_firing_rate(spikes)
    unit_info['first_spike'] = spikes[0]
    unit_info['last_spike'] = spikes[-1]
    unit_info['region'] = unit['region']
    unit_info['fam'] = unit['fam']
    unit_info['neuron_type'] = unit['neuron_type']

    return unit_info


def create_unit_str(unit_info):
    """Create a string representation of the unit info."""

    string = '\n'.join([
        '# spikes:    {:10d}'.format(unit_info['n_spikes']),
        'firing rate:  {:10.2f}'.format(unit_info['firing_rate']),
        'location:    {}'.format(unit_info['region']),
        'neuron_type:    {}'.format(unit_info['neuron_type']),
        'familarity level:    {}'.format(unit_info['fam']),

        'Recording time range: {:5.4f} - {:5.4f}'.format(\
           unit_info['first_spike'], unit_info['last_spike'])
    ])

    return string

def create_f_place_field_str(results):
    """Create a string representation of the group info."""

    string = '\n'.join([
        '\n',
        'Number of Place field:  {}'.format(results['f_num_place_fields']),
        'Number of Spikes:  {}'.format(results['f_num_spikes']),
        'Percentage of trials peak place bins close to average: {}'.format(np.round(results['f_percentage_trials_close_to_peak'],2)),
        'Place Field Bins: {}'.format(results['f_place_field_bins']),
        'Noise bins: {}'.format(results['f_noise_bins']),
        'Presence Ratio: {}'.format(results['f_presence_ratio'])
        
    ])

    return string


def create_b_place_field_str(results):
    """Create a string representation of the group info."""

    string = '\n'.join([
        '\n',
        'Number of Place field:  {}'.format(results['b_num_place_fields']),
        'Number of Spikes:  {}'.format(results['b_num_spikes']),
        'Percentage of trials peak place bins close to average : {}'.format(np.round(results['b_percentage_trials_close_to_peak'],2)),
        'Place Field Bins: {}'.format(results['b_place_field_bins']),
        'Noise bins: {}'.format(results['b_noise_bins']),
        'Presence Ratio: {}'.format(results['b_presence_ratio'])
        
    ])

    return string


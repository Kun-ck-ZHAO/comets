"""Local utility functions."""

###################################################################################################
###################################################################################################
import numpy as np
from scipy.stats import spearmanr
from scipy.ndimage import label,gaussian_filter


from convnwb.io import load_nwbfile
from spiketools.utils.epoch import epoch_data_by_range, epoch_spikes_by_range
#from spiketools.utils.base import count_elements
from spiketools.utils.trials import recombine_trial_data
from spiketools.utils.extract import get_range, get_values_by_time_range, get_values_by_times,get_inds_by_times
from spiketools.utils.checks import check_axis


# def group_array_by_key(keys, values):
#     result = {}
#     for key, value in zip(keys, values):
#         if key not in result:
#             result[key] = []
#         result[key].append(value)
#     return result
def get_values_by_times(timestamps, values, timepoints, time_threshold=None,
                        drop_null=True, axis=None):
    axis = check_axis(axis, values)
    inds = get_inds_by_times(timestamps, timepoints, time_threshold, drop_null)

    if drop_null:
        outputs = values.take(indices=inds, axis=check_axis(axis, values))
    else:
        outputs = np.full([np.atleast_2d(values).shape[0], len(timepoints)], np.nan)
        mask = inds >= 0
        outputs[:, np.where(mask)[0]] = values.take(indices=inds[mask], axis=axis)
        outputs = np.squeeze(outputs, axis=0 if outputs.shape == (1, 1) else None)

    return outputs,inds


def select_movement(data, move_starts, move_stops, recombine=True):
    """Helper function to select data from during navigation periods."""

    times_trials, values_trials = epoch_data_by_range(\
        data.timestamps[:], data.data[:], move_starts, move_stops)

    if not recombine:
        return times_trials, values_trials
    else:
        times, values = recombine_trial_data(times_trials, values_trials)
        return times, values
    

def find_place_field(rate_map, place_field_thresh=0.5, noise_thresh=0.5):
    """
    Identify contiguous regions of high firing rates above a threshold, and remove noisy regions below a threshold.
    
    Parameters:
    - rate_map: 2D array representing the spatial firing rate map.
    - place_field_thresh: Fraction of the peak firing rate used for thresholding (e.g., 0.2).
    - noise_thresh: Threshold below which regions are considered noise and excluded.
    
    Returns:
    - place_field_mask: Binary map of the identified place fields (True for regions above the threshold).
    - place_bins_passed_thresh: Number of bins above the place field threshold.
    - labeled_fields: 2D array where each contiguous region above the threshold is labeled with a unique integer.
    - num_fields: Number of contiguous place fields found.
    - peak_rate: Peak firing rate used for thresholding.
    """
    # Find the peak firing rate in the rate map
    peak_rate = np.max(rate_map)
    peak_loc = np.argmax(rate_map)
    
    # Define the threshold value (fraction of peak firing rate)
    place_field_thresh_value = place_field_thresh * peak_rate
    noise_thresh_value = noise_thresh*peak_rate
    
    # Create a binary mask of regions with firing rates above the threshold
    place_field_mask = rate_map >= place_field_thresh_value
    
    # Apply the noise threshold: Remove regions where firing rate is below the noise threshold
    noise_mask = rate_map < noise_thresh_value
    
    
    # Count how many bins (or regions) passed the combined threshold
    num_place_field_bins = place_field_mask.sum()
    num_noise_bins = noise_mask.sum()
    
    # Label contiguous regions above the threshold
    labeled_place_fields, num_place_fields = label(place_field_mask)
    
    # Return all the relevant information
    return num_place_field_bins,num_noise_bins, labeled_place_fields, num_place_fields, peak_rate,peak_loc


def find_place_location(trial_place_bins, peak_loc, tolerance=1):
    max_locs = []  # List to store the indices of max values
    max_values = []  # List to store the max values
          
    for indx, trial in enumerate(trial_place_bins):
        # Find the index of the maximum value in the trial
        max_index = np.argmax(trial)
        max_value = np.max(trial)
        
        # Store the results
        max_locs.append(max_index)
        max_values.append(max_value)

    max_locs = np.array(max_locs)
    max_values = np.array(max_values)

    # Compute the number of max_locs close to the peak_loc within a given tolerance
    close_to_peak = np.abs(max_locs - peak_loc) <= tolerance
    num_close_to_peak = np.sum(close_to_peak)
    percentage_num_close_to_peak = num_close_to_peak/len(trial_place_bins)
    
    return max_locs, max_values, num_close_to_peak,percentage_num_close_to_peak

def get_significant_percentage(stats_vals,p_vals,increment,threshold = 0.05):

    ## Calculate the number of steps
    max_value = int(np.max(stats_vals))+1
    n_step = int(max_value/increment)
    categories = []
    significant_percentages = []
    for i in range(n_step):
        lower_bound = i*increment
        upper_bound = (i+1)*increment

        # Find indices and compute percentage of significant p values
        indx_in_category = np.where((stats_vals >=lower_bound)& (stats_vals < upper_bound))[0]
        if len(indx_in_category) > 0:
            p_vals_in_category = p_vals[indx_in_category]
            significant_p_vals = p_vals_in_category[p_vals_in_category<threshold]

            n_significant_p_vals = len(significant_p_vals)
            significant_percentage = (n_significant_p_vals / len(p_vals_in_category)) * 100
            category = upper_bound
        else:
            significant_percentage = None
            category = None
        
        significant_percentages.append(significant_percentage)
        categories.append(category)
    return categories,significant_percentages




def get_agreement_percentage(stats, p_values1, p_values2, increment=0.5,threshold = 0.05):
    """
    Calculate the percentage of agreement between two sets of p-values.
    """
    pass
    max_value = int(np.max(stats)) + 1
    num_steps = int(max_value / increment)
    categories = []
    agreement_percentages = []
    significant_percentages = []
    not_significant_percentages  = []

    for i in range(num_steps):  # Loop over increments
        lower_bound = i * increment
        upper_bound = (i + 1) * increment
        categories.append(upper_bound)
        indices_in_category = np.where((stats >= lower_bound) & (stats < upper_bound))[0]

        if len(indices_in_category) > 0:
            p_values1_in_category = p_values1[indices_in_category]
            p_values2_in_category = p_values2[indices_in_category]

            significant_indices = (p_values1_in_category < threshold) & (p_values2_in_category < threshold)
            not_significant_indices = (p_values1_in_category > threshold) & (p_values2_in_category > threshold)

            n_agree = np.sum(significant_indices)+np.sum(not_significant_indices)  # Count significant values

            # Calculate percentage
            agreement_percentages.append(n_agree / len(indices_in_category) * 100)
            significant_percentages.append(np.sum(significant_indices) / len(indices_in_category) * 100)
            not_significant_percentages.append(np.sum(not_significant_indices) / len(indices_in_category) * 100)
        else:
            agreement_percentages.append(None)
            significant_percentages.append(None)
            not_significant_percentages.append(None)
    return categories, agreement_percentages, significant_percentages, not_significant_percentages
  




import numpy as np
from collections import Counter

def find_local_extrema(data,mini = 50,maxi = 250,dist=100):
    """
    Find indices of local minima and maxima in the given 1D numpy array.
    
    :param data: A 1D numpy array of the time series data.
    :return: A tuple of two lists, first containing indices of local minima and second containing indices of local maxima.
    """
    # Initialize lists to hold the indices of local minima and maxima
    minima = []
    maxima = []

    # Iterate over the data, skipping the first and last points
    for i in range(1, len(data) - 1):
        # Local minimum if the current value is less than both neighbors
        if data[i - 1] > data[i] < data[i + 1]:
             if data[i] <mini:
                minima.append(i)
        # Local maximum if the current value is greater than both neighbors
        elif data[i - 1] < data[i] > data[i + 1]:
             if data[i] >maxi:
                maxima.append(i)
                

    return minima, maxima


def find_median_clusters(minima,maxima,dist=150):
    # Calculate the differences between consecutive elements
    min_diff = np.diff(minima)
    max_diff = np.diff(maxima)

    # Find where the differences are greater than 150 (indicating a new cluster)
    min_clu_indx = np.where(min_diff>dist)[0] + 1
    max_clu_indx = np.where(max_diff>dist)[0] + 1

    # Split the array into clusters
    min_clus = np.split(minima, min_clu_indx)
    max_clus = np.split(maxima, max_clu_indx)

    # Get the median points of every cluster
    
    
    trial_start = []
    trial_end = []

    # Get the median points of every cluster
    for clu in min_clus:
        if len(clu) > 0:  # Ensure the cluster is not empty
            min_med= np.median(clu)
            trial_start.append(int(min_med))
            
    for clu in max_clus:
        if len(clu) > 0:  # Ensure the cluster is not empty
            max_med= np.median(clu)
            trial_end.append(int(max_med))

    return trial_start,trial_end

def get_forward_trial(mini,maxi,num_trial):
    #num_trial = len(mini)
    f_trial_start = []
    f_trial_end = []

    for i in range(num_trial):
        if maxi[i] > mini[i]:

            f_trial_start.append(mini[i])
            f_trial_end.append(maxi[i])
        else:
            
            for k in range(i+1, len(maxi)):
                if maxi[k] > mini[i]:

                    f_trial_start.append(mini[i])
                    f_trial_end.append(maxi[k])
                    break  

    f_trial_start = np.array(f_trial_start)
    f_trial_end = np.array(f_trial_end)
    return f_trial_start,f_trial_end




def get_backward_trial(mini, maxi):
    b_trial_start = []  # This will hold the start of backward trials (peaks)
    b_trial_end = []    # This will hold the end of backward trials (troughs)

    # Loop through the trial_end array to find backward trials
    for i in range(len(maxi)):
        # Find the first trough after the peak
        for j in range(len(mini)):
            if mini[j] > maxi[i]:
                b_trial_start.append(maxi[i])
                b_trial_end.append(mini[j])
                break  # Stop the loop once the first trough is found

    b_trial_start = np.array(b_trial_start)
    b_trial_end = np.array(b_trial_end)

    return b_trial_start, b_trial_end

def drop_repeated_elements(data):
    counter = Counter(data)
    rep_ele = [elem for elem, count in counter.items() if count > 1]
    print(rep_ele)
    last_occurrence_indices = {}

    for index, element in enumerate(data):
        if element in rep_ele:
            last_occurrence_indices[element] = index
            
    new_data = []
    dropped_ind= []

    for idx, elem in enumerate(data):
        if counter[elem] == 1 or idx == last_occurrence_indices[elem]:
            new_data.append(elem) 
        else:
            dropped_ind.append(idx)
    return new_data,dropped_ind


def drop_corresponding_indx(data,dropped_ind):
    new_data= [elem for idx, elem in enumerate(data) if idx not in dropped_ind]
    return new_data 





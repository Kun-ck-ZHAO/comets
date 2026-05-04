import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr

def read_position_data(file_path):
    data_list = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()  # Remove leading/trailing whitespace
            values = line.split('\t')  # Split the line using tabs

            if len(values) == 4:
                value0 = float(values[0])
                value1 = float(values[1])
                value2 = float(values[2])
                value3 = float(values[3])
                data_list.append({'a_x': value0, 'a_y': value1, 'p_x': value2, 'p_y': value3})

    pos_df = pd.DataFrame(data_list)  # Convert the list of dictionaries into a DataFrame
    return pos_df


def check_pos_correlation(ant_pos, pot_pos, threshold=0.99):
    # Calculate correlation coefficients
    corr_x = spearmanr(ant_pos[:,0],pot_pos[:,0],nan_policy = 'omit')
    corr_y = spearmanr(ant_pos[:,1],pot_pos[:,1],nan_policy = 'omit')

    # Check if correlations are smaller than the threshold
    return corr_x.statistic,corr_y.statistic 
#     if corr_x.statistic or corr_y.statistic < threshold:
#         raise ValueError(f"Error: The correlation coefficients ({corr_x.statistic} and / or {corr_y.statistic}) are smaller than {threshold}.")
        
        
def get_best_fit_line(pos,raw_pot_pos,plot = True):

    x = pos[:, 0]
    y = pos[:, 1]
    model = LinearRegression().fit(x.reshape(-1, 1), y)
    
    m = model.coef_[0]
    b = model.intercept_

    x_values = np.array([np.min(x), np.max(x)])
    y_values = m * x_values + b
    x_proj=[]
    y_proj=[]

    for i in range(len(raw_pot_pos)):
        x_dot =raw_pot_pos[i, 0]
        y_dot =raw_pot_pos[i, 1]
        x_proj.append((x_dot + m*y_dot - m*b) / (1 + m**2))
        y_proj.append((m*x_dot + (m**2)*y_dot + b) / (1 + m**2))

    if plot == True:
        plt.scatter(x, y, color='lightblue', label='Positions')
        plt.scatter(x_proj, y_proj, color='green', label='Positions')
        plt.plot(x_values, y_values, color='red', label='Best Fit Line')

        plt.title('Best Fit Line for Positions')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend()
        plt.show()
    return x_values, y_values,m,b, x_proj, y_proj

def get_projected_position(x_proj,y_proj,track_start):
    proj_coor = np.array([x_proj,y_proj]).transpose()
    positions = np.zeros(len(proj_coor))
    for i in range(len(proj_coor)):
        positions[i] = np.linalg.norm(proj_coor[i]-track_start)
    return positions 



def get_raw_positions(pos_path,sr_pos,plot=False):
    pos_df = read_position_data(pos_path)
    raw_ptimes = np.arange(len(pos_df))/sr_pos

    raw_ant_pos = pos_df[['a_x', 'a_y']].to_numpy()
    raw_pot_pos = pos_df[['p_x', 'p_y']].to_numpy()
    
    mask_ant = (raw_ant_pos[:, 0] == -1)
    mask_pot = (raw_pot_pos[:, 0] == -1)
# Set values in ant_pos where x is -1 to NaN
    raw_ant_pos[mask_ant, :] = np.nan
    raw_pot_pos[mask_pot, :] = np.nan
    
    if plot == True:

        plot_positions(raw_pot_pos, figsize=(6, 4))
        plt.title('Posterior positions')

    corr_x,corr_y = check_pos_correlation(raw_ant_pos, raw_pot_pos)
    
    print('AP Correlation Coef X:', np.round(corr_x,2))
    print('AP Correlation Coef Y:', np.round(corr_y,2))
    return raw_ant_pos,raw_pot_pos, raw_ptimes, corr_x,corr_y


def get_pre_positions(raw_pos,raw_ptimes):
    nan_mask_pot = np.isnan(raw_pos).any(axis=1)
    valid_mask_pot = ~nan_mask_pot

    pos = raw_pos[valid_mask_pot]
    ptimes = raw_ptimes[valid_mask_pot]
    return pos,ptimes
    
    
def get_track_info(x_coor,y_coor):
    start_idx = np.argmin(x_coor)
    end_idx = np.argmax(x_coor)
    
    track_start = np.array([x_coor[start_idx],y_coor[start_idx]])
    track_end = np.array([x_coor[end_idx],y_coor[end_idx]])

    track_length = np.linalg.norm(track_end-track_start)
    return track_start,track_end,track_length


def get_positions(pos_path,sr_pos,plot=False):
    raw_ant_pos,raw_pot_pos, raw_ptimes, corr_x,corr_y = get_raw_positions(pos_path,sr_pos)
    pos,ptimes = get_pre_positions(raw_pot_pos,raw_ptimes)
    x_coor,y_coor,m,b,x_proj,y_proj = get_best_fit_line(pos)
    track_start,track_end,track_length = get_track_info(x_coor,y_coor)
    positions = get_projected_position(x_proj,y_proj,track_start)
    return ptimes,positions


# def plot_positions_with_speed(raw_ptimes, positions, speed, speed_thresh,ax=None):
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(16, 5))
#     filtered_speed_indx = np.where(speed <= speed_thresh)[0]
    
#     # Plot all positions on the provided ax with a basic color (e.g., blue)
#     ax.plot(raw_ptimes, positions, alpha=0.5)

#     # Highlight positions with filtered speeds using a different color (e.g., red)
#     ax.scatter(raw_ptimes[filtered_speed_indx], positions[filtered_speed_indx], color='green',alpha=0.5, label='Filtered Positions')

#     # Add labels, legend to the ax
#     ax.set_xlabel('Time')
#     ax.set_ylabel('Positions')
#     ax.set_title('Positions with Speed Threshold')
#     ax.legend()
    

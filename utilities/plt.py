import numpy as np
import matplotlib.pyplot as plt


from spiketools.plts.utils import check_ax, get_kwargs, savefig
from spiketools.plts.settings import SET_KWARGS, OTHER_KWARGS
from spiketools.plts.style import set_plt_kwargs

def plot_firing_rates(rates, unit_labels, spike_count, elec, ax=None, ylim_pad=0.2):
    """
    Plots firing rates as bars for each unit in an electrode on the given axis.

    Parameters:
    - rates: List of firing rates for each unit.
    - unit_labels: Labels for each unit.
    - spike_count: Number of spikes for each unit.
    - elec: Electrode number.
    - ax: Matplotlib axis object to plot on.
    - ylim_pad: Padding to add above the highest bar for text annotations.
    """
    bars = ax.bar(unit_labels, rates)  # Plot bars on the provided axis

    # Annotating each bar with the firing rate and number of spikes
    for bar, rate, count in zip(bars, rates, spike_count):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + (max(rates) * ylim_pad), f'FR: {rate:.2f} Hz\nSpk: {count}',
                ha='center', va='bottom')

    #ax.set_ylim(0, max(rates) + max(rates) * ylim_pad)  # Adjust ylim to make room for annotations
    ax.set_ylabel('Firing Rate (Hz)')
    ax.set_title(f'Electrode_{elec}')
    ax.tick_params(axis='x', rotation=45)  # Rotate labels to fit if they overlap
    
def plot_positions_with_speed(raw_ptimes, positions, speed, speed_thresh,ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 5))
    filtered_speed_indx = np.where(speed <= speed_thresh)[0]
    
    # Plot all positions on the provided ax with a basic color (e.g., blue)
    ax.plot(raw_ptimes, positions, alpha=0.5)

    # Highlight positions with filtered speeds using a different color (e.g., red)
    ax.scatter(raw_ptimes[filtered_speed_indx], positions[filtered_speed_indx],s=5, color='green',alpha=0.5, label='Filtered Positions')

    # Add labels, legend to the ax
    ax.set_xlabel('Time')
    ax.set_ylabel('Positions')
    ax.set_title('Positions with Speed Threshold')
    ax.legend()

@set_plt_kwargs
def plot_percentages(steps,percentages,ax= None,**plt_kwargs):
    ax = check_ax(ax,figsize = plt_kwargs.pop('figsize',None))
    ax.plot(steps,percentages,**plt_kwargs)



    

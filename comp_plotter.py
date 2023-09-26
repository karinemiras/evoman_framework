import os
import pandas as pd
import matplotlib.pyplot as plt

def process_single_run(experiment_dir, enemy, run_num):
    # Generate the filenames based on the naming convention
    comp_filename = f'comp-data-{enemy}-{run_num}.csv'

    comp_filepath = os.path.join(experiment_dir, comp_filename)

    if os.path.exists(comp_filepath):
        # Read the CSV file for computer data
        comp_data = pd.read_csv(comp_filepath)

        # Group data by generation and calculate mean values
        comp_data_mean = comp_data.groupby('generation').agg({
            'time': 'sum',
            'cpu_usage_percent': 'mean',
            'memory_usage_MB': 'mean'
        })

        return comp_data_mean

def process_all_runs(experiment_dir, enemy, num_runs):
    avg_comp_data = None

    for run_num in range(num_runs):
        comp_data_mean = process_single_run(experiment_dir, enemy, run_num)

        if comp_data_mean is not None:
            if avg_comp_data is None:
                avg_comp_data = comp_data_mean
            else:
                avg_comp_data += comp_data_mean

    if avg_comp_data is not None:
        avg_comp_data /= num_runs

    return avg_comp_data

def plot_average_data(avg_comp_data):
    # Create subplots for CPU usage, memory usage, and time
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)

    # Plot CPU usage
    ax1.plot(avg_comp_data.index, avg_comp_data['cpu_usage_percent'], label='Average CPU Usage')
    ax1.set_ylabel('CPU Usage (%)')

    # Plot memory usage
    ax2.plot(avg_comp_data.index, avg_comp_data['memory_usage_MB'], label='Average Memory Usage', color='orange')
    ax2.set_ylabel('Memory Usage (MB)')

    # Plot time
    ax3.plot(avg_comp_data.index, avg_comp_data['time'], label='Total Time (s)', color='red')
    ax3.set_xlabel('Generation')
    ax3.set_ylabel('Total Time (s)')

    plt.legend()
    plt.show()

if __name__ == '__main__':
    experiment_dir = 'exp1'  # Replace with your experiment directory
    enemy = 1  # Set to the enemy you want to analyze
    num_runs = 1  # Replace with the number of runs you have

    avg_comp_data = process_all_runs(experiment_dir, enemy, num_runs)
    plot_average_data(avg_comp_data)

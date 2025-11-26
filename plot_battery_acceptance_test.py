import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def plot_battery_data(data_dir, window_title, calculate_capacity=False, capacity_mode='discharge'):
    """
    Plot battery test data from CSV files in the specified directory.
    
    Parameters:
    -----------
    data_dir : str
        Path to directory containing CSV files
    window_title : str
        Title for the plot window
    calculate_capacity : bool
        If True, calculate capacity and annotate the plot
    capacity_mode : str
        'discharge' for voltage-based calculation (V/20 ohms), 'charge' for current-based (from in_curr columns)
    """
    # Find all CSV files in the specified directory
    csv_files = glob.glob(os.path.join(data_dir, '*.CSV')) + glob.glob(os.path.join(data_dir, '*.csv'))
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in {data_dir}")

    print(f"Found {len(csv_files)} CSV file(s) in {data_dir}: {csv_files}")

    # Read all CSV files and combine them
    dataframes = []
    for csv_file in csv_files:
        print(f"Reading {csv_file}...")
        df_temp = pd.read_csv(csv_file)
        dataframes.append(df_temp)

    # Combine all dataframes
    df = pd.concat(dataframes, ignore_index=True)

    # Find and parse timestamp column (ISO 8601 format with space instead of T, no Z)
    timestamp_col = None
    for col in df.columns:
        if 'timestamp' in col.lower():
            timestamp_col = col
            break

    if timestamp_col:
        # Parse ISO 8601 format with space instead of T, no Z
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        # Sort by timestamp to ensure proper time series ordering
        df = df.sort_values(by=timestamp_col).reset_index(drop=True)
        # Calculate elapsed time in minutes from the start
        time_start = df[timestamp_col].iloc[0]
        time_axis = (df[timestamp_col] - time_start).dt.total_seconds() / 60.0  # Convert to minutes
    else:
        raise ValueError("No timestamp column found in CSV file")

    # Filter columns that don't have "spare" or "timestamp" in the title (case-insensitive)
    columns_to_plot = [col for col in df.columns if 'spare' not in col.lower() and 'timestamp' not in col.lower()]

    # Group columns for overplotting
    temp_columns = [col for col in columns_to_plot if 'temp' in col.lower()]
    volt_columns = [col for col in columns_to_plot if 'volt' in col.lower()]
    in_curr_columns = [col for col in columns_to_plot if 'in_curr' in col.lower()]
    out_curr_columns = [col for col in columns_to_plot if 'out_curr' in col.lower()]

    # Get remaining individual columns
    grouped_cols = set(temp_columns + volt_columns + in_curr_columns + out_curr_columns)
    individual_columns = [col for col in columns_to_plot if col not in grouped_cols]

    print(f"Found {len(temp_columns)} temperature columns: {temp_columns}")
    print(f"Found {len(volt_columns)} voltage columns: {volt_columns}")
    print(f"Found {len(in_curr_columns)} input current columns: {in_curr_columns}")
    print(f"Found {len(out_curr_columns)} output current columns: {out_curr_columns}")
    print(f"Found {len(individual_columns)} individual columns: {individual_columns}")

    # Create 2x2 grid for the 4 plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(window_title, fontsize=16, fontweight='bold')
    axes = axes.flatten()

    plot_idx = 0

    # Plot temperature columns together
    if temp_columns:
        ax = axes[plot_idx]
        for col in temp_columns:
            # Assign colors based on column name: bat1=blue, bat2=orange, board=green
            col_lower = col.lower()
            if 'bat1' in col_lower:
                color = 'blue'
            elif 'bat2' in col_lower:
                color = 'orange'
            elif 'board' in col_lower:
                color = 'green'
            else:
                color = None  # Use default matplotlib color cycle
            ax.plot(time_axis, df[col], label=col, color=color)
        ax.set_title('Temperature')
        ax.set_xlabel('Elapsed Time [minutes]')
        ax.set_ylabel('Temperature (C)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Plot voltage columns together
    if volt_columns:
        ax = axes[plot_idx]
        for col in volt_columns:
            ax.plot(time_axis, df[col], label=col)
        ax.set_title('Voltage')
        ax.set_xlabel('Elapsed Time [minutes]')
        ax.set_ylabel('Voltage (V)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Plot input current columns together
    if in_curr_columns:
        ax = axes[plot_idx]
        for col in in_curr_columns:
            ax.plot(time_axis, df[col], label=col)
        ax.set_title('Input Current')
        ax.set_xlabel('Elapsed Time [minutes]')
        ax.set_ylabel('Current (A)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Plot output current columns together
    if out_curr_columns:
        ax = axes[plot_idx]
        for col in out_curr_columns:
            ax.plot(time_axis, df[col], label=col)
        ax.set_title('Output Current')
        ax.set_xlabel('Elapsed Time [minutes]')
        ax.set_ylabel('Current (A)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Plot individual columns
    for col in individual_columns:
        ax = axes[plot_idx]
        ax.plot(time_axis, df[col])
        ax.set_title(col)
        ax.set_xlabel('Elapsed Time [minutes]')
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Hide unused subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].set_visible(False)

    # Calculate capacity if requested
    capacity_estimate = None
    if calculate_capacity:
        # Calculate time differences in hours (time_axis is already in minutes)
        time_deltas = time_axis.diff() / 60.0  # Convert minutes to hours
        time_deltas.iloc[0] = 0  # First row has no previous time
        
        if capacity_mode == 'discharge' and volt_columns:
            # For discharge: calculate current from voltage measurements (V/20 ohms)
            # Sum all voltage columns to get total voltage, then calculate current: I = V / R, where R = 20 ohms
            total_voltage = df[volt_columns].sum(axis=1)
            current_calc = total_voltage / 20.0  # Amperes
            
            # Integrate: capacity = sum(I * dt) in Ah
            capacity_estimate = np.sum(current_calc * time_deltas)
            
            # Add annotation to the figure
            fig.text(0.02, 0.02, f'Estimated Discharge Capacity: {capacity_estimate:.3f} Ah', 
                    fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    verticalalignment='bottom')
        
        elif capacity_mode == 'charge' and in_curr_columns:
            # For charge: use input current telemetry directly
            # Sum all input current columns to get total charge current
            total_current = df[in_curr_columns].sum(axis=1)
            
            # Integrate: capacity = sum(I * dt) in Ah
            capacity_estimate = np.sum(total_current * time_deltas)
            
            # Add annotation to the figure
            fig.text(0.02, 0.02, f'Estimated Charge Capacity: {capacity_estimate:.3f} Ah', 
                    fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    verticalalignment='bottom')

    plt.tight_layout()
    return fig

# Main execution
suncet_data = os.getenv('suncet_data')
if not suncet_data:
    raise ValueError("Environment variable 'suncet_data' is not set")

# Plot data from temperature heater test
data_dir_1 = os.path.join(suncet_data, 'test_data', '2025-11-24_battery_fm1_temperature_heater_test')
fig1 = plot_battery_data(data_dir_1, 'Battery FM1 Temperature Heater Test (2025-11-24)')

# Plot data from discharge capacity test
data_dir_2 = os.path.join(suncet_data, 'test_data', '2025-11-25_battery_fm1_discharge_capacity_test')
fig2 = plot_battery_data(data_dir_2, 'Battery FM1 Discharge Capacity Test (2025-11-25)', 
                         calculate_capacity=True, capacity_mode='discharge')

# Plot data from charge capacity test
data_dir_3 = os.path.join(suncet_data, 'test_data', '2025-11-25_battery_fm1_charge_capacity_test')
fig3 = plot_battery_data(data_dir_3, 'Battery FM1 Charge Capacity Test (2025-11-25)', 
                         calculate_capacity=True, capacity_mode='charge')

plt.show()


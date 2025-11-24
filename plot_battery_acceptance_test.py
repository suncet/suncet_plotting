import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Find all CSV files in the specified directory
suncet_data = os.getenv('suncet_data')
if not suncet_data:
    raise ValueError("Environment variable 'suncet_data' is not set")

data_dir = os.path.join(suncet_data, 'test_data', '2025-11-24_battery_fm1_temperature_heater_test')
csv_files = glob.glob(os.path.join(data_dir, '*.CSV')) + glob.glob(os.path.join(data_dir, '*.csv'))
if not csv_files:
    raise FileNotFoundError(f"No CSV file found in {data_dir}")

print(f"Found {len(csv_files)} CSV file(s): {csv_files}")

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
    time_axis = df[timestamp_col]
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
axes = axes.flatten()

plot_idx = 0

# Plot temperature columns together
if temp_columns:
    ax = axes[plot_idx]
    for col in temp_columns:
        ax.plot(time_axis, df[col], label=col)
    ax.set_title('Temperature')
    ax.set_xlabel('Time')
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
    ax.set_xlabel('Time')
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
    ax.set_xlabel('Time')
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
    ax.set_xlabel('Time')
    ax.set_ylabel('Current (A)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plot_idx += 1

# Plot individual columns
for col in individual_columns:
    ax = axes[plot_idx]
    ax.plot(time_axis, df[col])
    ax.set_title(col)
    ax.set_xlabel('Time')
    ax.set_ylabel(col)
    ax.grid(True, alpha=0.3)
    plot_idx += 1

# Hide unused subplots
for idx in range(plot_idx, len(axes)):
    axes[idx].set_visible(False)

plt.tight_layout()
plt.show()


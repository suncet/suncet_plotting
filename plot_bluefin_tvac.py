import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os

def fix_timestamps(timestamps):
    """Fix any backwards jumps in timestamps by enforcing monotonicity.
    When a backwards jump is detected, all subsequent timestamps are shifted by the same amount."""
    fixed_timestamps = timestamps.copy()
    for i in range(1, len(timestamps)):
        if fixed_timestamps[i] <= fixed_timestamps[i-1]:
            # Calculate the shift needed to make this timestamp monotonic
            shift = fixed_timestamps[i-1] + 60 - fixed_timestamps[i]
            # Apply the same shift to all subsequent timestamps
            fixed_timestamps[i:] += shift
    return fixed_timestamps

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(os.path.expanduser("~/Dropbox/suncet_dropbox/9000 Processing/data/test_data/2025-10-24_bluefin_fm2_acceptance_temp_vs_freq/decoded/csv/xband_hk_pkt.csv"))

# Extract the required columns
adc = df['xband_hk_power_amp_temp_raw'].astype(float)
current = df['xband_hk_input_amps'].astype(float)
voltage = df['xband_hk_input_volts'].astype(float)
timestamp = df['ccsdsSecHeader2_sec_xband_hk_pkt'].astype(float)

# Fix timestamps and calculate time in minutes from start
fixed_timestamp = fix_timestamps(timestamp)
time_minutes = (fixed_timestamp - fixed_timestamp.iloc[0]) / 60

# Calculate power
power = current * voltage

# Apply the temperature formula step by step
def adc_to_degC(x):
    C = [-0.1168449147, 4.84010647, -82.1852667, 462.556891]
    adc_frac = x / 4095.0
    r_therm = 5000.0 * adc_frac / (1 - adc_frac)
    r = np.log(r_therm)
    return C[0] * r**3 + C[1] * r**2 + C[2] * r + C[3]

temperature_calculated = adc_to_degC(adc)

# Create first figure with 4 subplots
fig1, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

# Plot temperature (calibrated)
ax1.plot(time_minutes, temperature_calculated, label='Power Amp Temperature (calibrated)')
ax1.set_ylabel('Temperature (°C)')
ax1.legend()
ax1.grid(True)

# Plot power
ax2.plot(time_minutes, power, label='Power', color='red')
ax2.set_ylabel('Input Power (W)')
ax2.legend()
ax2.grid(True)

# Plot current
ax3.plot(time_minutes, current, label='Current', color='green')
ax3.set_ylabel('Current (A)')
ax3.legend()
ax3.grid(True)

# Plot voltage
ax4.plot(time_minutes, voltage, label='Voltage', color='purple')
ax4.set_ylabel('Voltage (V)')
ax4.set_xlabel('Time (minutes)')
ax4.legend()
ax4.grid(True)

plt.suptitle('X-Band Power Amp Parameters vs. Time')
plt.tight_layout()

# Create first figure with 4 subplots
fig1, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

# Plot temperature (calibrated)
ax1.plot(temperature_calculated, label='Power Amp Temperature (calibrated)')
ax1.set_ylabel('Temperature (°C)')
ax1.legend()
ax1.grid(True)

# Plot power
ax2.plot(power, label='Power', color='red')
ax2.set_ylabel('Input Power (W)')
ax2.legend()
ax2.grid(True)

# Plot current
ax3.plot(current, label='Current', color='green')
ax3.set_ylabel('Current (A)')
ax3.legend()
ax3.grid(True)

# Plot voltage
ax4.plot(voltage, label='Voltage', color='purple')
ax4.set_ylabel('Voltage (V)')
ax4.set_xlabel('Sample Index')
ax4.legend()
ax4.grid(True)

plt.suptitle('X-Band Power Amp Parameters vs. Sample Index')
plt.tight_layout()


# Create second figure for all current measurements
fig2, ax = plt.subplots(figsize=(12, 6))

# Find all columns containing 'amps' and plot them
current_columns = [col for col in df.columns if 'amps' in col.lower()]
for col in current_columns:
    ax.plot(time_minutes, df[col].astype(float), label=col)

ax.set_ylabel('Current (A)')
ax.set_xlabel('Time (minutes)')
ax.legend()
ax.grid(True)
plt.title('All Current Measurements vs. Time')
plt.tight_layout()

pass
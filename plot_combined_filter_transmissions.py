import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

mesh_transmission = 0.98

# Paths to your CSV files
csv_path_1 = '/Users/masonjp2/Dropbox/suncet_dropbox/9000 Processing/data/filter_transmission/C_20nm_thick_0.01-2066nm_range.csv'
csv_path_2 = '/Users/masonjp2/Dropbox/suncet_dropbox/9000 Processing/data/filter_transmission/Al_150nm_thick_0.01-1250nm_range.csv'

# Read the CSV files
df1 = pd.read_csv(csv_path_1)
df2 = pd.read_csv(csv_path_2)

common_wavelength = df2['wavelength [angstrom]']

# Interpolation functions
interp_func_1 = interp1d(df1['wavelength [angstrom]'], df1['transmission'], kind='linear', bounds_error=False, fill_value="extrapolate")

# Interpolate to the common grid
transmission_1_on_common_grid = interp_func_1(common_wavelength)
transmission_2_on_common_grid = df2['transmission']

# Multiply the transmission values
combined_transmission = transmission_1_on_common_grid * transmission_2_on_common_grid * mesh_transmission

# Create a DataFrame and save to CSV
combined_df = pd.DataFrame({
    'wavelength [nm]': common_wavelength/10,
    'transmission': combined_transmission
})

# Save to CSV
combined_df.to_csv('combined_transmission.csv', index=False)

plt.figure(figsize=(10, 6))
plt.plot(combined_df['wavelength [nm]'], combined_df['transmission'])
plt.yscale('log')
plt.xlabel('Wavelength [nm]')
plt.ylabel('Transmission [% as fraction]')
plt.title('Combined Transmission: 20 nm C, 150 nm Al, 98% transmissive nickel mesh')
plt.grid(True)
plt.show()


pass
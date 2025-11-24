import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.constants import h, c, e
from scipy.integrate import simps

def read_whi_reference_spectrum():
    dataloc = os.path.join(os.getenv('suncet_data'), 'reference_solar_spectrum')
    file_path = os.path.join(dataloc, 'ref_solar_irradiance_whi-2008_ver2.dat')

    data = pd.read_csv(file_path, skiprows=142, delim_whitespace=True, header=None)
    wavelength = data.iloc[:, 0].values  # [nm]
    irradiance = data.iloc[:, 2].values  # [W/m^2/nm]

    return {
        'wavelength': wavelength,
        'irradiance': irradiance,
        'wave_unit': 'nm',
        'irrad_unit': 'W/m^2/nm'
    }


def suncet_load_final_mirror_coating_measurements(fm=1, separate_mirrors=False):
    # Set up environment variable
    dataloc = os.path.join(os.getenv('suncet_data'), 'mirror_reflectivity', '2024-03-21 rigaku measurements final')

    if fm == 1:
        m1_filename = 'm1_sn2_final.csv'
        m2_filename = 'm2_sn3_final.csv'

    # Read data
    rigaku_m1 = pd.read_csv(os.path.join(dataloc, m1_filename))
    rigaku_m2 = pd.read_csv(os.path.join(dataloc, m2_filename))

    if not separate_mirrors:
        common_wavelength = np.linspace(
            min(rigaku_m1['wavelength [nm]'].min(), rigaku_m2['wavelength [nm]'].min()),
            max(rigaku_m1['wavelength [nm]'].max(), rigaku_m2['wavelength [nm]'].max()),
            200
        )
        interp_reflectivity_m1 = np.interp(common_wavelength, rigaku_m1['wavelength [nm]'], rigaku_m1['reflectivity [% as fraction]'])
        interp_reflectivity_m2 = np.interp(common_wavelength, rigaku_m2['wavelength [nm]'], rigaku_m2['reflectivity [% as fraction]'])
        average_reflectivity = np.maximum((interp_reflectivity_m1 + interp_reflectivity_m2) / 2.0, 0.0)
        rigaku = {'wavelength [nm]': common_wavelength, 'reflectivity': average_reflectivity}
        return rigaku
    else:
        return {'rigaku_m1': rigaku_m1, 'rigaku_m2': rigaku_m2}


def read_ascii(file_path, template=None):
    return pd.read_csv(file_path)


def integrate(x, y):
    return simps(y, x)


def calculate_effective_area_and_electrons(mirror_coating='flight_fm1', comparisons=None):
    comparisons = list(map(str.lower, comparisons))

    # Set up environment variable
    base_path = os.getenv('suncet_data')
    reflectivity_path = os.path.join(base_path, 'mirror_reflectivity')

    # Constants
    j2ev = 6.242e18  # [ev/J]
    arcsec2rad = 4.8481e-6  # [radian/arcsec]
    one_au_cm = 1.496e13  # [cm]
    average_rsun_arc = 959.63  # [arcsec]
    rsun_cm = 6.957e10  # [cm]
    one_au_sun_sr = 6.7993e-5  # [sr]

    # Load full solar spectrum
    solar_spectrum = read_whi_reference_spectrum()

    # Instrument parameters
    entrance_aperture = 6.5  # [cm]
    secondary_mirror_obscuration = 0.413  # [% as a fraction]
    aperture = np.pi * (entrance_aperture / 2) ** 2 * (1 - secondary_mirror_obscuration)
    mesh_transmission = 0.95
    quantum_efficiency = 0.85
    exposure_time = 15.0  # [seconds]

    # Load and interpolate filter transmission data
    single_filter_transmission = read_ascii(os.path.join(base_path, 'filter_transmission/Al_150nm_thick_0.01-1250nm_range.csv'))
    filter_wavelength = single_filter_transmission['wavelength [angstrom]'] / 10.0  # [nm]
    carbon_transmission = read_ascii(os.path.join(base_path, 'filter_transmission/C_20nm_thick_0.01-2066nm_range.csv'))
    carbon_wavelength = carbon_transmission['wavelength [angstrom]'] / 10.0  # [nm]
    carbon_transmission = np.interp(filter_wavelength, carbon_wavelength, carbon_transmission['transmission'])
    filter_transmission_raw = single_filter_transmission['transmission'] ** 2 * carbon_transmission * mesh_transmission
    filter_transmission = np.interp(solar_spectrum['wavelength'], filter_wavelength, filter_transmission_raw)

    # Load and interpolate mirror reflectivity data
    if mirror_coating.lower() == 'flight_fm1':
        b4c = suncet_load_final_mirror_coating_measurements(fm=1)
        r_wave = b4c['wavelength [nm]']
        reflect = b4c['reflectivity']
    elif mirror_coating.lower() == 'b4c':
        b4c = read_ascii(os.path.join(reflectivity_path, 'XRO47864_TH=5.0.txt'))
        r_wave = b4c['wave']  # [nm]
        reflect = b4c['reflectance']
    else:
        raise ValueError('No matching mirror coating supplied. Must be either "B4C", "AlZr", or "SiMo".')

    mirror_reflectivity = np.interp(solar_spectrum['wavelength'], r_wave, reflect)

    # Effective area calculation
    effective_area = np.maximum(aperture * mirror_reflectivity ** 2 * filter_transmission, 0)  # [cm^2]
    quantum_yield = (h * c / (solar_spectrum['wavelength'] * 1e-9)) * j2ev / 3.63  # [e-/phot]

    # Load effective area data for GOES/SUVI and EUVE
    suvi_171_effective_area = read_ascii(os.path.join(base_path, 'effective_area/suvi_171_effective_area.csv'))
    suvi_195_effective_area = read_ascii(os.path.join(base_path, 'effective_area/suvi_195_effective_area.csv'))
    euve = pd.read_csv(os.path.join(base_path, 'effective_area/EUVE_Deep_Survey_B_Aeff.csv'), skiprows=1)
    
    # Make wavelength units consistent
    euve['wavelength [angstrom]'] /= 10.0
    euve.rename(columns={'wavelength [angstrom]': 'wavelength [nm]'}, inplace=True)

    # Some statistics
    integrated_effective_area = integrate(solar_spectrum['wavelength'], effective_area)
    main_bandpass_indices = (solar_spectrum['wavelength'] >= 15) & (solar_spectrum['wavelength'] <= 25)
    integrated_effective_area_main_bandpass = integrate(solar_spectrum['wavelength'][main_bandpass_indices], effective_area[main_bandpass_indices])
    suvi_171_integrated = integrate(suvi_171_effective_area['x'][1:-1], suvi_171_effective_area[' y'][1:-1])
    suvi_195_integrated = integrate(suvi_195_effective_area['x'], suvi_195_effective_area[' y'])
    euve_integrated = integrate(euve['wavelength [nm]'], euve['effective area [cm2]'])

    # Push spectrum through effective area and detector
    irradiance = solar_spectrum['irradiance'] * 1e-4  # [W/cm^2/nm]
    irradiance_photons = irradiance / (h * c / (solar_spectrum['wavelength'] * 1e-9))  # [photons/s/cm^2/nm]
    instrument_response = np.maximum(irradiance_photons * exposure_time * effective_area * quantum_efficiency * quantum_yield, 0)  # [electrons/nm]

    # Get per pixel response as well
    radiance = irradiance_photons / 2.16e-5  # [photons/s/cm^2/nm/sr]
    radiance *= 5.42e-10  # [photons/s/cm^2/nm/pixel]
    instrument_response_per_pixel = np.maximum(radiance * exposure_time * effective_area * quantum_efficiency * quantum_yield, 0)  # [electrons/nm/pixel]
    in_band_indices = (solar_spectrum['wavelength'] >= 17) & (solar_spectrum['wavelength'] <= 20)
    short_indices = solar_spectrum['wavelength'] < 17
    long_indices = solar_spectrum['wavelength'] > 20
    instrument_response_per_pixel_in_band = integrate(solar_spectrum['wavelength'][in_band_indices], instrument_response_per_pixel[in_band_indices])
    instrument_response_per_pixel_short = integrate(solar_spectrum['wavelength'][short_indices], instrument_response_per_pixel[short_indices])
    instrument_response_per_pixel_long = integrate(solar_spectrum['wavelength'][long_indices], instrument_response_per_pixel[long_indices])
    instrument_response_per_pixel_out_of_band = instrument_response_per_pixel_short + instrument_response_per_pixel_long

    # Create plots
    plt.figure()
    plt.plot(solar_spectrum['wavelength'], effective_area, linewidth=2)
    plt.xscale('log')
    plt.ylim(-0.1, 1.2)
    plt.xlim(10, 2500)
    plt.xlabel('wavelength [nm]')
    plt.ylabel('effective area [cm^2]')
    plt.title('SunCET baseline config')
    plt.text(0.6, 0.8, f'integral = {integrated_effective_area:.2f}', transform=plt.gca().transAxes)
    plt.axhline(0, color='tomato', linestyle='--')

    plt.figure()
    plt.plot(solar_spectrum['wavelength'], effective_area, linewidth=3, color='black')
    plt.ylim(-0.1, 1.2)
    plt.xlim(15, 25)
    plt.xlabel('wavelength [nm]')
    plt.ylabel('effective area [cm^2]')
    plt.title('SunCET baseline config')
    plt.text(0.95, 0.8, f'SunCET integral = {integrated_effective_area_main_bandpass:.2f}', transform=plt.gca().transAxes, ha='right')
    lowest_text=0.8
    if 'suvi' in comparisons:
        plt.plot(suvi_171_effective_area['x'], suvi_171_effective_area[' y'], color='grey', linestyle='--')
        plt.plot(suvi_195_effective_area['x'], suvi_195_effective_area[' y'], color='grey', linestyle='--')
        plt.text(0.95, lowest_text-0.05, f'GOES/SUVI 171 integral = {suvi_171_integrated:.2f}', color='grey', transform=plt.gca().transAxes, ha='right')
        plt.text(0.95, lowest_text-0.1, f'GOES/SUVI 195 integral = {suvi_195_integrated:.2f}', color='grey', transform=plt.gca().transAxes, ha='right')
        lowest_text-= 0.1
    if 'euve' in comparisons: 
        plt.plot(euve['wavelength [nm]'], euve['effective area [cm2]'], color='dodgerblue', linestyle='--')
        plt.text(0.95, lowest_text-0.05, f'EUVE Deep Survey integral = {euve_integrated:.2f}', color='dodgerblue', transform=plt.gca().transAxes, ha='right')
    plt.axhline(0, color='tomato', linestyle='--')

    plt.figure()
    plt.plot(solar_spectrum['wavelength'], instrument_response, linewidth=2)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(-1e5, 2e12)
    plt.xlim(10, 2500)
    plt.xlabel('wavelength [nm]')
    plt.ylabel('instrument response [electrons/nm]')
    plt.title('solar spectrum through SunCET in baseline config')

    plt.figure()
    plt.plot(solar_spectrum['wavelength'], instrument_response_per_pixel, linewidth=2)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(1e-10, 1e10)
    plt.xlim(10, 2500)
    plt.xlabel('wavelength [nm]')
    plt.ylabel('instrument response [electrons/nm/pixel]')
    plt.title('solar spectrum through SunCET in baseline config')
    plt.fill_between([17, 20], 1e-10, 1e10, color='dodgerblue', alpha=0.3)
    plt.text(0.95, 0.80, f'in-band integrated response = {instrument_response_per_pixel_in_band:.0f} electrons/pixel', transform=plt.gca().transAxes, ha='right', color='dodgerblue')
    plt.text(0.95, 0.75, f'out-of-band integrated response = {instrument_response_per_pixel_out_of_band:.0f} electrons/pixel', transform=plt.gca().transAxes, ha='right')
    plt.text(0.95, 0.70, f'ratio = {instrument_response_per_pixel_in_band/instrument_response_per_pixel_out_of_band:.0f}x', transform=plt.gca().transAxes, ha='right')

    pass

calculate_effective_area_and_electrons(comparisons=['SUVI', 'EUVE'])

/**
 * Curated SunCET beacon display metadata.
 *
 * Names, engineering units, and enum labels follow the processing pipeline's
 * public_beacon_schema.csv. Additional Hydra display names follow its public
 * realtime_display/color_limits_tlm.xml and the existing display layout.
 * Database values have already been decoded into engineering units: conversion
 * coefficients must not be applied again here. Body rates receive only the
 * display-unit conversion from radians per second to degrees per second.
 */

export const GROUPS = [
  { id: 'temperatures', label: 'Temperatures', description: 'Solar arrays, batteries, avionics, instruments, and radios.' },
  { id: 'electrical', label: 'Power', description: 'Measured voltage and current, paired by supply. Power is calculated as V × I.' },
  { id: 'modes', label: 'Modes & system states', description: 'Spacecraft modes, subsystem power, charging, and aliveness.' },
  { id: 'attitude', label: 'Attitude & pointing', description: 'Body rates, wheel speeds, pointing error, and ADCS validity.' },
  { id: 'timing', label: 'Timing & counters', description: 'Uptime, mission elapsed time, command loss timer, and resets.' },
  { id: 'heaters', label: 'Heaters', description: 'Battery and CSIE heater enable states.' },
  { id: 'storage', label: 'NAND storage', description: 'Read and write pointers for housekeeping, attitude, science, and logs.' },
  { id: 'science', label: 'Science', description: 'CSIE histogram, Dual-SPS flare telemetry, and measured Sun positions.' },
];

const yesNo = { 0: 'NO', 1: 'YES' };
const onOff = { 0: 'OFF', 1: 'ON' };
const charging = { 0: 'DISCHARGING', 1: 'CHARGING' };
const systemMode = { 0: 'PHOENIX', 1: 'SAFE', 2: 'SCIENCE', 3: 'DOWNLINK' };
const adcsMode = { 0: 'SUN_POINT', 1: 'FINE_REF_POINT' };
const sunPointState = {
  0: 'SUN_POINT', 1: 'FINE_REF_POINT', 2: 'SEARCH_INIT', 3: 'SEARCHING',
  4: 'WAITING', 5: 'CONVERGING', 6: 'ON_SUN', 7: 'NOT_ACTV',
};
const flarePhase = {
  0: 'NOT_IN_SUN', 1: 'FILLING_HISTORY', 2: 'NOT_IN_FLARE',
  4: 'FLARE_LIKELY', 24: 'IN_FLARE_DECREASING', 40: 'IN_FLARE_RISING',
};

const number = (field, label, unit, group, note) => ({
  field, label, unit, group, kind: 'number', ...(note ? { note } : {}),
});
const state = (field, label, group, enums, note) => ({
  field, label, unit: '', group, kind: 'state',
  ...(enums ? { enums } : {}), ...(note ? { note } : {}),
});

const bodyRate = axis => {
  const field = `beac_adcs_body_rt${axis}`;
  return {
    ...number(field, `Body rate ${axis}`, 'deg/s', 'attitude',
      'Converted from decoded radians per second to degrees per second.'),
    derive(row) {
      const value = row[field];
      if (value === null || value === undefined ||
          !['number', 'bigint', 'boolean', 'string'].includes(typeof value) ||
          (typeof value === 'string' && value.trim() === '')) return null;
      const degrees = Number(value) * 180 / Math.PI;
      return Number.isFinite(degrees) ? degrees : null;
    },
  };
};

export const POWER_PAIRS = [
  { id: 'eps_bus', label: 'EPS bus', voltage: 'beac_ana_eps_bus_v', current: 'beac_ana_eps_bus_i' },
  { id: 'sa_8_cell', label: 'Solar array · 8-cell string', voltage: 'beac_ana_sa_8_cell_str_v', current: 'beac_ana_sa_8_cell_str_i' },
  { id: 'sa_9_cell', label: 'Solar array · 9-cell string', voltage: 'beac_ana_sa_9_cell_str_v', current: 'beac_ana_sa_9_cell_str_i' },
  {
    id: 'battery_1', label: 'Battery 1 · charging',
    voltage: 'beac_ana_bat1_v', current: 'beac_batt1_charge_current',
    note: 'Voltage × charge current estimates charge power; this is not a measurement of net battery power or discharge power.',
  },
  {
    id: 'battery_2', label: 'Battery 2 · charging',
    voltage: 'beac_ana_bat2_v', current: 'beac_batt2_charge_current',
    note: 'Voltage × charge current estimates charge power; this is not a measurement of net battery power or discharge power.',
  },
  { id: '3p3', label: '3.3 V rail', voltage: 'beac_ana_3p3_v', current: 'beac_ana_3p3_i' },
  { id: 'adcs', label: 'ADCS / XACT', voltage: 'beac_ana_xact_v', current: 'beac_ana_xact_i' },
  { id: 'uhf', label: 'UHF', voltage: 'beac_ana_uhf_v', current: 'beac_ana_uhf_i' },
  { id: 'xband', label: 'X-band', voltage: 'beac_ana_xband_v', current: 'beac_ana_xband_i' },
  { id: 'csie', label: 'CSIE', voltage: 'beac_ana_csie_volt', current: 'beac_ana_csie_curr' },
  { id: 'dsps', label: 'Dual-SPS', voltage: 'beac_ana_dsps_volt', current: 'beac_ana_dsps_curr' },
];

export const SIGNALS = [
  // Keep every temperature together, with related sensors next to each other.
  ...[
    ['beac_ana_sa_minus_y_temp', 'Solar array −Y'],
    ['beac_ana_sa_plus_y_temp', 'Solar array +Y'],
    ['beac_batt1_temp', 'Battery 1'],
    ['beac_batt_board_temp', 'Battery board'],
    ['beac_ana_cdh_temp', 'CDH'],
    ['beac_ana_eps_temp', 'EPS'],
    ['beac_ana_ifb_therm1', 'Interface board'],
    ['beac_csie_temp', 'CSIE detector'],
    ['beac_dsps_sensor_board_temp', 'Dual-SPS sensor board'],
    ['beac_adcs_ana_motor1_temp', 'ADCS motor 1'],
    ['beac_uhf_temp', 'UHF'],
    ['beac_xband_pa_temp', 'X-band power amplifier'],
  ].map(([field, label]) => number(field, label, '°C', 'temperatures')),

  ...POWER_PAIRS.flatMap(pair => [
    number(pair.voltage, `${pair.label} voltage`, 'V', 'electrical'),
    number(pair.current, `${pair.label} current`, 'A', 'electrical', pair.note),
  ]),
  number('beac_ana_cdh_3p3_ref', 'CDH 3.3 V reference', 'V', 'electrical'),
  number('beac_ana_eps_3p3_ref', 'EPS 3.3 V reference', 'V', 'electrical'),
  number('beac_xband_pa_curr', 'X-band power amplifier current', 'A', 'electrical',
    'The beacon has no matching amplifier voltage measurement, so amplifier power is not calculated.'),

  state('beac_mode_system_mode', 'CDH mode', 'modes', systemMode),
  state('beac_adcs_mode', 'ADCS mode', 'modes', adcsMode),
  state('beac_csie_cap_state', 'CSIE capture state', 'modes', undefined,
    'Capture-state codes are displayed as stored; the public beacon schema does not define their labels.'),
  ...[
    ['adcs', 'ADCS'], ['uhf', 'UHF'], ['xband', 'X-band'], ['csie', 'CSIE'], ['dsps', 'Dual-SPS'],
  ].map(([field, label]) => state(`beac_eps_pwr_state_${field}`, `${label} power state`, 'modes', onOff)),
  state('beac_adcs_alive', 'ADCS aliveness', 'modes', undefined,
    'Displayed as stored; the public beacon schema does not define numeric aliveness labels for ADCS.'),
  state('beac_uhf_alive', 'UHF aliveness', 'modes', { 0: 'OFF', 1: 'ALIVE', 2: 'DEAD' }),
  state('beac_batt1_charging_state', 'Battery 1 charging state', 'modes', charging),
  state('beac_batt2_charging_state', 'Battery 2 charging state', 'modes', charging),
  state('beac_telescope_door_pin_pulled', 'Telescope door release pin', 'modes', { 0: 'PULLED', 1: 'ENGAGED' }),

  ...[1, 2, 3].map(bodyRate),
  ...[1, 2, 3].map(wheel => number(`beac_adcs_wheel_sp${wheel}`, `Wheel ${wheel} speed`, 'rpm', 'attitude')),
  number('beac_adcs_att_ctrl_sun_point_ang_err', 'Sun pointing angle error', '°', 'attitude'),
  state('beac_adcs_sun_point_state', 'Sun pointing state', 'attitude', sunPointState),
  state('beac_adcs_att_vld', 'Attitude valid', 'attitude', yesNo),
  state('beac_adcs_ref_vld', 'References valid', 'attitude', yesNo),
  state('beac_adcs_time_vld', 'Time valid', 'attitude', yesNo),

  number('sequence_count', 'Beacon sequence count', 'count', 'timing'),
  number('beac_time_since_boot', 'Time since boot', 's', 'timing'),
  number('beac_time_alive', 'Time alive', 's', 'timing'),
  number('beac_time_mission_elapsed_time', 'Mission elapsed time', 's', 'timing'),
  number('beac_mode_seconds_since_mode_change', 'Time since mode change', 's', 'timing'),
  number('beac_mode_clt_hours_left', 'Command loss timer remaining', 'h', 'timing'),
  number('beac_num_sc_resets', 'Spacecraft resets', 'count', 'timing'),

  state('beac_battery1_heater_enable', 'Battery 1 heater enabled', 'heaters', yesNo),
  state('beac_battery2_heater_enable', 'Battery 2 heater enabled', 'heaters', yesNo),
  state('beac_csie_heater_enable', 'CSIE heater enabled', 'heaters', yesNo),

  ...[
    ['hk', 'Housekeeping'], ['adcs', 'ADCS'], ['dsps', 'Dual-SPS'], ['sci', 'Science'],
  ].flatMap(([partition, label]) => [
    number(`sw_store_partition_write_${partition}_beac`, `${label} write pointer`, 'address', 'storage'),
    number(`sw_store_partition_read_${partition}_beac`, `${label} read pointer`, 'address', 'storage'),
  ]),
  number('beac_store_partition_write_log', 'Log write pointer', 'address', 'storage'),
  number('beac_csie_nand_sci_write_ptr', 'CSIE image write pointer', 'address', 'storage'),
  number('beac_csie_meta_nand_sci_write_ptr', 'CSIE metadata write pointer', 'address', 'storage'),

  ...[0, 1, 2, 3, 4, 5].map(bin => number(
    `beac_csie_img_hist_${bin}`, `CSIE histogram bin ${bin}`, 'count', 'science',
    'Histogram bin ranges are configurable; the beacon carries the first six bins of the full histogram.',
  )),
  state('beac_dsps_flare_phase', 'Dual-SPS flare phase', 'science', flarePhase,
    'This field contains bit flags. Known combinations use the public beacon labels; other combinations retain their stored code.'),
  number('beac_dsps_flare_level', 'Dual-SPS flare trigger level', 'log10(XRS-B flux)', 'science'),
  number('beac_dsps_flare_magnitude', 'Dual-SPS flare magnitude', 'log10(XRS-B flux)', 'science'),
  ...[
    ['visible', 'Visible'], ['x_ray', 'X-ray'],
  ].flatMap(([sensor, label]) => ['x', 'y'].map(axis => number(
    `beac_dsps_${sensor}_sps_sun_pos_${axis}`, `${label} SPS Sun position ${axis.toUpperCase()}`, 'arcsec', 'science',
  ))),
];

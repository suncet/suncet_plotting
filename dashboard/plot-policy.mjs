import { numericValue, powerValue } from './data.mjs';

/**
 * Display guardrails, not operating limits or telemetry validation.
 *
 * These deliberately include off rails, negative sensor noise, hot/cold hardware,
 * and the 181-degree Sun-pointing sentinel. Hydra's much narrower red/yellow
 * thresholds are alarms and must not be used to hide real excursions. Values
 * outside a display range remain in the plotted traces and in CSV exports.
 */
export function engineeringBounds(signal) {
  if (!signal || signal.kind === 'state') return null;
  if (Array.isArray(signal.displayRange) && signal.displayRange.length === 2 &&
      signal.displayRange.every(Number.isFinite) && signal.displayRange[0] < signal.displayRange[1]) {
    return [...signal.displayRange];
  }
  const field = String(signal.field || '').toLowerCase();
  switch (signal.unit) {
    case '°C': return field.includes('_sa_') ? [-50, 100] : [-20, 50];
    case 'V':
      if (field.includes('3p3')) return [-0.25, 5];
      if (/_(csie|dsps)_/.test(field)) return [-0.5, 15];
      return [-1, 32];
    case 'A': return /eps_bus|batt\d+_charge/.test(field) ? [-10, 10] : [-0.25, 5];
    case 'W': return [-320, 320];
    case 'deg/s': return [-180 / Math.PI, 180 / Math.PI];
    case 'rpm': return [-10000, 10000];
    case '°': return field.includes('sun_point_ang_err') ? [-1, 185] : [-360, 360];
    case 'arcsec': return [-36000, 36000];
    case 'log10(XRS-B flux)': return [-12, 1];
    // Counts, elapsed times, and NAND addresses have no defensible fixed mission
    // maximum. Their default display uses a robust fit instead of a made-up cap.
    default: return null;
  }
}

/** Metadata for same-packet V × I power, with guards inherited from both rails. */
export function derivePowerSignal(pair) {
  const voltage = engineeringBounds({ field: pair.voltage, unit: 'V' });
  const current = engineeringBounds({ field: pair.current, unit: 'A' });
  const corners = voltage.flatMap(v => current.map(i => v * i));
  return {
    field: `${pair.id}_power`,
    label: `${pair.label} power`,
    unit: 'W',
    group: 'electrical',
    kind: 'number',
    displayRange: [Math.min(...corners), Math.max(...corners)],
    rangeValue: row => powerRangeValue(row, pair),
  };
}

/**
 * Value used only to fit a power axis. A corrupt voltage multiplied by a small
 * current can produce an apparently plausible wattage; guard both source
 * measurements first. The raw trace/CSV must still use powerValue(row, pair).
 */
export function powerRangeValue(row, pair) {
  for (const [field, unit] of [[pair.voltage, 'V'], [pair.current, 'A']]) {
    const value = numericValue(row[field]);
    const [low, high] = engineeringBounds({ field, unit });
    if (value === null || value < low || value > high) return null;
  }
  return powerValue(row, pair);
}

const minimumSpans = {
  '°C': 5, V: 0.5, A: 0.1, W: 1, 'deg/s': 0.01 * 180 / Math.PI, rpm: 100,
  '°': 1, arcsec: 30, 'log10(XRS-B flux)': 0.2,
  count: 1, address: 1, s: 1, h: 1,
};
const nonnegativeUnits = new Set(['count', 'address', 's', 'h']);

function quantile(sorted, fraction) {
  const index = (sorted.length - 1) * fraction;
  const lower = Math.floor(index), upper = Math.ceil(index), remainder = index - lower;
  return sorted[lower] * (1 - remainder) + sorted[upper] * remainder;
}

function paddedRange(low, high, minimumSpan, bounds = null) {
  const center = low / 2 + high / 2;
  const width = Math.max(minimumSpan, (high - low) * 1.16);
  let range = [center - width / 2, center + width / 2];
  if (bounds) {
    range = [Math.max(bounds[0], range[0]), Math.min(bounds[1], range[1])];
    // Preserve the minimum span even for a constant at a guardrail boundary.
    if (range[1] - range[0] < minimumSpan) {
      range[1] = Math.min(bounds[1], range[0] + minimumSpan);
      range[0] = Math.max(bounds[0], range[1] - minimumSpan);
    }
  }
  return range.map((v, i) => Number.isFinite(v) ? v : (i === 0 ? -Number.MAX_VALUE : Number.MAX_VALUE));
}

function robustRange(sorted, minimumSpan, nonnegative) {
  let candidates = nonnegative ? sorted.filter(v => v >= 0) : sorted;
  if (!candidates.length) return sorted.length && nonnegative ? [0, minimumSpan] : null;
  // A percentile alone still follows an isolated million-valued sample in a
  // short selection. A deliberately loose median/MAD fence handles that case.
  // The scale floor preserves ordinary small counter changes and sensor noise.
  if (candidates.length >= 3) {
    const median = quantile(candidates, 0.5);
    const deviations = candidates.map(v => Math.abs(v - median)).sort((a, b) => a - b);
    const mad = quantile(deviations, 0.5);
    const fence = 12 * Math.max(mad, Math.abs(median) * 0.02, minimumSpan / 2);
    candidates = candidates.filter(v => Math.abs(v - median) <= fence);
  }
  const low = quantile(candidates, 0.01), high = quantile(candidates, 0.99);
  return paddedRange(low, high, minimumSpan, nonnegative ? [0, Number.MAX_VALUE] : null);
}

/**
 * Choose an axis range without changing any samples.
 *
 * Engineering mode fits all in-guard values, preserving every plausible
 * transient. Unbounded quantities use the robust central fit. Typical mode
 * explicitly requests that robust fit for any quantity; full mode delegates to
 * Plotly's ordinary autorange. `outside` counts finite raw values beyond the
 * final displayed range, including values rejected only for scale selection.
 */
export function selectRange(signals, values, mode = 'engineering') {
  if (mode === 'full') return { range: null, outside: 0, basis: 'full' };
  const sorted = values.map(numericValue).filter(v => v !== null).sort((a, b) => a - b);
  const minimumSpan = Math.max(1e-12, ...signals.map(s => minimumSpans[s.unit] || 1));
  const bounds = signals.map(engineeringBounds);
  let range, basis;
  if (mode === 'engineering' && bounds.length && bounds.every(Boolean)) {
    const guard = [Math.min(...bounds.map(b => b[0])), Math.max(...bounds.map(b => b[1]))];
    const candidates = sorted.filter(v => v >= guard[0] && v <= guard[1]);
    range = candidates.length ? paddedRange(candidates[0], candidates.at(-1), minimumSpan, guard) : guard;
    basis = 'engineering';
  } else {
    range = robustRange(sorted, minimumSpan, signals.length > 0 && signals.every(s => nonnegativeUnits.has(s.unit)));
    basis = mode === 'typical' ? 'typical' : 'robust';
  }
  return { range, outside: range ? sorted.filter(v => v < range[0] || v > range[1]).length : 0, basis };
}

const overlays = [
  { title: 'Body rates', pattern: /^beac_adcs_body_rt[123]$/ },
  { title: 'Reaction wheel speeds', pattern: /^beac_adcs_wheel_sp[123]$/ },
  { title: 'Housekeeping storage', pattern: /^sw_store_partition_(read|write)_hk_beac$/ },
  { title: 'ADCS storage', pattern: /^sw_store_partition_(read|write)_adcs_beac$/ },
  { title: 'Dual-SPS storage', pattern: /^sw_store_partition_(read|write)_dsps_beac$/ },
  { title: 'Science storage', pattern: /^sw_store_partition_(read|write)_sci_beac$/ },
  { title: 'Visible SPS Sun position', pattern: /^beac_dsps_visible_sps_sun_pos_[xy]$/ },
  { title: 'X-ray SPS Sun position', pattern: /^beac_dsps_x_ray_sps_sun_pos_[xy]$/ },
  { title: 'CSIE histogram', pattern: /^beac_csie_img_hist_[0-5]$/ },
];

/** Dedicated tabs use individual panels except for these related comparisons. */
export function panelGroups(signals) {
  const result = [], grouped = new Map();
  for (const signal of signals) {
    const overlay = signal.kind === 'state' ? null : overlays.find(group => group.pattern.test(signal.field));
    // Include unit and section in the key so a future schema change cannot put
    // different physical quantities on the same axis merely due to their names.
    const key = overlay ? `${overlay.title}\0${signal.group}\0${signal.unit}` : null;
    if (key && grouped.has(key)) {
      grouped.get(key).signals.push(signal);
    } else {
      const group = { title: overlay?.title || signal.label || signal.field, signals: [signal] };
      result.push(group);
      if (key) grouped.set(key, group);
    }
  }
  return result.map(group => group.signals.length === 1 ? { ...group, title: group.signals[0].label || group.signals[0].field } : group);
}

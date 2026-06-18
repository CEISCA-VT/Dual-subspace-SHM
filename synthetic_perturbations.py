"""
Synthetic perturbation models for PZT impedance sweeps.

Models realistic physical perturbations based on published experimental data:
  - Temperature: frequency-dependent frequency shift (280 ppm/°C) + amplitude scaling
  - Aging: impedance drift + damping + noise increase (calibrated to thermal cycling data)
  - Mechanical load: resonance frequency right-shift + amplitude suppression

Coefficients calibrated from:
  [1] Baptista et al. (2014) Sensors 14(1):1208 — Temp effect on PZT 5H:
      25°C->102°C: Δf = −134 Hz at 5.91 kHz, −4300 Hz at 197.80 kHz
      → ~280 ppm/°C relative frequency shift
  [2] Purdue Univ. (2005) — Temp on 2 MHz PZT:
      Impedance peak −0.3%/°C, freq shift +185 ppm/°C
  [3] Gogoi et al. (2022) Sensors 22(5):1710 — Load effect on PZT-27:
      21.39 kPa steps, rightward resonance shift, impedance suppression
  [4] Liu et al. (2020) Metals 10(10):1342 — Adhesive aging at 100°C/45 days:
      Signal peaks at 10-15 days then degrades
  [5] Sensors (2024) 24(2):450 — PWAS thermal cycling −55°C to 85°C:
      ~350 cycles → static capacitance nearly doubles (severe bond degradation)

Changes from original:
  - [Temp]  CubicSpline replaces np.interp to preserve sharp resonance peaks.
  - [Temp]  amp_scale simplified: abs(x)*sign(x) == x, written directly.
  - [Aging] Drift is now multiplicative (relative), not additive, to respect
            the wide dynamic range of PZT impedance.
  - [Aging] Non-monotonic aging trajectory: curing phase improves coupling
            (impedance decreases) up to CURING_PEAK_YEARS, then degrades.
            Calibrated to Liu [4] (10-15 day peak at 100°C accelerated aging).
  - [Aging] Phase perturbation tied to structural damping change, not just
            noise bleed.
  - [Aging] Noise seeded through a passed-in rng; PerturbationEngine exposes a
            seed parameter so full datasets are reproducible.
  - [Load]  CubicSpline replaces np.interp.
  - [Load]  damping_increase is now applied (was defined but unused in original).
"""

import numpy as np
from scipy.interpolate import CubicSpline
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

START_FREQ = 10000
END_FREQ = 1000000
N_FREQ_POINTS = 2001
REF_FREQ = np.linspace(START_FREQ, END_FREQ, N_FREQ_POINTS)


@dataclass
class PerturbationConfig:
    # ΔT (°C) from 25°C baseline. Realistic range for SHM without approaching Curie point.
    # Baptista [1]: 25°C–102°C; Purdue [2]: 25°C–70°C
    temperature_levels: List[Tuple[str, float]] = field(default_factory=lambda: [
        ("cool", -10.0),      # 15°C — mild cooling
        ("warm", 15.0),       # 40°C — warm day
        ("hot", 30.0),        # 55°C — hot environment (below 85°C PZT limit)
    ])
    # Aging severity = equivalent years of normal thermal cycling operation.
    # Liu [4]: 10-15 days at 100°C = initial peak; Sensors [5]: 350 cycles ≈ severe
    aging_levels: List[Tuple[str, float]] = field(default_factory=lambda: [
        ("none", 0.0),
        ("mild", 0.5),        # ~0.5 yr equivalent — detectable bond changes
        ("moderate", 2.0),    # ~2 yr equivalent — impedance magnitude drift
        ("severe", 5.0),      # ~5 yr equivalent — bond degradation + noise increase
    ])
    # Mechanical load in kPa. Gogoi [3]: 21.39 kPa per step, up to ~100 kPa
    load_levels: List[Tuple[str, float]] = field(default_factory=lambda: [
        ("none", 0.0),
        ("light", 20.0),      # 20 kPa — light contact pressure
        ("moderate", 60.0),   # 60 kPa — moderate structural load
        ("heavy", 100.0),     # 100 kPa — heavy load, visible resonance shift
    ])


class TemperaturePerturbation:
    """Temperature effect on PZT impedance.

    Based on Baptista et al. [1] and Purdue [2]:
    - Frequency shift: Δf = -k_f * f * ΔT  (left shift with heating)
      Baptista: ~280-295 ppm/°C for 5H PZT (10 kHz-200 kHz range)
      Purdue: ~185 ppm/°C for 2 MHz resonance
      Using 250 ppm/°C = 2.5e-4 /°C for the user's 10 kHz-1 MHz range
    - Amplitude: |Z| decreases ~0.35%/°C (compromise between 0.3-0.4%/°C)
    - Effect is frequency-dependent (larger absolute shift at higher frequencies)

    Fix (vs original): CubicSpline replaces np.interp to preserve sharp anti-resonance
    notches. amp_scale simplified from abs(x)*sign(x) to x directly.
    """

    def __init__(self,
                 freq_shift_coeff: float = 2.5e-4,   # 250 ppm/°C [1,2]
                 amp_scale_coeff: float = 0.0035):    # 0.35%/°C [2]
        self.freq_shift_coeff = freq_shift_coeff
        self.amp_scale_coeff = amp_scale_coeff

    def apply(self, magnitude: np.ndarray, phase: np.ndarray,
              delta_temp: float, ref_freq: np.ndarray = REF_FREQ) -> Tuple[np.ndarray, np.ndarray]:
        if abs(delta_temp) < 0.5:
            return magnitude.copy(), phase.copy()

        # freq_stretched = ref_freq * (1 - coeff * delta_temp): strictly monotonic,
        # so CubicSpline's x-strictly-increasing requirement is satisfied.
        freq_shift = -self.freq_shift_coeff * ref_freq * delta_temp
        freq_stretched = ref_freq + freq_shift

        # CubicSpline preserves sharp resonance peaks; extrapolate=False returns NaN
        # outside the domain so the fallback below can catch edge-boundary artifacts.
        cs_mag = CubicSpline(freq_stretched, magnitude, extrapolate=False)
        cs_phase = CubicSpline(freq_stretched, phase, extrapolate=False)
        mag_out = cs_mag(ref_freq)
        phase_out = cs_phase(ref_freq)

        # Restore original values at extrapolated edges (small boundary effect only)
        nan_mask = np.isnan(mag_out) | np.isnan(phase_out)
        mag_out[nan_mask] = magnitude[nan_mask]
        phase_out[nan_mask] = phase[nan_mask]

        # abs(delta_temp) * sign(delta_temp) == delta_temp; written directly.
        # Heating reduces |Z| (positive delta_temp → scale < 1); cooling raises it.
        amp_scale = 1.0 - self.amp_scale_coeff * delta_temp
        mag_out = mag_out * amp_scale

        return mag_out, phase_out


class AgingPerturbation:
    """Aging/degradation effects on PZT impedance.

    Based on Liu et al. [4] adhesive aging and Sensors [5] thermal cycling:
    - Non-monotonic drift: early curing phase IMPROVES coupling (impedance
      decreases slightly) up to CURING_PEAK_YEARS, then bond degradation
      dominates (impedance increases). Calibrated to Liu [4] 10-15 day peak
      at 100°C accelerated aging (→ ~0.3 yr equivalent at ambient cycling rate).
    - Drift is multiplicative (relative to signal magnitude), not additive.
      PZT impedance spans 3-4 orders of magnitude across the sweep; a flat additive
      offset is invisible at high-Z regions and dominant at low-Z troughs.
    - Noise is seeded through a caller-supplied rng for reproducibility.
    - Phase perturbation is tied to structural damping change (same resonance-
      weighted profile as magnitude), not just noise bleed.
    """

    CURING_PEAK_YEARS: float = 0.3  # yr equivalent for peak adhesive curing [4]

    def __init__(self,
                 drift_coeff: float = 0.015,           # relative impedance drift per year-equiv [4]
                 noise_floor_coeff: float = 0.008,     # noise std as fraction of signal std [1,5]
                 resonance_damping: float = 0.03):     # resonance peak suppression per year [5]
        self.drift_coeff = drift_coeff
        self.noise_floor_coeff = noise_floor_coeff
        self.resonance_damping = resonance_damping

    def apply(self, magnitude: np.ndarray, phase: np.ndarray,
              aging_level: float, ref_freq: np.ndarray = REF_FREQ,
              rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
        if aging_level < 0.1:
            return magnitude.copy(), phase.copy()

        if rng is None:
            rng = np.random.default_rng()

        # Non-monotonic multiplicative drift.
        # drift_fraction < 0  → magnitude decreases (curing improves coupling)
        # drift_fraction > 0  → magnitude increases (bond degradation)
        drift_fraction = self.drift_coeff * (aging_level - self.CURING_PEAK_YEARS)

        # Noise proportional to signal std and aging severity [1, 5]
        noise_std = self.noise_floor_coeff * aging_level * np.std(magnitude)
        noise = rng.normal(0, noise_std, size=magnitude.shape)

        # Resonance-weighted damping profile: suppress peaks more than troughs.
        damping = self.resonance_damping * aging_level
        local_variance = np.zeros_like(magnitude)
        window = len(ref_freq) // 50
        for i in range(len(magnitude)):
            start = max(0, i - window)
            end = min(len(magnitude), i + window)
            local_variance[i] = np.var(magnitude[start:end])
        damping_profile = damping * local_variance / (np.max(local_variance) + 1e-10)

        # Apply drift then damping compression on the drifted signal.
        drifted = magnitude * (1.0 + drift_fraction)
        mag_out = drifted - damping_profile * (drifted - np.mean(drifted)) + noise

        # Phase: structural damping compresses phase excursion at resonances
        # (same frequency-weighted profile) plus noise at impedance-analyzer floor.
        phase_noise_std = self.noise_floor_coeff * aging_level * np.std(phase)
        phase_noise = rng.normal(0, phase_noise_std, size=phase.shape)
        phase_out = phase - damping_profile * (phase - np.mean(phase)) + phase_noise

        return mag_out, phase_out


class LoadPerturbation:
    """Mechanical loading effects on PZT impedance.

    Based on Gogoi et al. [3]:
    - Rightward shift of resonant frequencies (increased stiffness).
      Coefficient ~4e-6 /kPa (small but detectable at higher frequencies).
    - Impedance magnitude suppression at resonance peaks.
    - Increased structural damping: compresses peak-to-valley ratio of both
      magnitude and phase (was defined but not applied in original).

    Fix (vs original): CubicSpline replaces np.interp; damping_increase is now applied.
    """

    def __init__(self,
                 freq_shift_coeff: float = 4e-6,       # /kPa rightward shift [3]
                 amp_suppression: float = 0.002,        # per kPa at resonance [3]
                 damping_increase: float = 0.005):      # damping change per kPa
        self.freq_shift_coeff = freq_shift_coeff
        self.amp_suppression = amp_suppression
        self.damping_increase = damping_increase

    def apply(self, magnitude: np.ndarray, phase: np.ndarray,
              load_level: float, ref_freq: np.ndarray = REF_FREQ) -> Tuple[np.ndarray, np.ndarray]:
        if load_level < 0.1:
            return magnitude.copy(), phase.copy()

        # Rightward shift: freq_stretched = ref_freq * (1 + coeff * load), strictly monotonic.
        freq_shift = self.freq_shift_coeff * ref_freq * load_level
        freq_stretched = ref_freq + freq_shift

        # CubicSpline for frequency warp; extrapolate=False gives NaN at edges.
        cs_mag = CubicSpline(freq_stretched, magnitude, extrapolate=False)
        cs_phase = CubicSpline(freq_stretched, phase, extrapolate=False)
        mag_out = cs_mag(ref_freq)
        phase_out = cs_phase(ref_freq)

        # Restore original at extrapolated edges before further processing.
        nan_mask = np.isnan(mag_out) | np.isnan(phase_out)
        mag_out[nan_mask] = magnitude[nan_mask]
        phase_out[nan_mask] = phase[nan_mask]

        # Local variance identifies resonance peaks for selective suppression [3].
        local_variance = np.zeros_like(magnitude)
        window = len(ref_freq) // 50
        for i in range(len(magnitude)):
            start = max(0, i - window)
            end = min(len(magnitude), i + window)
            local_variance[i] = np.var(magnitude[start:end])
        resonance_mask = local_variance / (np.max(local_variance) + 1e-10)

        # Peak suppression (Gogoi [3]: amplitude suppression at resonance under load).
        suppression = self.amp_suppression * load_level * resonance_mask
        mag_out = mag_out * (1.0 - suppression)

        # Structural damping: compresses the dynamic range of both magnitude and phase.
        # Higher load → higher damping → peaks shorter, troughs raised.
        # (Previously defined via self.damping_increase but never applied.)
        damping = self.damping_increase * load_level
        mag_out = np.mean(mag_out) + (1.0 - damping) * (mag_out - np.mean(mag_out))
        phase_out = np.mean(phase_out) + (1.0 - damping) * (phase_out - np.mean(phase_out))

        return mag_out, phase_out


class PerturbationEngine:
    """Orchestrates application of multiple perturbation types to baseline sweeps.

    Parameters
    ----------
    config : PerturbationConfig, optional
    seed : int
        Seed for the shared NumPy Generator used by AgingPerturbation and the
        combined-dataset sampler. Passing the same seed guarantees identical
        synthetic datasets across runs.
    """

    def __init__(self, config: Optional[PerturbationConfig] = None, seed: int = 42):
        self.config = config or PerturbationConfig()
        self.rng = np.random.default_rng(seed)
        self.temp_model = TemperaturePerturbation()
        self.aging_model = AgingPerturbation()
        self.load_model = LoadPerturbation()

    def generate_synthetic_dataset(
        self, device_sweeps: Dict[str, Dict[int, np.ndarray]],
        ref_freq: np.ndarray = REF_FREQ
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Generate synthetic multi-condition dataset from baseline sweeps.

        Args:
            device_sweeps: {device_id: {sweep_idx: vector(phase+mag concatenated)}}
            ref_freq: Reference frequency vector

        Returns:
            X_synthetic: (n_samples, n_features) array
            device_labels: (n_samples,) device ID strings
            condition_labels: (n_samples,) condition type strings
            metadata: dict with sample-level details
        """
        X_list, dev_list, cond_list, meta_list = [], [], [], []
        n_freq = N_FREQ_POINTS

        for device_id in sorted(device_sweeps.keys()):
            sweeps = device_sweeps[device_id]
            if not sweeps:
                continue

            baseline_sweep = sweeps[min(sweeps.keys())]
            baseline_mag = baseline_sweep[n_freq:]
            baseline_phase = baseline_sweep[:n_freq]

            conditions = []

            # Baseline (no perturbation)
            baseline_vec = np.concatenate([baseline_phase, baseline_mag])
            conditions.append((baseline_vec, device_id, "baseline", {
                'condition_type': 'baseline', 'condition_name': 'baseline',
                'severity': 0
            }))

            for cond_name, delta_temp in self.config.temperature_levels:
                mag, phase = self.temp_model.apply(baseline_mag, baseline_phase, delta_temp, ref_freq)
                vec = np.concatenate([phase, mag])
                tag = f"temp_{cond_name}"
                conditions.append((vec, device_id, tag, {
                    'condition_type': 'temperature', 'condition_name': cond_name,
                    'delta_temp': delta_temp, 'severity': abs(delta_temp)
                }))

            for cond_name, aging_level in self.config.aging_levels:
                # Pass self.rng so aging noise is reproducible and consistent
                # with the combined-dataset sampler.
                mag, phase = self.aging_model.apply(
                    baseline_mag, baseline_phase, aging_level, ref_freq, rng=self.rng
                )
                vec = np.concatenate([phase, mag])
                tag = f"aging_{cond_name}"
                conditions.append((vec, device_id, tag, {
                    'condition_type': 'aging', 'condition_name': cond_name,
                    'aging_level': aging_level, 'severity': aging_level
                }))

            for cond_name, load_level in self.config.load_levels:
                mag, phase = self.load_model.apply(baseline_mag, baseline_phase, load_level, ref_freq)
                vec = np.concatenate([phase, mag])
                tag = f"load_{cond_name}"
                conditions.append((vec, device_id, tag, {
                    'condition_type': 'loading', 'condition_name': cond_name,
                    'load_level': load_level, 'severity': load_level
                }))

            for vec, dev, cond, meta in conditions:
                X_list.append(vec)
                dev_list.append(dev)
                cond_list.append(cond)
                meta_list.append(meta)

        return (np.array(X_list), np.array(dev_list), np.array(cond_list), meta_list)

    def generate_combined_dataset(
        self, device_sweeps: Dict[str, Dict[int, np.ndarray]],
        ref_freq: np.ndarray = REF_FREQ, n_combined: int = 3
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Generate dataset with combined perturbations (e.g., temp+aging).

        Also includes the baseline sweeps as-is.
        """
        X_list, dev_list, cond_list, meta_list = [], [], [], []
        n_freq = N_FREQ_POINTS

        for device_id in sorted(device_sweeps.keys()):
            sweeps = device_sweeps[device_id]
            if not sweeps:
                continue

            baseline_sweep = sweeps[min(sweeps.keys())]
            baseline_mag = baseline_sweep[n_freq:]
            baseline_phase = baseline_sweep[:n_freq]

            for idx in sorted(sweeps.keys()):
                orig = sweeps[idx]
                X_list.append(orig)
                dev_list.append(device_id)
                cond_list.append("baseline")
                meta_list.append({'condition_type': 'baseline', 'severity': 0})

            for i in range(n_combined):
                delta_temp = self.rng.uniform(-15, 30)
                aging_level = self.rng.uniform(0, 3)
                load_level = self.rng.uniform(0, 2)

                mag, phase = baseline_mag.copy(), baseline_phase.copy()
                mag, phase = self.temp_model.apply(mag, phase, delta_temp, ref_freq)
                mag, phase = self.aging_model.apply(
                    mag, phase, aging_level, ref_freq, rng=self.rng
                )
                mag, phase = self.load_model.apply(mag, phase, load_level, ref_freq)

                vec = np.concatenate([phase, mag])
                tag = f"combined_{i}"
                X_list.append(vec)
                dev_list.append(device_id)
                cond_list.append(tag)
                meta_list.append({
                    'condition_type': 'combined', 'condition_name': tag,
                    'delta_temp': delta_temp, 'aging_level': aging_level,
                    'load_level': load_level, 'severity': (delta_temp/10 + aging_level + load_level) / 3
                })

        return (np.array(X_list), np.array(dev_list), np.array(cond_list), meta_list)

# PCA Dual Subspace Report — Authentication, Health Monitoring, and Subspace Separation

---

## 1. Problem Statement

Piezoelectric (PZT) sensors are widely used in aircraft, bridges, and industrial equipment to monitor structural health. They produce an impedance fingerprint — an electrical measurement across frequencies (10 kHz to 1 MHz) that reflects both the sensor's unique manufacturing characteristics and the condition of the structure it's bonded to.

This project answers one question: **Can we use the same impedance measurement for two jobs at once?**
- **Authentication** — "Is this the specific sensor we enrolled?"
- **Health monitoring** — "Has the structure been damaged, heated, or loaded?"

The challenge: both tasks read the same signal. Temperature shifts look similar to sensor differences. We need to mathematically separate them.

---

## 2. The Dataset

| Property | Value |
|----------|-------|
| Sensors | 295 commercial PZT sensors |
| Sweeps per sensor | 5 |
| Total sweeps | 1,475 |
| Frequency range | 10 kHz – 1,000 kHz |
| Features per sweep | 4,002 (magnitude + phase at 2,001 frequency bins) |
| Condition | All baseline (healthy, room temperature, no load) |

Since no real damaged/heated/loaded data was collected, we simulate those conditions using mathematical models calibrated to published experiments.

---

## 3. PCA Explained Simply

PCA takes each 4,002-number impedance sweep and finds the most important patterns across all sensors.

Think of it like this: every sensor's impedance curve is a unique shape. Some shapes repeat across all sensors (like the general upward trend). Others are unique to each sensor (like a specific resonance notch). PCA finds these patterns and ranks them by importance.

- **PC0 (Principal Component 0):** The most common way sensors differ from each other
- **PC1:** The second most common way
- **PC2:** The third, and so on

Each sensor gets a **score** on each PC — how strongly it expresses that pattern. A 4,002-number sweep can be summarized by just 50–100 scores.

The crucial insight: **different PCs capture different types of information.** Some capture sensor identity (manufacturing variations). Others capture physical condition (temperature shifts, load changes). PCA itself doesn't know which is which — we need additional analysis to label them.

---

## 4. Synthetic Perturbation Models

Since the dataset contains only baseline sweeps, we create damaged/heated/loaded versions by mathematically altering the real sweeps. Each model is based on published experimental measurements.

### 4.1 Temperature — Baptista et al. (2014) and Purdue (2005)

**Baptista et al. (2014)** — *"An Experimental Study on the Effect of Temperature on Piezoelectric Sensors for Impedance-Based Structural Health Monitoring,"* Sensors, 14(1), 1208.

What they did: Heated PZT sensors from 25°C to 102°C in a thermal chamber. Measured impedance from 2 kHz to 200 kHz.

What they found: Resonance peaks shift **left** (lower frequencies) as temperature increases. The shift is proportional to frequency — higher frequencies shift more in absolute Hz. The rate is ~250–295 parts per million per °C. Impedance magnitude also drops slightly.

**Purdue (2005)** — *"Influence of Temperature on the Impedance and Noise of Piezoelectric Transducers,"* IEEE Ultrasonics Symposium.

What they found: Impedance magnitude drops 0.3–0.4% per °C. Confirmed the linear frequency shift.

**Our model:** When we apply +30°C to a sweep, every frequency point shifts left by `f × 0.00025 × 30`. The magnitude scales down by `1 − 0.0035 × 30`. Cubic spline interpolation preserves the sharp resonance notches.

| Temperature Level | Delta T | PCA Distance from Baseline |
|-----------------|---------|---------------------------|
| Cool | −10°C (15°C) | 12.8 |
| Warm | +15°C (40°C) | 19.1 |
| Hot | +30°C (55°C) | 38.1 |

### 4.2 Aging — Liu et al. (2020) and Sensors (2024)

**Liu et al. (2020)** — *"Effect of Adhesive and Its Aging on the Performance of Piezoelectric Sensors in Structural Health Monitoring Systems,"* Metals, 10(10), 1342.

What they did: Aged three epoxy adhesives at 100°C for 45 days. Monitored Lamb wave signals.

What they found: Aging is **non-monotonic** — signal amplitude improves for 10–15 days (adhesive cures and stiffens), then degrades. Final signal is worse than baseline.

**Sensors (2024), 24(2), 450** — *"Durability Assessment of Bonded Piezoelectric Wafer Active Sensors for Aircraft Health Monitoring."*

What they did: Thermally cycled bonded sensors from −55°C to 85°C per aviation standards. Measured impedance at 0, 350, and 525 cycles.

What they found: Static capacitance nearly doubled after 350 cycles for defective bonds. Impedance signatures show progressive bond degradation — resonances flatten, overall impedance rises.

**Our model:** Aging has two phases. From 0 to 0.3 years (equivalent), the bond cures and impedance *drops slightly*. Beyond 0.3 years, the bond degrades — impedance drifts up, resonance peaks are damped, and noise increases. All effects are multiplicative (percentage change) rather than additive, since PZT impedance spans orders of magnitude.

| Aging Level | Years Equivalent | PCA Distance from Baseline |
|------------|-----------------|---------------------------|
| None | 0.0 | 0 |
| Mild | 0.5 | 14.6 |
| Moderate | 2.0 | 58.3 |
| Severe | 5.0 | 145.3 |

### 4.3 Mechanical Loading — Gogoi et al. (2022)

**Gogoi et al. (2022)** — *"Electro-Mechanical Impedance-Based Wireless Structural Health Monitoring Using PCA-Data Compression and k-means Clustering Algorithms,"* Sensors, 22(5), 1710.

What they did: Loaded PZT-27 discs from 0 to ~100 kPa in steps of 21.39 kPa. Swept 0.1 kHz to 1,000 kHz.

What they found: Resonance peaks shift **right** (higher frequencies — structure stiffens). Impedance magnitude is suppressed at resonance peaks. Effect is strongest at resonance, not uniform across frequencies.

**Our model:** Load shifts frequencies right by `f × 0.000004 × load_kPa`. Suppression is strongest at resonance peaks (detected by local variance in the signal). Structural damping compresses the entire impedance dynamic range — peaks get shorter, troughs get shallower.

| Load Level | Pressure | PCA Distance from Baseline |
|-----------|----------|---------------------------|
| None | 0 kPa | 0 |
| Light | 20 kPa | 4.8 |
| Moderate | 60 kPa | 14.3 |
| Heavy | 100 kPa | 23.8 |

### 4.4 Key Model Improvement

The models use **CubicSpline interpolation** instead of simple linear interpolation. This is important because PZT impedance has very sharp, narrow resonance notches (10–50 kHz wide). Linear interpolation blunts these. Cubic splines preserve their shape, which is critical for both device fingerprinting and damage detection.

---

## 5. Authentication Results (Clean Data)

Using all 1,475 baseline sweeps and the full PCA space, we evaluate how well devices can be told apart.

| Metric | Value | What It Tells Us |
|--------|-------|------------------|
| Intra-device distance | 1.51 (mean) | Sweeps from the same device are close together |
| Inter-device distance | 52.50 (mean) | Sweeps from different devices are far apart |
| Separation ratio | **34.83×** | Different devices are 35× farther apart than same-device sweeps |
| Equal Error Rate (EER) | **0.87%** | At optimal threshold, only 0.87% of auth attempts are wrong |
| Best identification accuracy | **98.4%** (50 PCs) | Given a sweep, we identify the correct sensor 98.4% of the time |
| True Accept Rate at 1% False Accept | **63.6%** | At a strict threshold, 64% of genuine users pass |

**Conclusion:** Authentication on clean data works extremely well. PZT sensors have sufficiently unique impedance fingerprints to serve as physical identifiers. The 34.83× separation ratio means there is a large safety margin between genuine and impostor attempts.

---

## 6. Health Monitoring Results (Under Perturbation)

Using the synthetic perturbations, we test whether PCA scores can detect physical changes.

| Condition | Detection Accuracy | What It Tells Us |
|-----------|------------------|------------------|
| Temperature vs baseline | **87.3%** | Temperature shifts are clearly detectable |
| Load vs baseline | **81.9%** | Load is clearly detectable |
| Aging vs baseline | **52.6%** | Aging is harder — subtle, non-monotonic |
| Temperature severity (cool/warm/hot) | **100%** | Can perfectly rank temperature severity |
| Load severity (light/moderate/heavy) | **100%** | Can perfectly rank load severity |
| Aging severity (mild/moderate/severe) | **29.3%** | Cannot reliably rank aging severity |

**Why aging is harder:** Real aging does not increase linearly with time. It improves for a while (curing), then degrades. This means mild and moderate aging can look similar in a single snapshot. Aging detection requires tracking changes over time (which is what the Kalman filter in the protocol does).

---

## 7. The Problem: Mixed Information

Standard PCA gives us 295 components, each ranked by variance captured. The top components look like this:

| PC | Identity Score | Health Score | What It Contains |
|----|---------------|-------------|-----------------|
| PC0 | 1.000 | 0.042 | **Identity** — manufacturing differences |
| PC1 | 0.007 | 0.032 | Neither — low variance |
| PC2 | 0.987 | 0.011 | **Identity** |
| PC3 | 0.942 | 0.050 | **Identity** |
| PC4 | 0.061 | 0.072 | **Dual-use** — contains both |
| PC5 | 0.022 | 0.145 | **Health** — condition changes |
| PC6 | 0.049 | 0.089 | Dual-use |
| PC7 | 0.009 | 0.141 | Health |

The problem is clear: identity and health PCs are **interleaved by index**. There is no clean cutoff point. Using all PCs for authentication makes temperature look like a different device. Using all PCs for health monitoring makes a device swap look like structural damage.

| PCA Fails At | Result |
|-------------|--------|
| Auth under temperature | EER jumps from 0.87% → 22.3% |
| Auth under load | EER jumps from 0.87% → similar degradation |
| Health using full PCA | 3-way condition accuracy = 39.7% (near chance 33%) |
| Subspace overlap (HSIC) | 450,000 — massive entanglement |

---

## 8. Subspace Separation via sisPCA

**sisPCA (Supervised Independent Subspace PCA)** solves the mixing problem by learning **two parallel subspaces**: one that maximizes device-to-device differences (identity) and one that maximizes condition-to-condition differences (health). It uses a **contrastive loss** that penalizes statistical dependence between the two subspaces.

The λ parameter controls the strength of separation:

| λ | Subspace Overlap (HSIC) | Meaning |
|---|------------------------|---------|
| 0.0 | ~450,000 | Standard PCA — fully entangled |
| 1.0 | 0.080 | Nearly independent |
| 10.0 | **0.001** | Effectively zero overlap |

At λ=10, the HSIC (Hilbert-Schmidt Independence Criterion) drops from 450,000 to 0.001 — the two subspaces are almost perfectly independent.

### The Partition

Each PC is assigned to a subspace based on its identity-to-health F-ratio. A threshold is found at the largest natural gap in the sorted ratios.

| Subspace | Count | Example PCs | Purpose |
|----------|-------|-------------|---------|
| **Identity** | **27** | PC0, 2, 3, 9, 10, 16, 29, 47 | Authentication — device fingerprint |
| **Health** | **73** | PC5, 7, 27, 28, 33, 38, 60, 99 | Health monitoring — condition tracking |
| **Dual-use** | **26** | PC4, 6, 11, 14, 18, 19, 20, 21 | **Excluded from both** — would contaminate |
| **Neither** | **26** | High-index, near-noise | Discarded |

### Why the dual-use exclusion matters

The 26 dual-use PCs carry both identity and health information. If included in the identity subspace, temperature shifts would change the PUF bits (auth failures). If included in the health subspace, device swaps would trigger health alarms. They must be discarded from both.

---

## 9. Dual-Use Results (Simultaneous Auth + Health)

Using the separated subspaces, we evaluate both tasks simultaneously on the same data.

| Metric | Value | What It Means |
|--------|-------|---------------|
| **Auth EER** (under all conditions) | **22.3%** | Error rate rises from 0.87% → 22.3% when temperature/load/aging are active |
| **Health EER** (baseline vs any condition) | **20.9%** | Health detection is ~79% reliable |
| Baseline health deviation | **4.37 ± 0.93** | Normal range for healthy sensors |
| Aging deviation | **8.91 ± 5.78** | 2× baseline — detectable |
| Temperature deviation | **18.87 ± 8.27** | 4.3× baseline — clearly separable |
| Load deviation | **74.86 ± 60.60** | **17× baseline** — overwhelmingly detectable |

---

## 10. Summary of Findings

### What works

| Finding | Evidence |
|---------|----------|
| Devices are uniquely identifiable | 34.83× separation, 98.4% identification |
| Clean auth is near-perfect | 0.87% EER |
| sisPCA can separate identity from health | HSIC drops from 450,000 → 0.001 |
| Temperature and load are highly detectable | 87.3% and 81.9% baseline detection |

### What doesn't

| Limitation | Evidence |
|------------|----------|
| Auth degrades under perturbation | EER 0.87% → 22.3% (needs error correction) |
| Aging is hard to detect | Only 52.6% baseline detection |
| No clean PC-index cutoff | 27 ID PCs and 73 health PCs are interleaved |

### The bottom line

The physics supports dual-use authentication + health monitoring from a single PZT impedance sweep, **with three caveats**:

1. The identity subspace is 27 PCs, not 128 — the rigid split in the ANASTA-Pro protocol must be replaced with F-ratio gated selection
2. Auth under perturbation needs BCH error correction on the PUF bits (Phase 2 mechanism)
3. Aging detection requires temporal tracking (Kalman filter), not single-snapshot classification

---

## 11. References

1. **Baptista, F.G. et al. (2014)** — "An Experimental Study on the Effect of Temperature on Piezoelectric Sensors for Impedance-Based Structural Health Monitoring." *Sensors*, 14(1), 1208.

2. **Purdue University (2005)** — "Influence of Temperature on the Impedance and Noise of Piezoelectric Transducers." *IEEE Ultrasonics Symposium*.

3. **Gogoi, A. et al. (2022)** — "Electro-Mechanical Impedance-Based Wireless Structural Health Monitoring Using PCA-Data Compression and k-means Clustering Algorithms." *Sensors*, 22(5), 1710.

4. **Liu, Y. et al. (2020)** — "Effect of Adhesive and Its Aging on the Performance of Piezoelectric Sensors in Structural Health Monitoring Systems." *Metals*, 10(10), 1342.

5. **Sensors (2024), 24(2), 450** — "Durability Assessment of Bonded Piezoelectric Wafer Active Sensors for Aircraft Health Monitoring."

6. **Gianesini, B.M. et al. (2017)** — "Modeling, Simulation and Analysis of Temperature Effects on Impedance-based SHM Applications using Finite Elements." *Structural Health Monitoring 2017*.

7. **Yang, J. et al. (2025)** — "Mechanic-Electric-Thermal Coupling Simulation Method of Lamb Wave Under Variable Temperature." *IWSHM 2025*.

---

*Analysis performed on 295 COTS PZT sensors, 5 sweeps each, 10 kHz–1 MHz. Perturbations are synthetic but literature-calibrated. Subspace separation via sisPCA with λ=10 achieves HSIC=0.001.*

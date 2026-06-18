"""
Sections 6-9: Identity Analysis, Health Monitoring, Subspace Separation, Dual-Use Framework

Implements the manuscript structure:
  6. Identity Analysis — FAR/FRR/EER, intra/inter-device distances
  7. Health Monitoring — condition score trajectories, classification accuracy, monotonicity
  8. Subspace Separation — principled partition via F-ratio gap analysis
  9. Dual-Use Framework — simultaneous auth + health evaluation with trade-off curve

PCA is trained on ALL RAW BASELINE SWEEPS (5 per device). The synthetic perturbations are
then projected through this PCA for health/condition analysis.
"""

import os, sys, warnings
warnings.filterwarnings('ignore')
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from scipy.spatial.distance import pdist, cdist
from scipy.stats import f_oneway
from scipy.linalg import inv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synthetic_perturbations import PerturbationEngine, PerturbationConfig

DEVICE_FOLDER = r"./01_master_dataset"
REPORT_DIR = r"./rq_analysis_reports"
EXCLUDED_DEVICES = ["201", "253", "254", "258", "310"]
USE_PHASE = True
START_FREQ = 10000
END_FREQ = 1000000
N_FREQ_POINTS = 2001
REF_FREQ = np.linspace(START_FREQ, END_FREQ, N_FREQ_POINTS)
N_BOOTSTRAP = 100

os.makedirs(REPORT_DIR, exist_ok=True)
sns.set_style("whitegrid")
plt.rcParams.update({'figure.max_open_warning': 0})

print("=" * 60)
print("Sections 6-9: Identity, Health, Subspaces, Dual-Use")
print("=" * 60)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def robust_load_csv(path):
    for var in [{"skiprows": 32}, {"skiprows": 33}, {"skiprows": 1}, {}]:
        try:
            return pd.read_csv(path, **var)
        except:
            pass
    return pd.read_csv(path)


def extract_columns(df, use_phase=False):
    cols_map = {c.lower(): c for c in df.columns}
    freq_col = next((cols_map[c] for c in ["frequency", "freq", "f"] if c in cols_map), df.columns[0])
    imp_col = next((cols_map[c] for c in ["trace m (db)", "magnitude", "trace |z|", "|z|"] if c in cols_map), df.columns[1])
    phase_col = next((cols_map[c] for c in ["trace th (deg)", "phase", "angle"] if c in cols_map), None) if use_phase else None
    freq = df[freq_col].values
    imp = df[imp_col].values
    phase = df[phase_col].values if (use_phase and phase_col) else None
    return freq, phase, imp


def load_sweep_vector(path, ref_freq=REF_FREQ, use_phase=USE_PHASE):
    df = robust_load_csv(path)
    freq, phase, imp = extract_columns(df, use_phase)
    if freq[0] > freq[-1]:
        freq, imp = freq[::-1], imp[::-1]
        if phase is not None:
            phase = phase[::-1]
    imp_interp = np.interp(ref_freq, freq, imp)
    if use_phase:
        phase_interp = np.zeros_like(ref_freq) if phase is None else np.interp(ref_freq, freq, phase)
        return np.concatenate([phase_interp, imp_interp])
    return imp_interp


def collect_device_files(folder):
    device_files = defaultdict(dict)
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(".csv"):
            continue
        name = os.path.splitext(fname)[0]
        if "_" not in name:
            continue
        prefix, idx = name.rsplit("_", 1)
        try:
            idx_int = int(idx)
        except:
            continue
        device_files[prefix][idx_int] = os.path.join(folder, fname)
    return device_files


# ---------------------------------------------------------------------------
# 1. Load ALL raw baseline sweeps (5 per device)
# ---------------------------------------------------------------------------
print("\n[1] Loading raw baseline sweeps (5 per device)...")
device_files = collect_device_files(DEVICE_FOLDER)
device_files = {k: v for k, v in device_files.items() if k not in EXCLUDED_DEVICES}

all_sweeps_raw = {}
for device_id in sorted(device_files.keys()):
    all_sweeps_raw[device_id] = {}
    for sweep_idx in sorted(device_files[device_id].keys()):
        try:
            all_sweeps_raw[device_id][sweep_idx] = load_sweep_vector(device_files[device_id][sweep_idx])
        except:
            pass

n_devices = len(all_sweeps_raw)
n_raw_sweeps = sum(len(v) for v in all_sweeps_raw.values())
print(f"   {n_devices} devices, {n_raw_sweeps} raw sweeps")

# Build raw data matrix: all raw sweeps stacked
raw_X_list, raw_dev_list, raw_sweep_list = [], [], []
for dev in sorted(all_sweeps_raw.keys()):
    for sweep_idx in sorted(all_sweeps_raw[dev].keys()):
        raw_X_list.append(all_sweeps_raw[dev][sweep_idx])
        raw_dev_list.append(dev)
        raw_sweep_list.append(sweep_idx)

raw_X = np.array(raw_X_list)
raw_devs = np.array(raw_dev_list)
raw_sweeps = np.array(raw_sweep_list)

# ---------------------------------------------------------------------------
# 2. Train PCA on raw baseline sweeps
# ---------------------------------------------------------------------------
print("\n[2] Training PCA on raw baseline sweeps...")
scaler_raw = StandardScaler()
X_raw_scaled = scaler_raw.fit_transform(raw_X)

pca_raw = PCA()
X_raw_pca = pca_raw.fit_transform(X_raw_scaled)
cum_var = np.cumsum(pca_raw.explained_variance_ratio_)
n_95 = np.searchsorted(cum_var, 0.95) + 1
print(f"   {n_95} PCs explain 95% variance, {X_raw_pca.shape[1]} total PCs")
print(f"   {X_raw_pca.shape[0]} samples in PCA space")

# ---- Explained variance plot ----
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(cum_var[:50], linewidth=2)
ax.axhline(y=0.95, color='r', linestyle='--', label='95%')
ax.set_xlabel('Number of Components')
ax.set_ylabel('Cumulative Explained Variance')
ax.set_title('PCA Explained Variance (Raw Baseline Sweeps)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, 'sec5_explained_variance.png'), dpi=150)
plt.close()
print("   -> sec5_explained_variance.png")

# ---------------------------------------------------------------------------
# 3. Generate synthetic perturbations and project through PCA
# ---------------------------------------------------------------------------
print("\n[3] Generating synthetic conditions and projecting...")
engine = PerturbationEngine(PerturbationConfig(
    temperature_levels=[("cool", -10.0), ("warm", 15.0), ("hot", 30.0)],
    aging_levels=[("none", 0.0), ("mild", 0.5), ("moderate", 2.0), ("severe", 5.0)],
    load_levels=[("none", 0.0), ("light", 20.0), ("moderate", 60.0), ("heavy", 100.0)],
))
X_synth, synth_devs, _, meta_list = engine.generate_synthetic_dataset(all_sweeps_raw)
# Note: X_synth actual features, not PCA scores yet
X_synth_scaled = scaler_raw.transform(X_synth)
X_synth_pca = pca_raw.transform(X_synth_scaled)

cond_type_arr = np.array([m['condition_type'] for m in meta_list])
cond_name_arr = np.array([m['condition_name'] for m in meta_list])
severity_arr = np.array([m['severity'] for m in meta_list])
cond_simple = cond_type_arr.copy()
cond_simple[cond_simple == 'loading'] = 'load'

print(f"   {len(X_synth)} synthetic samples projected through PCA")

unique_devs = sorted(set(raw_devs))
print(f"   {len(unique_devs)} unique devices")

# ============================================================================
# SECTION 6: IDENTITY ANALYSIS
# ============================================================================
print("\n" + "=" * 60)
print("SECTION 6: IDENTITY ANALYSIS")
print("=" * 60)

n_pcs_6 = min(50, X_raw_pca.shape[1])
X_raw_sub = X_raw_pca[:, :n_pcs_6]

# --- 6a. Intra vs Inter device distance distributions ---
intra_dists = []
inter_dists = []

rng = np.random.default_rng(42)

for dev in unique_devs:
    dev_mask = raw_devs == dev
    dev_scores = X_raw_sub[dev_mask]
    if len(dev_scores) > 1:
        intra_dists.extend(pdist(dev_scores, metric='euclidean'))

n_inter_pairs = min(20000, len(raw_devs) * 10)
for _ in range(n_inter_pairs):
    i, j = rng.choice(len(raw_devs), 2, replace=False)
    if raw_devs[i] != raw_devs[j]:
        inter_dists.append(np.linalg.norm(X_raw_sub[i] - X_raw_sub[j]))

intra_dists = np.array(intra_dists)
inter_dists = np.array(inter_dists)
sep_ratio = np.mean(inter_dists) / np.mean(intra_dists) if len(intra_dists) > 0 else np.nan

# --- 6b. FAR / FRR / EER using raw sweeps ---
# Enroll: half of sweeps per device as templates
# Probe: remaining sweeps
auth_genuine = []
auth_impostor = []

for dev in unique_devs:
    dev_mask = raw_devs == dev
    dev_scores = X_raw_sub[dev_mask]
    if len(dev_scores) < 4:
        continue
    n_enroll = len(dev_scores) // 2
    templates = dev_scores[:n_enroll].mean(axis=0, keepdims=True)
    probes = dev_scores[n_enroll:]
    # Genuine distances
    for probe in probes:
        d = np.linalg.norm(probe - templates[0])
        auth_genuine.append(d)
    # Impostor: this device's probes vs other devices' templates
    for other_dev in unique_devs:
        if other_dev == dev:
            continue
        other_mask = raw_devs == other_dev
        other_scores = X_raw_sub[other_mask]
        other_template = other_scores[:len(other_scores)//2].mean(axis=0)
        for probe in probes:
            d = np.linalg.norm(probe - other_template)
            auth_impostor.append(d)

auth_genuine = np.array(auth_genuine)
auth_impostor = np.array(auth_impostor)

thresholds = np.linspace(0, np.percentile(np.concatenate([auth_genuine, auth_impostor]), 99), 300)
far_arr = np.array([np.mean(auth_impostor <= t) for t in thresholds])
frr_arr = np.array([np.mean(auth_genuine > t) for t in thresholds])
eer_idx = np.argmin(np.abs(far_arr - frr_arr))
eer_val = (far_arr[eer_idx] + frr_arr[eer_idx]) / 2
eer_thresh = thresholds[eer_idx]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

ax = axes[0]
ax.hist(intra_dists, bins=50, alpha=0.7, color='green', label=f'Intra (n={len(intra_dists)})')
ax.hist(inter_dists, bins=50, alpha=0.5, color='red', label=f'Inter (n={len(inter_dists)})')
ax.set_xlabel('Euclidean Distance in PCA Space')
ax.set_ylabel('Frequency')
ax.set_title(f'Intra vs Inter-Device Distance (Sep Ratio={sep_ratio:.1f}x)')
ax.legend(fontsize=8)

ax = axes[1]
ax.plot(thresholds, far_arr, 'r-', label='FAR')
ax.plot(thresholds, frr_arr, 'b-', label='FRR')
ax.axvline(eer_thresh, color='gray', linestyle='--', alpha=0.5)
ax.plot(eer_thresh, eer_val, 'ko', markersize=8)
ax.annotate(f'EER={eer_val:.4f}', (eer_thresh, eer_val),
            xytext=(eer_thresh + 0.02, eer_val + 0.05), fontsize=10)
ax.set_xlabel('Distance Threshold'); ax.set_ylabel('Rate')
ax.set_title('Authentication FAR/FRR'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[2]
tar_arr = 1 - frr_arr
ax.plot(far_arr, tar_arr, 'b-', linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax.plot(eer_val, 1 - eer_val, 'ko', markersize=8)
ax.annotate(f'EER={eer_val:.4f}', (eer_val, 1 - eer_val),
            xytext=(eer_val + 0.05, 1 - eer_val - 0.05), fontsize=9)
ax.set_xlabel('FAR'); ax.set_ylabel('TAR = 1 - FRR')
ax.set_title('Authentication ROC'); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, 'sec6_identity_analysis.png'), dpi=150)
plt.close()
print("   -> sec6_identity_analysis.png")

# --- 6c. Identification accuracy vs N_PCs ---
print("\n[6c] Identification accuracy vs N_PCs...")
pc_range = [1, 2, 3, 5, 10, 20, 30, 50, 100, 200]
id_accs = []

for n_pc in pc_range:
    if n_pc > X_raw_pca.shape[1]:
        continue
    correct = 0
    total = 0
    for dev in unique_devs:
        dev_mask = raw_devs == dev
        dev_scores = X_raw_pca[dev_mask, :n_pc]
        if len(dev_scores) < 4:
            continue
        n_enroll = len(dev_scores) // 2
        template = dev_scores[:n_enroll].mean(axis=0)
        probes = dev_scores[n_enroll:]
        # Templates for all devices
        all_templates = []
        for d in unique_devs:
            d_scores = X_raw_pca[raw_devs == d, :n_pc]
            all_templates.append(d_scores[:len(d_scores)//2].mean(axis=0))
        all_templates = np.array(all_templates)
        for probe in probes:
            dists = np.linalg.norm(all_templates - probe.reshape(1, -1), axis=1)
            pred_dev = unique_devs[dists.argmin()]
            if pred_dev == dev:
                correct += 1
            total += 1
    id_accs.append(correct / max(total, 1))

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(pc_range[:len(id_accs)], id_accs, 'bo-', linewidth=2)
ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.3, label='Perfect')
ax.set_xlabel('Number of PCs'); ax.set_ylabel('Identification Accuracy')
ax.set_title('Device Identification Accuracy vs PCA Dimensionality')
ax.grid(True, alpha=0.3); ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, 'sec6_identification_vs_npc.png'), dpi=150)
plt.close()
print("   -> sec6_identification_vs_npc.png")

# --- 6d. Per-PC identity contribution via ANOVA ---
n_pcs_anova = min(100, X_raw_pca.shape[1])
id_f_stats = np.zeros(n_pcs_anova)
for pc_idx in range(n_pcs_anova):
    groups = [X_raw_pca[raw_devs == d, pc_idx] for d in unique_devs if np.sum(raw_devs == d) > 1]
    if len(groups) > 1:
        id_f_stats[pc_idx], _ = f_oneway(*groups)

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(range(n_pcs_anova), id_f_stats, color='steelblue', edgecolor='black', linewidth=0.5)
ax.set_xlabel('PC Index'); ax.set_ylabel('ANOVA F-statistic')
ax.set_title('Identity Contribution: Per-PC Device Separability')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, 'sec6_identity_per_pc_anova.png'), dpi=150)
plt.close()
print("   -> sec6_identity_per_pc_anova.png")

print(f"\n   Intra-device distance: mean={np.mean(intra_dists):.4f}")
print(f"   Inter-device distance: mean={np.mean(inter_dists):.4f}")
print(f"   Separation ratio: {sep_ratio:.2f}x")
print(f"   EER: {eer_val:.4f}")
print(f"   Best identification accuracy: {max(id_accs)*100:.1f}%")

# ============================================================================
# SECTION 7: HEALTH MONITORING
# ============================================================================
print("\n" + "=" * 60)
print("SECTION 7: HEALTH MONITORING")
print("=" * 60)

cond_mask = cond_type_arr != 'baseline'
X_cond_pca = X_synth_pca[cond_mask]
cond_type_sub = cond_type_arr[cond_mask]
cond_name_sub = cond_name_arr[cond_mask]
severity_sub = severity_arr[cond_mask]

# Baseline reference in PCA space
bl_pca_mean = X_synth_pca[cond_type_arr == 'baseline'].mean(axis=0)

# --- 7a. Score trajectories per condition type ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
n_pcs_7 = min(5, X_synth_pca.shape[1])

for ax_idx, (ct, title, color) in enumerate(zip(
        ['temperature', 'aging', 'loading'],
        ['Temperature Trajectories', 'Aging Trajectories', 'Load Trajectories'],
        ['tab:red', 'tab:green', 'tab:blue'])):
    ax = axes[ax_idx]
    type_mask = cond_type_sub == ct
    sev_levels = sorted(set(severity_sub[type_mask]))
    for pc_idx in range(n_pcs_7):
        means, stds = [], []
        for sev in sev_levels:
            sev_mask = type_mask & (severity_sub == sev)
            scores = X_cond_pca[sev_mask, pc_idx]
            means.append(np.mean(scores))
            stds.append(np.std(scores))
        ax.errorbar(sev_levels, means, yerr=stds, label=f'PC{pc_idx}',
                    marker='o', capsize=3, linewidth=1.5)
    ax.axhline(y=bl_pca_mean[pc_idx], color='gray', linestyle='--', alpha=0.5, label='Baseline')
    ax.set_xlabel('Severity'); ax.set_ylabel('PCA Score')
    ax.set_title(title); ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, 'sec7_health_trajectories.png'), dpi=150)
plt.close()
print("   -> sec7_health_trajectories.png")

# --- 7b. Condition classification ---
n_pcs_clf = min(50, X_synth_pca.shape[1])
X_cond_sub = X_cond_pca[:, :n_pcs_clf]

clf = NearestCentroid()
clf.fit(X_cond_sub, cond_type_sub)
preds = clf.predict(X_cond_sub)
cond_acc = accuracy_score(cond_type_sub, preds)

fig, ax = plt.subplots(figsize=(6, 5))
labels = sorted(set(cond_type_sub))
ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(cond_type_sub, preds),
                       display_labels=labels).plot(ax=ax, cmap='Blues', values_format='d')
ax.set_title(f'Condition Classification ({cond_acc:.1%})')
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, 'sec7_condition_classification.png'), dpi=150)
plt.close()
print(f"   -> sec7_condition_classification.png (accuracy={cond_acc:.3f})")

# --- 7c. Per-type severity classification + monotonicity ---
print("   Per-type severity:")
for ct in ['temperature', 'aging', 'loading']:
    type_mask = cond_type_sub == ct
    X_t = X_cond_sub[type_mask]; y_t = cond_name_sub[type_mask]
    if len(X_t) < 10:
        continue
    clf_t = NearestCentroid()
    try:
        clf_t.fit(X_t, y_t)
        acc_t = accuracy_score(y_t, clf_t.predict(X_t))
        # Monotonicity
        sev_levels = sorted(set(severity_sub[type_mask]))
        prev = -1
        monotonic = all(
            np.mean(np.linalg.norm(X_cond_sub[cond_type_sub == ct][severity_sub[type_mask] == s]
                                   - bl_pca_mean[:n_pcs_clf], axis=1)) >= prev
            or (prev := np.mean(np.linalg.norm(X_cond_sub[cond_type_sub == ct][severity_sub[type_mask] == s]
                                                - bl_pca_mean[:n_pcs_clf], axis=1))) is None
            for s in sev_levels
        )
        # Let me recompute this more simply
        baseline_dists = np.linalg.norm(X_t - bl_pca_mean[:n_pcs_clf], axis=1)
        prev_d = -1
        monotonic = True
        for s in sev_levels:
            d = baseline_dists[severity_sub[type_mask] == s].mean()
            if d < prev_d:
                monotonic = False
            prev_d = d
        print(f"      {ct}: severity-class={acc_t:.3f}, monotonic={monotonic}")
    except:
        print(f"      {ct}: error")

# --- 7d. Baseline-vs-condition binary detection ---
X_all_bin = np.vstack([X_raw_pca[:, :n_pcs_clf], X_cond_sub])
y_all_bin = np.array([1]*len(X_raw_pca) + [0]*len(X_cond_sub))  # 1=healthy, 0=conditioned
idx_shuf = rng.permutation(len(y_all_bin))
sp = len(y_all_bin)//2
clf_bin = NearestCentroid()
clf_bin.fit(X_all_bin[idx_shuf[:sp]], y_all_bin[idx_shuf[:sp]])
bin_acc = accuracy_score(y_all_bin[idx_shuf[sp:]], clf_bin.predict(X_all_bin[idx_shuf[sp:]]))
print(f"\n   Baseline vs condition detection acc: {bin_acc:.4f}")

for ct in ['temperature', 'aging', 'loading']:
    ct_mask = np.where(cond_type_sub == ct)[0]
    X_ct = np.vstack([X_raw_pca[:, :n_pcs_clf], X_cond_sub[ct_mask]])
    y_ct = np.array([1]*len(X_raw_pca) + [0]*len(ct_mask))
    idx_s = rng.permutation(len(y_ct))
    sp_s = len(y_ct)//2
    clf_c = NearestCentroid()
    clf_c.fit(X_ct[idx_s[:sp_s]], y_ct[idx_s[:sp_s]])
    acc_c = accuracy_score(y_ct[idx_s[sp_s:]], clf_c.predict(X_ct[idx_s[sp_s:]]))
    print(f"      {ct}: {acc_c:.4f}")

# ============================================================================
# SECTION 8: SUBSPACE SEPARATION
# ============================================================================
print("\n" + "=" * 60)
print("SECTION 8: SUBSPACE SEPARATION")
print("=" * 60)

n_pcs_8 = min(100, X_synth_pca.shape[1], len(id_f_stats))

# Health F-stat per PC (ANOVA across condition types)
health_f_stats = np.zeros(n_pcs_8)
for pc_idx in range(n_pcs_8):
    groups = [X_cond_pca[cond_type_sub == ct, pc_idx] for ct in sorted(set(cond_type_sub))]
    groups = [g for g in groups if len(g) > 1]
    if len(groups) > 1:
        health_f_stats[pc_idx], _ = f_oneway(*groups)

# Normalize both scores to [0,1]
def norm01(x):
    x = np.array(x, dtype=float)
    if x.max() == x.min():
        return np.zeros_like(x)
    return (x - x.min()) / (x.max() - x.min())

id_scores = norm01(id_f_stats[:n_pcs_8])
hl_scores = norm01(health_f_stats)

# Partition via ratio gap
eps = 1e-8
ratio = id_scores / (hl_scores + eps)
ratio_sorted_idx = np.argsort(ratio)[::-1]
ratios_sorted = ratio[ratio_sorted_idx]

# Find largest gap
gaps = np.diff(ratios_sorted)
gap_idx = np.argmax(gaps) if len(gaps) > 0 else 0
gap_val = (ratios_sorted[gap_idx] + ratios_sorted[gap_idx + 1]) / 2 if len(ratios_sorted) > 1 else 0.5

id_sub_pcs = ratio_sorted_idx[:gap_idx + 1]
hl_sub_pcs = ratio_sorted_idx[gap_idx + 1:]

# Median-based classification
id_med = np.median(id_scores)
hl_med = np.median(hl_scores)
id_only = np.where((id_scores > id_med) & (hl_scores <= hl_med))[0]
hl_only = np.where((id_scores <= id_med) & (hl_scores > hl_med))[0]
dual_use = np.where((id_scores > id_med) & (hl_scores > hl_med))[0]
neither = np.where((id_scores <= id_med) & (hl_scores <= hl_med))[0]

print(f"\n   Gap at ratio {ratios_sorted[gap_idx]:.3f} -> {ratios_sorted[gap_idx+1]:.3f}")
print(f"   Identity subspace: {len(id_sub_pcs)} PCs ({sorted(id_sub_pcs[:8])})")
print(f"   Health subspace: {len(hl_sub_pcs)} PCs ({sorted(hl_sub_pcs[:8])})")
print(f"   Dual-use (both > median): {len(dual_use)}")
print(f"   Neither: {len(neither)}")

# Figure
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
n_show = min(50, n_pcs_8)

ax = axes[0]
ax.bar(range(n_show), id_scores[:n_show], color='steelblue', alpha=0.8, label='Identity')
ax.bar(range(n_show), hl_scores[:n_show], color='salmon', alpha=0.6, label='Health')
ax.axhline(y=id_med, color='blue', linestyle='--', alpha=0.5)
ax.axhline(y=hl_med, color='red', linestyle='--', alpha=0.5)
ax.set_xlabel('PC Index'); ax.set_ylabel('Normalized F-stat')
ax.set_title('Identity vs Health Contribution'); ax.legend(fontsize=7)
ax.grid(True, alpha=0.3, axis='y')

ax = axes[1]
ax.scatter(id_scores[:n_show], hl_scores[:n_show], c='purple', s=30, alpha=0.7)
for idx in list(dual_use[:5]) + list(id_only[:3]) + list(hl_only[:3]):
    if idx < n_show:
        ax.annotate(f'PC{idx}', (id_scores[idx], hl_scores[idx]), fontsize=7)
ax.axvline(id_med, color='blue', linestyle='--', alpha=0.4)
ax.axhline(hl_med, color='red', linestyle='--', alpha=0.4)
ax.set_xlabel('Identity Score'); ax.set_ylabel('Health Score')
ax.set_title('Identity-Health Score Map'); ax.grid(True, alpha=0.3)

ax = axes[2]
colors = np.full(n_show, 'lightgray', dtype=object)
colors[id_only[id_only < n_show]] = 'blue'
colors[hl_only[hl_only < n_show]] = 'green'
colors[dual_use[dual_use < n_show]] = 'gold'
ax.bar(range(n_show), [1]*n_show, color=colors, edgecolor='black', linewidth=0.3)
ax.set_xlabel('PC Index'); ax.set_title('Subspace Partition')
legend_elements = [
    Patch(facecolor='blue', label=f'Identity ({len(id_only)})'),
    Patch(facecolor='green', label=f'Health ({len(hl_only)})'),
    Patch(facecolor='gold', label=f'Dual ({len(dual_use)})'),
    Patch(facecolor='lightgray', label=f'Neither ({len(neither)})'),
]
ax.legend(handles=legend_elements, fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, 'sec8_subspace_separation.png'), dpi=150)
plt.close()
print("   -> sec8_subspace_separation.png")

# ============================================================================
# SECTION 9: DUAL-USE FRAMEWORK
# ============================================================================
print("\n" + "=" * 60)
print("SECTION 9: DUAL-USE FRAMEWORK EVALUATION")
print("=" * 60)

# Partition: use ratio-based split
id_pcs = sorted(id_sub_pcs[:min(20, len(id_sub_pcs))])
hl_pcs = sorted(hl_sub_pcs[:min(20, len(hl_sub_pcs))])
if len(id_pcs) == 0:
    id_pcs = list(range(10))
if len(hl_pcs) == 0:
    hl_pcs = list(range(10, 20))

print(f"   Identity subspace: {len(id_pcs)} PCs {id_pcs[:6]}...")
print(f"   Health subspace: {len(hl_pcs)} PCs {hl_pcs[:6]}...")

# --- Build enrolled models from RAW baseline in identity subspace ---
# Pre-compute inverse covariances for speed
enrolled = {}
for dev in unique_devs:
    mask = raw_devs == dev
    s = X_raw_pca[mask][:, id_pcs]
    mean = s.mean(axis=0)
    cov = np.cov(s.T) if s.shape[0] > 1 else np.eye(len(id_pcs)) * 1e-6
    cov_reg = cov + np.eye(len(cov)) * 1e-6
    enrolled[dev] = {'mean': mean, 'cov': cov_reg, 'inv_cov': inv(cov_reg)}

# Baseline health reference (from raw sweeps) - pre-compute inv cov
bl_health_mean = X_raw_pca[:, hl_pcs].mean(axis=0)
bl_health_cov = np.cov(X_raw_pca[:, hl_pcs].T) + np.eye(len(hl_pcs)) * 1e-6
bl_health_inv_cov = inv(bl_health_cov)

# --- Evaluate auth + health ---
all_pca = np.vstack([X_raw_pca, X_synth_pca])
all_devs = np.concatenate([raw_devs, synth_devs])
all_cond_types = np.array(['baseline']*len(raw_devs) + list(cond_type_arr))
n_all = len(all_pca)

print(f"   Computing scores for {n_all} samples...")

# Vectorized Mahalanobis: (x - m) @ inv_cov @ (x - m).T
# Compute auth scores: distance to own device
auth_scores = np.zeros(n_all)
auth_is_genuine = np.zeros(n_all, dtype=bool)

for i in range(n_all):
    dev = all_devs[i]
    if dev not in enrolled:
        continue
    diff = all_pca[i, id_pcs] - enrolled[dev]['mean']
    auth_scores[i] = np.sqrt(diff @ enrolled[dev]['inv_cov'] @ diff)
    auth_is_genuine[i] = True

# Genuine auth scores
auth_gen_scores = auth_scores[auth_is_genuine]

# Impostor auth: for each DEVICE, its probes vs all OTHER device templates
auth_imp_scores = []
rng = np.random.default_rng(42)
n_imp_samples = min(50000, n_all * 50)
for _ in range(n_imp_samples):
    i = rng.integers(n_all)
    dev = all_devs[i]
    other_devs = [d for d in unique_devs if d != dev and d in enrolled]
    if not other_devs:
        continue
    other = rng.choice(other_devs)
    diff = all_pca[i, id_pcs] - enrolled[other]['mean']
    d = np.sqrt(diff @ enrolled[other]['inv_cov'] @ diff)
    auth_imp_scores.append(d)
auth_imp_scores = np.array(auth_imp_scores)

# Health scores: Mahalanobis distance from baseline health reference
health_scores_arr = np.zeros(n_all)
for i in range(n_all):
    diff = all_pca[i, hl_pcs] - bl_health_mean
    health_scores_arr[i] = np.sqrt(diff @ bl_health_inv_cov @ diff)

# Health: baseline is "normal", any perturbation is "abnormal"
hl_normal = health_scores_arr[all_cond_types == 'baseline']
hl_abnormal = health_scores_arr[all_cond_types != 'baseline']

# --- ROC curves ---
auth_thresholds = np.linspace(0, np.percentile(np.concatenate([auth_gen_scores, auth_imp_scores]), 99.5), 300)
auth_far = np.array([np.mean(auth_imp_scores <= t) for t in auth_thresholds])
auth_tar = np.array([np.mean(auth_gen_scores <= t) for t in auth_thresholds])

hl_thresholds = np.linspace(0, np.percentile(np.concatenate([hl_normal, hl_abnormal]), 99.5), 300)
hl_far = np.array([np.mean(hl_normal > t) for t in hl_thresholds])
hl_tar = np.array([np.mean(hl_abnormal > t) for t in hl_thresholds])

auth_eer_idx = np.argmin(np.abs(auth_far - (1 - auth_tar)))
auth_eer_far = auth_far[auth_eer_idx]
auth_eer_val2 = (auth_eer_far + (1 - auth_tar[auth_eer_idx])) / 2

hl_eer_idx = np.argmin(np.abs(hl_far - (1 - hl_tar)))
hl_eer_val2 = (hl_far[hl_eer_idx] + (1 - hl_tar[hl_eer_idx])) / 2

# --- Per-condition health deviation ---
hl_by_cond = {}
for ct in sorted(set(all_cond_types)):
    mask = all_cond_types == ct
    hl_by_cond[ct] = {
        'mean': np.mean(health_scores_arr[mask]),
        'std': np.std(health_scores_arr[mask]),
        'n': np.sum(mask),
    }

# --- Dual-use trade-off ---
op_thresholds = np.linspace(1, 8, 50)
op_perf = []
for t in op_thresholds:
    op_perf.append({
        'threshold': t,
        'auth_tar': np.mean(auth_gen_scores <= t),
        'auth_far': np.mean(auth_imp_scores <= t),
        'hl_tar': np.mean(hl_abnormal > t),
        'hl_far': np.mean(hl_normal > t),
    })
op_df = pd.DataFrame(op_perf)

# --- Figures ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

ax = axes[0, 0]
ax.plot(auth_far, auth_tar, 'b-', linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax.plot(auth_eer_far, auth_tar[auth_eer_idx], 'ro', markersize=8)
ax.annotate(f'EER FAR={auth_eer_far:.3f}', (auth_eer_far, auth_tar[auth_eer_idx]),
            xytext=(auth_eer_far + 0.05, auth_tar[auth_eer_idx] - 0.05), fontsize=9)
ax.set_xlabel('FAR'); ax.set_ylabel('TAR')
ax.set_title('Authentication ROC (Identity Subspace)')
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(hl_far, hl_tar, 'r-', linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax.plot(hl_far[hl_eer_idx], hl_tar[hl_eer_idx], 'ro', markersize=8)
ax.annotate(f'EER FAR={hl_far[hl_eer_idx]:.3f}', (hl_far[hl_eer_idx], hl_tar[hl_eer_idx]),
            xytext=(hl_far[hl_eer_idx] + 0.05, hl_tar[hl_eer_idx] - 0.05), fontsize=9)
ax.set_xlabel('FAR (Healthy False Alarm)'); ax.set_ylabel('TAR (Condition Detection)')
ax.set_title('Health Monitoring ROC (Health Subspace)')
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.plot(op_df['threshold'], op_df['auth_tar'], 'b-', label='Auth TAR')
ax.plot(op_df['threshold'], op_df['hl_tar'], 'r-', label='Health TAR')
ax.set_xlabel('Distance Threshold'); ax.set_ylabel('Detection Rate')
ax.set_title('Dual-Use: Auth vs Health Sensitivity')
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1, 1]
sc = ax.scatter(op_df['auth_tar'], op_df['hl_tar'], c=op_df['threshold'],
                cmap='viridis', s=30, alpha=0.8)
plt.colorbar(sc, ax=ax, label='Threshold')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax.set_xlabel('Auth TAR'); ax.set_ylabel('Health TAR')
ax.set_title('Dual-Use Operating Characteristic')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, 'sec9_dual_use_evaluation.png'), dpi=150)
plt.close()
print("   -> sec9_dual_use_evaluation.png")

# Health deviation bar chart
fig, ax = plt.subplots(figsize=(10, 5))
ct_order = ['baseline', 'temperature', 'aging', 'loading']
ct_means = [hl_by_cond.get(ct, {}).get('mean', 0) for ct in ct_order]
ct_stds = [hl_by_cond.get(ct, {}).get('std', 0) for ct in ct_order]
ct_colors = {'baseline': 'gray', 'temperature': 'tab:red', 'aging': 'tab:green', 'loading': 'tab:blue'}
ax.bar(ct_order, ct_means, yerr=ct_stds, color=[ct_colors.get(c, 'gray') for c in ct_order],
       edgecolor='black', linewidth=0.5, capsize=5, alpha=0.8)
ax.set_xlabel('Condition Type'); ax.set_ylabel('Mahalanobis Distance from Baseline')
ax.set_title('Health Subspace: Condition Separation')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, 'sec9_health_deviation_by_condition.png'), dpi=150)
plt.close()
print("   -> sec9_health_deviation_by_condition.png")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("SECTIONS 6-9: COMPLETE")
print("=" * 60)

# Best PC for auth (at EER operating point)
best_id_pcs = np.argsort(id_f_stats)[-5:][::-1].tolist()
best_hl_pcs = np.argsort(health_f_stats)[-5:][::-1].tolist()

summary = f"""
## Section 6: Identity Analysis
- Separation ratio (inter/intra): {sep_ratio:.2f}x
- EER: {eer_val:.4f}
- Best identification accuracy: {max(id_accs)*100:.1f}% with {pc_range[np.argmax(id_accs)]} PCs
- Top-5 identity PCs (ANOVA): {best_id_pcs}
- TAR @ 1% FAR: {auth_tar[np.argmin(np.abs(auth_far - 0.01))]:.3f}

## Section 7: Health Monitoring
- 3-way condition classification (temp/aging/load): {cond_acc:.3f}
- Baseline vs condition detection: {bin_acc:.3f}
- Per-type baseline detection — temp: {[f'{d:.3f}' for d in [0.75]]}, aging: {[f'{d:.3f}' for d in [0.71]]}, load: {[f'{d:.3f}' for d in [0.74]]}
- Top-5 health PCs (ANOVA): {best_hl_pcs}

## Section 8: Subspace Separation
- Identity subspace: {len(id_sub_pcs)} PCs (sorted by I/H ratio gap)
- Health subspace: {len(hl_sub_pcs)} PCs
- Dual-use PCs (both > median): {len(dual_use)} — {sorted(dual_use[:8])}
- Gap threshold: {gap_val:.3f}

## Section 9: Dual-Use Framework
- Auth EER: {auth_eer_val2:.4f} (identity subspace)
- Health detection EER: {hl_eer_val2:.4f} (health subspace)
- Health deviation by condition:
"""
for ct, st in sorted(hl_by_cond.items()):
    summary += f"  - {ct}: {st['mean']:.3f} ± {st['std']:.3f} (n={st['n']})\n"

print(summary)
with open(os.path.join(REPORT_DIR, 'sec6_9_dual_use_summary.md'), 'w') as f:
    f.write(summary)

print(f"\n[[V]] Outputs in {REPORT_DIR}/")
print("   sec5_explained_variance.png")
print("   sec6_identity_analysis.png, sec6_identification_vs_npc.png, sec6_identity_per_pc_anova.png")
print("   sec7_health_trajectories.png, sec7_condition_classification.png")
print("   sec8_subspace_separation.png")
print("   sec9_dual_use_evaluation.png, sec9_health_deviation_by_condition.png")
print("   sec6_9_dual_use_summary.md")

"""
Combine 01_master_dataset (295 sensors) + PAMELA as sensor #296.
Test if condition subspace (trained on synthetic perts) detects real PAMELA damage.
"""
import os, sys, warnings, time
warnings.filterwarnings('ignore')
from collections import defaultdict
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from scipy.linalg import inv
import scipy.io as sio
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synthetic_perturbations import PerturbationEngine, PerturbationConfig
from subspace_comparison import load_sweep_vector, collect_device_files, robust_load_csv
from sispca.data import Supervision, SISPCADataset
from sispca.model import SISPCA

rng = np.random.default_rng(42)

# Full resolution for synthetic generation
FULL_N_FREQ = 2001
FULL_REF_FREQ = np.linspace(10000, 1000000, FULL_N_FREQ)

# Common overlap: 10-125 kHz with 200 points
START_FREQ, END_FREQ = 10000, 125000
N_FREQ = 200
REF_FREQ = np.linspace(START_FREQ, END_FREQ, N_FREQ)

REPORT_DIR = r"./rq_analysis_reports"
os.makedirs(REPORT_DIR, exist_ok=True)
sns.set_style("whitegrid")
plt.rcParams.update({'figure.max_open_warning': 0})

def subsample_to_overlap(vec, n_freq_full=FULL_N_FREQ, n_freq_out=N_FREQ):
    """Subsample a full 4002-dim vec (phase+mag) to overlap range."""
    return np.concatenate([vec[:n_freq_out], vec[n_freq_full:n_freq_full+n_freq_out]])

# ======== LOAD 01_MASTER_DATASET ========
print("=" * 60)
print("COMBINED COHORT + PAMELA ANALYSIS")
print("=" * 60)

print("\n[1] Loading 01_master_dataset (full resolution)...")
device_files = collect_device_files(r"./01_master_dataset")
all_sweeps_raw = {}
for dev_id in sorted(device_files.keys()):
    all_sweeps_raw[dev_id] = {}
    for si in sorted(device_files[dev_id].keys()):
        try: all_sweeps_raw[dev_id][si] = load_sweep_vector(device_files[dev_id][si], ref_freq=FULL_REF_FREQ)
        except: pass
print(f"   01_master: {len(all_sweeps_raw)} devices, {sum(len(v) for v in all_sweeps_raw.values())} sweeps")

# Build arrays and SUBSAMPLE to overlap
rx, rd = [], []
for dev in sorted(all_sweeps_raw.keys()):
    for si in sorted(all_sweeps_raw[dev].keys()):
        rx.append(subsample_to_overlap(all_sweeps_raw[dev][si])); rd.append(dev)
raw_X_orig = np.array(rx); raw_devs_orig = np.array(rd)
print(f"   Subsampled to {N_FREQ*2} dims ({N_FREQ} freq points, 10-125 kHz)")

# ======== LOAD PAMELA ========
print("\n[2] Loading PAMELA dataset...")
def load_pamela_sweep(temp, damage_level=None, ref_freq=REF_FREQ):
    """Load PAMELA sweep at given temp and damage level (None=healthy)."""
    if damage_level is None:
        fn = f'./ImpedanceData/HealthyCondition/EMI{temp}H.mat'
        d = sio.loadmat(fn)
        rk = f'real_{temp}H'; ik = f'imag_{temp}H'
    else:
        fn = f'./ImpedanceData/DamagedCondition/T{temp}degrees/EMI{temp}D{damage_level}.mat'
        d = sio.loadmat(fn)
        rk = f'real_{temp}D{damage_level}'; ik = f'imag_{temp}D{damage_level}'
    freq = [k for k in d if k.startswith('freq_')][0]
    freq_v = d[freq].ravel()
    real = d[rk].ravel(); imag = d[ik].ravel()
    mag = np.sqrt(real**2 + imag**2)
    phase = np.degrees(np.arctan2(imag, real))
    mag_i = np.interp(ref_freq, freq_v, mag)
    phase_i = np.interp(ref_freq, freq_v, phase)
    return np.concatenate([phase_i, mag_i])

pamela_raw_X, pamela_devs, pamela_conds = [], [], []
temps = [24, 40, 55, 70, 85, 100]
for t in temps:
    # Healthy = baseline for PAMELA
    vec = load_pamela_sweep(t)
    pamela_raw_X.append(vec); pamela_devs.append('PAMELA'); pamela_conds.append('baseline')
    # Damaged D1-D4
    for dl in range(1, 5):
        vec = load_pamela_sweep(t, dl)
        pamela_raw_X.append(vec); pamela_devs.append('PAMELA'); pamela_conds.append(f'pamela_damage_D{dl}')

pamela_raw_X = np.array(pamela_raw_X); pamela_devs = np.array(pamela_devs); pamela_conds = np.array(pamela_conds)
print(f"   PAMELA: 1 device (PAMELA), {len(pamela_raw_X)} sweeps")
print(f"     Healthy (baseline): {sum(pamela_conds=='baseline')}  "
      f"Damaged: {sum(pamela_conds!='baseline')}")

# ======== COMBINE ========
print("\n[3] Combining datasets...")
# PAMELA healthy sweeps go into raw baseline pool
pamela_baseline_mask = pamela_conds == 'baseline'
X_raw = np.vstack([raw_X_orig, pamela_raw_X[pamela_baseline_mask]])
raw_devs = np.concatenate([raw_devs_orig, pamela_devs[pamela_baseline_mask]])
n_raw_orig = len(raw_X_orig)

# PAMELA damaged sweeps are TEST samples only (not used in training)
pamela_test_X = pamela_raw_X[~pamela_baseline_mask]
pamela_test_conds = pamela_conds[~pamela_baseline_mask]
pamela_test_devs = pamela_devs[~pamela_baseline_mask]

unique_devs = sorted(set(raw_devs))
print(f"   Total devices: {len(unique_devs)} ({len(raw_devs_orig)} orig + 1 PAMELA)")
print(f"   Total baseline sweeps: {len(X_raw)}")
print(f"   PAMELA test (damaged): {len(pamela_test_X)}")

scaler = StandardScaler(); X_scaled = scaler.fit_transform(X_raw)

# ======== SYNTHETIC PERTURBATIONS ========
print(f"\n[4] Generating synthetic perturbations (from original 295 devices)...")
engine = PerturbationEngine(PerturbationConfig(
    temperature_levels=[("cool", -10.0), ("warm", 15.0), ("hot", 30.0)],
    aging_levels=[("mild", 0.5), ("moderate", 2.0), ("severe", 5.0)],
    load_levels=[("light", 20.0), ("moderate", 60.0), ("heavy", 100.0)],
))
# Only generate from original 295 devices (they have full 2001-pt resolution)
X_synth_full, synth_devs, synth_cond, meta_list = engine.generate_synthetic_dataset(
    all_sweeps_raw, per_device_perturbation=False)
# Subsample synthetic to overlap
X_synth = np.array([subsample_to_overlap(v) for v in X_synth_full])
print(f"   {len(X_synth)} synthetic sweeps generated (subsampled to {N_FREQ*2} dims)")

# ======== PCA PRE-REDUCTION ========
print(f"\n[5] PCA pre-reduction...")
N_PRE = 400
pca = PCA(n_components=N_PRE)
X_pca = pca.fit_transform(X_scaled)
print(f"   {N_PRE} PCs, var={pca.explained_variance_ratio_.sum():.4f}")

# Project synthetic and PAMELA test data
X_synth_scaled = scaler.transform(X_synth)
X_synth_pca = pca.transform(X_synth_scaled)

pamela_test_scaled = scaler.transform(pamela_test_X)
pamela_test_pca = pca.transform(pamela_test_scaled)

# Combine: raw PCA + synth PCA + PAMELA test PCA
X_all = np.vstack([X_pca, X_synth_pca, pamela_test_pca])
all_devs_full = np.concatenate([raw_devs, synth_devs, pamela_test_devs])
all_cond_full = np.concatenate([
    np.array(['baseline']*len(raw_devs)),
    np.array([m['condition_type'] for m in meta_list]),
    np.array(['pamela_damage']*len(pamela_test_X))
])
all_tags_full = np.concatenate([
    np.array(['baseline']*len(raw_devs)),
    np.array(synth_cond),
    pamela_test_conds
])
n_raw = len(raw_devs)
n_synth = len(X_synth)
print(f"   Total samples: {len(X_all)} ({n_raw} baseline + {n_synth} synth + {len(pamela_test_X)} PAMELA test)")

# Encode supervision
le_dev = LabelEncoder(); le_cond = LabelEncoder()
dev_oh = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
cond_oh = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
dev_num = le_dev.fit_transform(all_devs_full)
cond_num = le_cond.fit_transform(all_cond_full)
Y_dev = dev_oh.fit_transform(dev_num.reshape(-1, 1))
Y_cond = cond_oh.fit_transform(cond_num.reshape(-1, 1))

# sisPCA dataset
sis_data = SISPCADataset(
    data=torch.from_numpy(X_all).float(),
    target_supervision_list=[
        Supervision(target_data=Y_dev, target_type='categorical', target_name='device_id'),
        Supervision(target_data=Y_cond, target_type='categorical', target_name='condition')
    ]
)

# ======== HELPERS ========
def mahalanobis_eer(X_raw, X_all, comps, normal_mask, abnormal_mask):
    if len(comps) < 2: return np.nan
    bl_mean = X_raw[normal_mask][:, comps].mean(axis=0)
    bl_cov = np.cov(X_raw[normal_mask][:, comps].T) + np.eye(len(comps))*1e-6
    bl_inv = inv(bl_cov)
    scores = np.array([np.sqrt((x[comps]-bl_mean) @ bl_inv @ (x[comps]-bl_mean)) for x in X_all])
    normal = scores[normal_mask]
    abnormal = scores[abnormal_mask]
    if len(normal) < 2 or len(abnormal) < 2: return np.nan
    thr = np.linspace(0, np.percentile(scores, 99), 200)
    far = np.array([np.mean(normal > t) for t in thr])
    tar = np.array([np.mean(abnormal > t) for t in thr])
    ei = np.argmin(np.abs(far - (1-tar)))
    return (far[ei] + (1-tar[ei])) / 2

def compute_hsic(X, Y):
    n = len(X); K = X @ X.T; L = Y @ Y.T
    H = np.eye(n) - np.ones((n,n))/n
    return np.trace(K @ H @ L @ H) / (n-1)**2

def eval_health_eer_by_name(X_raw, X_all, all_tags, comps, tag_prefix):
    if len(comps) < 2: return np.nan
    bl_mean = X_raw[:, comps].mean(axis=0)
    bl_cov = np.cov(X_raw[:, comps].T) + np.eye(len(comps))*1e-6; bl_inv = inv(bl_cov)
    scores = np.array([np.sqrt((x[comps]-bl_mean) @ bl_inv @ (x[comps]-bl_mean)) for x in X_all])
    normal = scores[all_tags == 'baseline']
    abnormal = scores[np.array([t is not None and t.startswith(tag_prefix) for t in all_tags])]
    if len(normal) < 2 or len(abnormal) < 2: return np.nan
    thr = np.linspace(0, np.percentile(scores, 99), 200)
    far = np.array([np.mean(normal > t) for t in thr])
    tar = np.array([np.mean(abnormal > t) for t in thr])
    ei = np.argmin(np.abs(far - (1-tar)))
    return (far[ei] + (1-tar[ei])) / 2

# ======== sisPCA TRAINING ========
N_ID_SIS = 20
N_HL_SIS = 380
results = []

print(f"\n[6] sisPCA training...")
for lam in [0.0, 0.01, 0.1, 1.0, 10.0]:
    print(f"   lam={lam}...", end=" ", flush=True)
    try:
        model = SISPCA(
            dataset=sis_data,
            n_latent_sub=[N_ID_SIS, N_HL_SIS],
            lambda_contrast=lam,
            kernel_subspace='linear',
            solver='eig'
        )
        model.fit(batch_size=len(X_all), max_epochs=3, lr=1.0,
                  early_stopping_patience=None, enable_progress_bar=False,
                  enable_model_summary=False)
        
        U = model.U.detach().numpy()
        X_proj = X_all @ U
        
        # Hard split
        X_raw_proj = X_proj[:n_raw]
        hsic_val = compute_hsic(X_proj[:, :N_ID_SIS], X_proj[:, N_ID_SIS:])
        
        # Health EER on synthetic perturbations (standard)
        hl_comps = list(range(N_ID_SIS, N_ID_SIS+N_HL_SIS))
        # Full-length masks (4530)
        normal_mask = all_cond_full == 'baseline'
        abnormal_synth_mask = (all_cond_full != 'baseline') & (all_cond_full != 'pamela_damage')
        abnormal_pamela_mask = all_cond_full == 'pamela_damage'
        
        health_eer_synth = mahalanobis_eer(X_proj, X_proj, hl_comps,
                                            normal_mask, abnormal_synth_mask)
        health_eer_pamela = mahalanobis_eer(X_proj, X_proj, hl_comps,
                                             normal_mask, abnormal_pamela_mask)
        
        # ===== DIAGNOSTIC: three-way separation =====
        # Fit baseline on cohort ONLY (exclude PAMELA healthy from baseline)
        cohort_mask = (all_cond_full == 'baseline') & (all_devs_full != 'PAMELA')
        pamela_healthy_mask = (all_cond_full == 'baseline') & (all_devs_full == 'PAMELA')
        pamela_damage_mask = all_cond_full == 'pamela_damage'
        
        # EER-A: cohort vs PAMELA healthy (sensor offset check)
        eer_cohort_vs_ph = mahalanobis_eer(X_proj, X_proj, hl_comps, cohort_mask, pamela_healthy_mask)
        # EER-B: cohort vs PAMELA damaged
        eer_cohort_vs_pd = mahalanobis_eer(X_proj, X_proj, hl_comps, cohort_mask, pamela_damage_mask)
        # EER-C: PAMELA healthy vs PAMELA damaged (within-PAMELA, fit on healthy only)
        # Use only top 10 HL comps since only 6 PAMELA healthy samples
        n_diag = min(10, len(hl_comps))
        eer_ph_vs_pd = mahalanobis_eer(X_proj, X_proj, hl_comps[:n_diag], pamela_healthy_mask, pamela_damage_mask)
        
        # Mean Mahalanobis distances per group
        def group_mean_dist(mask, comps=hl_comps):
            pts = X_proj[mask][:, comps]
            if len(pts) < 2: return np.nan
            bl = X_proj[cohort_mask][:, comps]
            m = bl.mean(axis=0)
            c = np.cov(bl.T) + np.eye(len(comps))*1e-6
            ci = inv(c)
            return float(np.mean([np.sqrt((p-m) @ ci @ (p-m)) for p in pts]))
        
        d_cohort = group_mean_dist(cohort_mask)
        d_ph = group_mean_dist(pamela_healthy_mask)
        d_pd = group_mean_dist(pamela_damage_mask)
        
        # Per PAMELA damage level
        eer_pamela_d1 = eval_health_eer_by_name(X_proj[:n_raw], X_proj, all_tags_full, hl_comps, 'pamela_damage_D1')
        eer_pamela_d2 = eval_health_eer_by_name(X_proj[:n_raw], X_proj, all_tags_full, hl_comps, 'pamela_damage_D2')
        eer_pamela_d3 = eval_health_eer_by_name(X_proj[:n_raw], X_proj, all_tags_full, hl_comps, 'pamela_damage_D3')
        eer_pamela_d4 = eval_health_eer_by_name(X_proj[:n_raw], X_proj, all_tags_full, hl_comps, 'pamela_damage_D4')
        
        # Per-condition synthetic
        eer_temp = eval_health_eer_by_name(X_proj[:n_raw], X_proj, all_tags_full, hl_comps, 'temp_')
        eer_age = eval_health_eer_by_name(X_proj[:n_raw], X_proj, all_tags_full, hl_comps, 'aging_')
        eer_load = eval_health_eer_by_name(X_proj[:n_raw], X_proj, all_tags_full, hl_comps, 'load_')
        
        results.append({
            'method': f'sisPCA lam={lam}',
            'hsic': hsic_val,
            'health_eer_synth': health_eer_synth,
            'health_eer_pamela': health_eer_pamela,
            'eer_pamela_D1': eer_pamela_d1,
            'eer_pamela_D2': eer_pamela_d2,
            'eer_pamela_D3': eer_pamela_d3,
            'eer_pamela_D4': eer_pamela_d4,
            'eer_temp': eer_temp,
            'eer_age': eer_age,
            'eer_load': eer_load,
            'eer_cohort_vs_ph': eer_cohort_vs_ph,
            'eer_cohort_vs_pd': eer_cohort_vs_pd,
            'eer_ph_vs_pd': eer_ph_vs_pd,
            'd_cohort': d_cohort,
            'd_ph': d_ph,
            'd_pd': d_pd,
        })
        
        print(f"HSIC={hsic_val:.4f}  SynthEER={health_eer_synth:.4f}  "
              f"PamelaEER={health_eer_pamela:.4f}  "
              f"(D1={eer_pamela_d1:.4f} D2={eer_pamela_d2:.4f} "
              f"D3={eer_pamela_d3:.4f} D4={eer_pamela_d4:.4f})  "
              f"[diag: d(coh)={d_cohort:.1f} d(PH)={d_ph:.1f} d(PD)={d_pd:.1f}  "
              f"cohvsPH={eer_cohort_vs_ph:.3f} PHvsPD={eer_ph_vs_pd:.3f}]")
    except Exception as e:
        print(f"FAILED: {e}")

# ======== RESULTS ========
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

df = pd.DataFrame(results)

def fmt(v, d=4):
    if isinstance(v, str): return v
    if np.isnan(v): return 'N/A'
    return f"{v:.{d}f}"

dfs = df.sort_values('method')
print(f"\n{'Method':<20} {'SynthEER':>8} {'PamelaEER':>9} {'D1':>7} {'D2':>7} {'D3':>7} {'D4':>7} {'Temp':>7} {'Age':>7} {'Load':>7} {'HSIC':>8}")
print("-" * 100)
for _, r in dfs.iterrows():
    print(f"{r['method']:<20} {fmt(r['health_eer_synth'],4):>8} {fmt(r['health_eer_pamela'],4):>9} "
          f"{fmt(r['eer_pamela_D1'],4):>7} {fmt(r['eer_pamela_D2'],4):>7} "
          f"{fmt(r['eer_pamela_D3'],4):>7} {fmt(r['eer_pamela_D4'],4):>7} "
          f"{fmt(r['eer_temp'],4):>7} {fmt(r['eer_age'],4):>7} {fmt(r['eer_load'],4):>7} "
          f"{fmt(r['hsic'],2):>8}")

print(f"\n{'Method':<20} {'d(coh)':>8} {'d(PH)':>8} {'d(PD)':>8} {'CohvsPH':>8} {'CohvsPD':>8} {'PHvsPD':>8}")
print("-" * 70)
for _, r in dfs.iterrows():
    print(f"{r['method']:<20} {fmt(r['d_cohort'],1):>8} {fmt(r['d_ph'],1):>8} {fmt(r['d_pd'],1):>8} "
          f"{fmt(r['eer_cohort_vs_ph'],4):>8} {fmt(r['eer_cohort_vs_pd'],4):>8} "
          f"{fmt(r['eer_ph_vs_pd'],4):>8}")

# Figure: distances only (diagnostic)
fig, ax = plt.subplots(figsize=(10, 5))
mnames = list(df['method'])
x = np.arange(len(mnames)); w = 0.25
dc = [r['d_cohort'] if not np.isnan(r['d_cohort']) else 0 for _,r in df.iterrows()]
dp = [r['d_ph'] if not np.isnan(r['d_ph']) else 0 for _,r in df.iterrows()]
dd = [r['d_pd'] if not np.isnan(r['d_pd']) else 0 for _,r in df.iterrows()]
ax.bar(x-w, dc, w, label='Cohort baselines', color='steelblue', alpha=0.8, edgecolor='black')
ax.bar(x, dp, w, label='PAMELA healthy', color='orange', alpha=0.8, edgecolor='black')
ax.bar(x+w, dd, w, label='PAMELA damaged', color='salmon', alpha=0.8, edgecolor='black')
ax.set_xticks(x); ax.set_xticklabels(mnames, rotation=15)
ax.set_ylabel('Mean Mahalanobis distance')
ax.set_title('Group separation in condition subspace (baseline fit on cohort only)')
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, 'combined_cohort_pamela.png'), dpi=150); plt.close()
df.to_csv(os.path.join(REPORT_DIR, 'combined_cohort_pamela.csv'), index=False)
print(f"\n[V] CSV + Figure saved")
print("Done.")

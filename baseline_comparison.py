"""
Compare sisPCA against PCA, FastICA, PLS baselines on the combined cohort + PAMELA transfer task.
All methods receive the same PCA-pre-reduced data; they differ in how they select
identity vs. condition subspaces.

Metrics: Health EER on synthetic perturbations and on real PAMELA damage (D1–D4).
"""
import os, sys, warnings, time
warnings.filterwarnings('ignore')
from collections import defaultdict
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.decomposition import PCA, FastICA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from scipy.linalg import inv
import scipy.io as sio
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synthetic_perturbations import PerturbationEngine, PerturbationConfig
from subspace_comparison import load_sweep_vector, collect_device_files
from sispca.data import Supervision, SISPCADataset
from sispca.model import SISPCA

rng = np.random.default_rng(42)

# Frequency setup
FULL_N_FREQ = 2001
FULL_REF_FREQ = np.linspace(10000, 1000000, FULL_N_FREQ)
START_FREQ, END_FREQ = 10000, 125000
N_FREQ = 200
REF_FREQ = np.linspace(START_FREQ, END_FREQ, N_FREQ)
N_PRE = 400

REPORT_DIR = r"./rq_analysis_reports"
os.makedirs(REPORT_DIR, exist_ok=True)
sns.set_style("whitegrid")
plt.rcParams.update({'figure.max_open_warning': 0})

def subsample_to_overlap(vec, n_freq_full=FULL_N_FREQ, n_freq_out=N_FREQ):
    return np.concatenate([vec[:n_freq_out], vec[n_freq_full:n_freq_full+n_freq_out]])

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

def eval_health_eer_by_name(X_raw, X_all, all_tags, comps, tag_prefix):
    if len(comps) < 2: return np.nan
    bl_mean = X_raw[:, comps].mean(axis=0)
    bl_cov = np.cov(X_raw[:, comps].T) + np.eye(len(comps))*1e-6; bl_inv = inv(bl_cov)
    scores = np.array([np.sqrt((x[comps]-bl_mean) @ bl_inv @ (x[comps]-bl_mean)) for x in X_all])
    normal = scores[all_tags == 'baseline']
    abnormal = scores[np.array([t is not None and str(t).startswith(tag_prefix) for t in all_tags])]
    if len(normal) < 2 or len(abnormal) < 2: return np.nan
    thr = np.linspace(0, np.percentile(scores, 99), 200)
    far = np.array([np.mean(normal > t) for t in thr])
    tar = np.array([np.mean(abnormal > t) for t in thr])
    ei = np.argmin(np.abs(far - (1-tar)))
    return (far[ei] + (1-tar[ei])) / 2

def compute_f_ratios(X, id_labels, cond_labels, n_pcs=None):
    if n_pcs is None: n_pcs = X.shape[1]
    id_scores, cond_scores = np.zeros(n_pcs), np.zeros(n_pcs)
    for pi in range(n_pcs):
        vals = X[:, pi]
        ss_total = np.var(vals) * len(vals)
        ss_between_id = sum(len(vals[id_labels == u]) * (np.mean(vals[id_labels == u]) - np.mean(vals))**2 for u in set(id_labels))
        ss_between_cond = sum(len(vals[cond_labels == c]) * (np.mean(vals[cond_labels == c]) - np.mean(vals))**2 for c in set(cond_labels))
        id_scores[pi] = ss_between_id / max(ss_total - ss_between_id, 1e-10)
        cond_scores[pi] = ss_between_cond / max(ss_total - ss_between_cond, 1e-10)
    return id_scores, cond_scores

def partition_by_ratio(id_scores, cond_scores):
    ratio = np.where(cond_scores > 0, id_scores / (id_scores + cond_scores + 1e-10), 0.5)
    # ID-only: ratio > 0.7, HL-only: ratio < 0.3, dual: in between
    id_only = np.where(ratio > 0.7)[0]
    hl_only = np.where(ratio < 0.3)[0]
    id_sub = np.where(ratio > 0.5)[0]   # at least ID-leaning
    hl_sub = np.where(ratio < 0.5)[0]   # at least HL-leaning
    dual = np.where((ratio >= 0.3) & (ratio <= 0.7))[0]
    neither = np.where((id_scores == 0) & (cond_scores == 0))[0]
    gap = 0
    return id_sub, hl_sub, id_only, hl_only, dual, neither, gap

def compute_hsic(X, Y):
    n = len(X); H = np.eye(n) - np.ones((n,n))/n
    K = X @ X.T; L = Y @ Y.T
    return np.trace(K @ H @ L @ H) / (n-1)**2

def norm01(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / max(hi - lo, 1e-10)

def evaluate_method(name, X_proj, n_raw, all_devs_all, all_cond_all, all_tags_full, n_id_comps, n_hl_comps):
    id_comps = list(range(n_id_comps))
    hl_comps = list(range(n_id_comps, n_id_comps + n_hl_comps))
    X_raw_proj = X_proj[:n_raw]
    hsic_val = compute_hsic(X_proj[:, id_comps], X_proj[:, hl_comps]) if len(id_comps) and len(hl_comps) else np.nan
    normal_mask = all_cond_all == 'baseline'
    abnormal_synth = (all_cond_all != 'baseline') & (all_cond_all != 'pamela_damage')
    abnormal_pamela = all_cond_all == 'pamela_damage'
    eer_synth = mahalanobis_eer(X_proj, X_proj, hl_comps, normal_mask, abnormal_synth)
    eer_pamela = mahalanobis_eer(X_proj, X_proj, hl_comps, normal_mask, abnormal_pamela)
    eer_d1 = eval_health_eer_by_name(X_raw_proj, X_proj, all_tags_full, hl_comps, 'pamela_damage_D1')
    eer_d2 = eval_health_eer_by_name(X_raw_proj, X_proj, all_tags_full, hl_comps, 'pamela_damage_D2')
    eer_d3 = eval_health_eer_by_name(X_raw_proj, X_proj, all_tags_full, hl_comps, 'pamela_damage_D3')
    eer_d4 = eval_health_eer_by_name(X_raw_proj, X_proj, all_tags_full, hl_comps, 'pamela_damage_D4')
    eer_temp = eval_health_eer_by_name(X_raw_proj, X_proj, all_tags_full, hl_comps, 'temp_')
    eer_age = eval_health_eer_by_name(X_raw_proj, X_proj, all_tags_full, hl_comps, 'aging_')
    eer_load = eval_health_eer_by_name(X_raw_proj, X_proj, all_tags_full, hl_comps, 'load_')
    return {
        'method': name, 'hsic': hsic_val, 'n_id': len(id_comps), 'n_hl': len(hl_comps),
        'health_eer_synth': eer_synth, 'health_eer_pamela': eer_pamela,
        'eer_pamela_D1': eer_d1, 'eer_pamela_D2': eer_d2,
        'eer_pamela_D3': eer_d3, 'eer_pamela_D4': eer_d4,
        'eer_temp': eer_temp, 'eer_age': eer_age, 'eer_load': eer_load,
    }

# ======== DATA LOADING (identical to combined_cohort_pamela.py) ========
print("=" * 60)
print("BASELINE COMPARISON: sisPCA vs PCA / ICA / PLS")
print("=" * 60)

print("\n[1] Loading 01_master_dataset...")
device_files = collect_device_files(r"./01_master_dataset")
all_sweeps_raw = {}
for dev_id in sorted(device_files.keys()):
    all_sweeps_raw[dev_id] = {}
    for si in sorted(device_files[dev_id].keys()):
        try: all_sweeps_raw[dev_id][si] = load_sweep_vector(device_files[dev_id][si], ref_freq=FULL_REF_FREQ)
        except: pass
print(f"   {len(all_sweeps_raw)} devices, {sum(len(v) for v in all_sweeps_raw.values())} sweeps")

rx, rd = [], []
for dev in sorted(all_sweeps_raw.keys()):
    for si in sorted(all_sweeps_raw[dev].keys()):
        rx.append(subsample_to_overlap(all_sweeps_raw[dev][si])); rd.append(dev)
raw_X_orig = np.array(rx); raw_devs_orig = np.array(rd)
print(f"   Subsampled to {N_FREQ*2} dims")

print("\n[2] Loading PAMELA...")
def load_pamela_sweep(temp, damage_level=None):
    if damage_level is None:
        fn = f'./ImpedanceData/HealthyCondition/EMI{temp}H.mat'
        d = sio.loadmat(fn); rk = f'real_{temp}H'; ik = f'imag_{temp}H'
    else:
        fn = f'./ImpedanceData/DamagedCondition/T{temp}degrees/EMI{temp}D{damage_level}.mat'
        d = sio.loadmat(fn); rk = f'real_{temp}D{damage_level}'; ik = f'imag_{temp}D{damage_level}'
    freq = [k for k in d if k.startswith('freq_')][0]
    freq_v = d[freq].ravel()
    real = d[rk].ravel(); imag = d[ik].ravel()
    mag = np.sqrt(real**2 + imag**2); phase = np.degrees(np.arctan2(imag, real))
    mag_i = np.interp(REF_FREQ, freq_v, mag); phase_i = np.interp(REF_FREQ, freq_v, phase)
    return np.concatenate([phase_i, mag_i])

pamela_raw_X, pamela_devs, pamela_conds = [], [], []
temps = [24, 40, 55, 70, 85, 100]
for t in temps:
    vec = load_pamela_sweep(t)
    pamela_raw_X.append(vec); pamela_devs.append('PAMELA'); pamela_conds.append('baseline')
    for dl in range(1, 5):
        vec = load_pamela_sweep(t, dl)
        pamela_raw_X.append(vec); pamela_devs.append('PAMELA'); pamela_conds.append(f'pamela_damage_D{dl}')
pamela_raw_X = np.array(pamela_raw_X); pamela_devs = np.array(pamela_devs); pamela_conds = np.array(pamela_conds)
print(f"   PAMELA: {len(pamela_raw_X)} sweeps ({sum(pamela_conds=='baseline')} healthy, {sum(pamela_conds!='baseline')} damaged)")

# Combine baselines
pamela_baseline_mask = pamela_conds == 'baseline'
X_raw = np.vstack([raw_X_orig, pamela_raw_X[pamela_baseline_mask]])
raw_devs = np.concatenate([raw_devs_orig, pamela_devs[pamela_baseline_mask]])

pamela_test_X = pamela_raw_X[~pamela_baseline_mask]
pamela_test_conds = pamela_conds[~pamela_baseline_mask]
pamela_test_devs = pamela_devs[~pamela_baseline_mask]
unique_devs = sorted(set(raw_devs))
print(f"   Combined: {len(unique_devs)} devices, {len(X_raw)} baselines")

# Scale + PCA pre-reduction
scaler = StandardScaler(); X_scaled = scaler.fit_transform(X_raw)
pca_pre = PCA(n_components=N_PRE)
X_pca = pca_pre.fit_transform(X_scaled)
print(f"   PCA pre: {N_PRE} PCs, var={pca_pre.explained_variance_ratio_.sum():.4f}")

# Synthetic perturbations (from original 295 only)
print(f"\n[3] Generating synthetic perturbations...")
engine = PerturbationEngine(PerturbationConfig(
    temperature_levels=[("cool", -10.0), ("warm", 15.0), ("hot", 30.0)],
    aging_levels=[("mild", 0.5), ("moderate", 2.0), ("severe", 5.0)],
    load_levels=[("light", 20.0), ("moderate", 60.0), ("heavy", 100.0)],
))
X_synth_full, synth_devs, synth_cond, meta_list = engine.generate_synthetic_dataset(
    all_sweeps_raw, per_device_perturbation=False)
X_synth = np.array([subsample_to_overlap(v) for v in X_synth_full])
print(f"   {len(X_synth)} synthetic sweeps")

# Project synth + PAMELA test through PCA
X_synth_scaled = scaler.transform(X_synth)
X_synth_pca = pca_pre.transform(X_synth_scaled)
pamela_test_scaled = scaler.transform(pamela_test_X)
pamela_test_pca = pca_pre.transform(pamela_test_scaled)

# Combined arrays
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
print(f"   Total: {len(X_all)} ({n_raw} baseline + {n_synth} synth + {len(pamela_test_X)} PAMELA test)")

# Supervision labels
le_dev = LabelEncoder(); le_cond = LabelEncoder()
dev_oh = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
cond_oh = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
dev_num = le_dev.fit_transform(all_devs_full)
cond_num = le_cond.fit_transform(all_cond_full)
Y_dev = dev_oh.fit_transform(dev_num.reshape(-1, 1))
Y_cond = cond_oh.fit_transform(cond_num.reshape(-1, 1))

# Split indices
raw_devs_only = raw_devs_orig  # original 295 devices (without PAMELA)
synth_devs_only = synth_devs

# For F-ratio computation: use raw baselines for ID, synthetic for condition
cond_type_arr = np.array([m['condition_type'] for m in meta_list])

# ======== METHODS ========
results = []

# -------------------------------------------------------
# METHOD 1: PCA (no separation) — top N_PRE PCs as-is
# -------------------------------------------------------
print(f"\n[4a] PCA (no separation)...", flush=True)
t0 = time.time()
X_proj_pca = X_all[:, :N_PRE]
id_fs, hl_fs = compute_f_ratios(X_proj_pca, all_devs_full, all_cond_full)
id_s = norm01(id_fs); hl_s = norm01(hl_fs)
_, _, id_only, hl_only, _, _, _ = partition_by_ratio(id_s, hl_s)
n_id = max(2, len(id_only))
n_hl = max(2, len(hl_only))
# If too few, take hard split at 20
if n_id < 20: n_id = 20
if n_hl < 20: n_hl = min(N_PRE - n_id, 380)
results.append(evaluate_method('PCA (F-ratio split)', X_proj_pca, n_raw, all_devs_full, all_cond_full, all_tags_full, n_id, n_hl))
r = results[-1]
print(f"   PCA: ID={r['n_id']}, HL={r['n_hl']}, SynthEER={r['health_eer_synth']:.4f}, PamelaEER={r['health_eer_pamela']:.4f}")

# -------------------------------------------------------
# METHOD 2: PCA (simple hard split) — first 20 = ID, rest = HL
# -------------------------------------------------------
print(f"[4b] PCA (hard split 20/380)...", flush=True)
results.append(evaluate_method('PCA (20/380 split)', X_proj_pca, n_raw, all_devs_full, all_cond_full, all_tags_full, 20, 380))
r = results[-1]
print(f"   PCA hard: SynthEER={r['health_eer_synth']:.4f}, PamelaEER={r['health_eer_pamela']:.4f}")

# PCA (constrained: 380 ID, 20 HL — like sisPCA but reversed, to test limited HL)
print(f"[4c] PCA (constrained 20 HL comps)...", flush=True)
results.append(evaluate_method('PCA (20 HL only)', X_proj_pca, n_raw, all_devs_full, all_cond_full, all_tags_full, 380, 20))
r = results[-1]
print(f"   PCA 20HL: SynthEER={r['health_eer_synth']:.4f}, PamelaEER={r['health_eer_pamela']:.4f}")

# -------------------------------------------------------
# METHOD 3: PCA (sep F-ratio: ID on raw, HL on synth)
# -------------------------------------------------------
print(f"[4d] PCA (sep F-ratio)...", flush=True)
id_fs_raw, _ = compute_f_ratios(X_pca[:n_raw], raw_devs, np.zeros(n_raw))
_, hl_fs_syn = compute_f_ratios(X_synth_pca, synth_devs, cond_type_arr)
id_s_sep = norm01(id_fs_raw); hl_s_sep = norm01(hl_fs_syn)
_, _, id_only_sep, hl_only_sep, _, _, _ = partition_by_ratio(id_s_sep, hl_s_sep)
n_id_sep = max(2, len(id_only_sep))
n_hl_sep = max(2, len(hl_only_sep))
if n_id_sep < 20: n_id_sep = 20
if n_hl_sep < 20: n_hl_sep = min(N_PRE - n_id_sep, 380)
results.append(evaluate_method('PCA (sep F-ratio)', X_proj_pca, n_raw, all_devs_full, all_cond_full, all_tags_full, n_id_sep, n_hl_sep))
r = results[-1]
print(f"   PCA sep: ID={r['n_id']}, HL={r['n_hl']}, SynthEER={r['health_eer_synth']:.4f}, PamelaEER={r['health_eer_pamela']:.4f}")

# -------------------------------------------------------
# METHOD 4: FastICA
# -------------------------------------------------------
print(f"[4d] FastICA...", flush=True)
t0 = time.time()
n_ica = min(100, N_PRE)
X_ica = FastICA(n_components=n_ica, random_state=42, max_iter=2000, tol=1e-4).fit_transform(X_all[:, :n_ica])
id_fs_ica, hl_fs_ica = compute_f_ratios(X_ica, all_devs_full, all_cond_full)
id_s_ica = norm01(id_fs_ica); hl_s_ica = norm01(hl_fs_ica)
_, _, id_only_ica, hl_only_ica, _, _, _ = partition_by_ratio(id_s_ica, hl_s_ica)
n_id_ica = max(2, len(id_only_ica))
n_hl_ica = max(2, len(hl_only_ica))
if n_id_ica < 10: n_id_ica = 10
if n_hl_ica < 20: n_hl_ica = min(n_ica - n_id_ica, 380)
results.append(evaluate_method('FastICA', X_ica, n_raw, all_devs_full, all_cond_full, all_tags_full, n_id_ica, n_hl_ica))
r = results[-1]; et = time.time()-t0
print(f"   ICA: ID={r['n_id']}, HL={r['n_hl']}, SynthEER={r['health_eer_synth']:.4f}, PamelaEER={r['health_eer_pamela']:.4f} ({et:.1f}s)")

# -------------------------------------------------------
# METHOD 5: PLS
# -------------------------------------------------------
print(f"[4e] PLS (on full data)...", flush=True)
t0 = time.time()
tgt = np.hstack([Y_dev, Y_cond])
n_pls = min(100, N_PRE, tgt.shape[1]-1)
pls = PLSRegression(n_components=n_pls, max_iter=500, tol=1e-4)
X_pls = pls.fit_transform(X_all[:, :N_PRE], tgt)[0]
id_fs_pls, hl_fs_pls = compute_f_ratios(X_pls, all_devs_full, all_cond_full)
id_s_pls = norm01(id_fs_pls); hl_s_pls = norm01(hl_fs_pls)
_, _, id_only_pls, hl_only_pls, _, _, _ = partition_by_ratio(id_s_pls, hl_s_pls)
n_id_pls = max(2, len(id_only_pls))
n_hl_pls = max(2, len(hl_only_pls))
if n_id_pls < 10: n_id_pls = 10
if n_hl_pls < 20: n_hl_pls = min(n_pls - n_id_pls, 380)
results.append(evaluate_method('PLS', X_pls, n_raw, all_devs_full, all_cond_full, all_tags_full, n_id_pls, n_hl_pls))
r = results[-1]; et = time.time()-t0
print(f"   PLS: ID={r['n_id']}, HL={r['n_hl']}, SynthEER={r['health_eer_synth']:.4f}, PamelaEER={r['health_eer_pamela']:.4f} ({et:.1f}s)")

# -------------------------------------------------------
# METHOD 6: sisPCA (from combined_cohort_pamela.py)
# -------------------------------------------------------
print(f"\n[5] sisPCA...", flush=True)
sis_data = SISPCADataset(
    data=torch.from_numpy(X_all).float(),
    target_supervision_list=[
        Supervision(target_data=Y_dev, target_type='categorical', target_name='device_id'),
        Supervision(target_data=Y_cond, target_type='categorical', target_name='condition')
    ]
)
for lam in [0.0, 0.01, 0.05, 0.1, 1.0, 10.0]:
    t0 = time.time()
    try:
        model = SISPCA(
            dataset=sis_data, n_latent_sub=[20, 380], lambda_contrast=lam,
            kernel_subspace='linear', solver='eig'
        )
        model.fit(batch_size=len(X_all), max_epochs=3, lr=1.0,
                  early_stopping_patience=None, enable_progress_bar=False,
                  enable_model_summary=False)
        U = model.U.detach().numpy()
        X_proj = X_all @ U
        results.append(evaluate_method(f'sisPCA lam={lam}', X_proj, n_raw,
                        all_devs_full, all_cond_full, all_tags_full, 20, 380))
        r = results[-1]; et = time.time()-t0
        print(f"   lam={lam}: SynthEER={r['health_eer_synth']:.4f}, "
              f"PamelaEER={r['health_eer_pamela']:.4f} ({et:.1f}s)")
    except Exception as e:
        print(f"   lam={lam}: FAILED — {e}")

# ======== RESULTS ========
print("\n" + "=" * 60)
print("BASELINE COMPARISON RESULTS")
print("=" * 60)

df = pd.DataFrame(results)
rename_map = {
    'method': 'Method', 'hsic': 'HSIC', 'n_id': 'N-ID', 'n_hl': 'N-HL',
    'health_eer_synth': 'SynthEER', 'health_eer_pamela': 'PamelaEER',
    'eer_pamela_D1': 'EER-D1', 'eer_pamela_D2': 'EER-D2',
    'eer_pamela_D3': 'EER-D3', 'eer_pamela_D4': 'EER-D4',
    'eer_temp': 'TempEER', 'eer_age': 'AgeEER', 'eer_load': 'LoadEER',
}
display_cols = ['Method', 'HSIC', 'SynthEER', 'PamelaEER', 'EER-D1', 'EER-D2', 'EER-D3', 'EER-D4', 'TempEER', 'AgeEER', 'LoadEER', 'N-ID', 'N-HL']

def fmt(v, d=4):
    if isinstance(v, str): return v
    if np.isnan(v): return 'N/A'
    return f"{v:.{d}f}"

print(f"\n{'Method':<22} {'HSIC':>10} {'Synth':>8} {'Pamela':>8} {'D1':>8} {'D2':>8} {'D3':>8} {'D4':>8} {'Temp':>8} {'Age':>8} {'Load':>8} {'N-ID':>5} {'N-HL':>5}")
print("-" * 120)
for _, r in df.iterrows():
    print(f"{r['method']:<22} {fmt(r['hsic'],2):>10} {fmt(r['health_eer_synth'],4):>8} "
          f"{fmt(r['health_eer_pamela'],4):>8} "
          f"{fmt(r['eer_pamela_D1'],4):>8} {fmt(r['eer_pamela_D2'],4):>8} "
          f"{fmt(r['eer_pamela_D3'],4):>8} {fmt(r['eer_pamela_D4'],4):>8} "
          f"{fmt(r['eer_temp'],4):>8} {fmt(r['eer_age'],4):>8} {fmt(r['eer_load'],4):>8} "
          f"{int(r['n_id']):>5} {int(r['n_hl']):>5}")

# Figure
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
mnames = list(df['method'])
palette = ['steelblue','royalblue','navy','green','orange','salmon','firebrick','darkred','purple','brown','pink','gold']
cs = palette[:len(mnames)]

ax = axes[0]
hs = [r['hsic'] if isinstance(r['hsic'],(int,float)) and not np.isnan(r['hsic']) else 0 for _,r in df.iterrows()]
ax.bar(mnames, hs, color=cs, alpha=0.8, edgecolor='black')
ax.set_ylabel('HSIC'); ax.set_title('Subspace Independence (lower = better)')
ax.tick_params(axis='x', rotation=20)

ax = axes[1]
se = [r['health_eer_synth'] if isinstance(r['health_eer_synth'],(int,float)) and not np.isnan(r['health_eer_synth']) else 0 for _,r in df.iterrows()]
pe = [r['health_eer_pamela'] if isinstance(r['health_eer_pamela'],(int,float)) and not np.isnan(r['health_eer_pamela']) else 0 for _,r in df.iterrows()]
x = np.arange(len(mnames)); w = 0.3
ax.bar(x-w, se, w, label='Synthetic', color='steelblue', alpha=0.8, edgecolor='black')
ax.bar(x+w, pe, w, label='PAMELA (real)', color='salmon', alpha=0.8, edgecolor='black')
ax.set_xticks(x); ax.set_xticklabels(mnames, rotation=20, fontsize=8)
ax.set_ylabel('Health EER'); ax.set_title('Synthetic vs Real Damage Detection'); ax.legend(fontsize=8)

ax = axes[2]
for idx, dl in enumerate(range(1,5)):
    vals = [r.get(f'eer_pamela_D{dl}',np.nan) if isinstance(r.get(f'eer_pamela_D{dl}'),(int,float)) and not np.isnan(r.get(f'eer_pamela_D{dl}',np.nan)) else 0 for _,r in df.iterrows()]
    ax.bar(x+(idx-1.5)*0.2, vals, 0.2, label=f'D{dl}', alpha=0.8, edgecolor='black')
ax.set_xticks(x); ax.set_xticklabels(mnames, rotation=20, fontsize=8)
ax.set_ylabel('EER'); ax.set_title('PAMELA Per-Damage-Level EER'); ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, 'baseline_comparison.png'), dpi=150); plt.close()

df_out = df.rename(columns=rename_map)
df_out.to_csv(os.path.join(REPORT_DIR, 'baseline_comparison.csv'), index=False)
print(f"\n[V] CSV + Figure saved to {REPORT_DIR}")
print("Done.")

"""
Subspace separation method comparison: PCA, FastICA, PLS, sisPCA.
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
from scipy.spatial.distance import pdist
from scipy.stats import f_oneway
from scipy.linalg import inv
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synthetic_perturbations import PerturbationEngine, PerturbationConfig

# sisPCA
from sispca.data import Supervision, SISPCADataset
from sispca.model import SISPCA

DEVICE_FOLDER = r"./01_master_dataset"
REPORT_DIR = r"./rq_analysis_reports"
EXCLUDED_DEVICES = ["201", "253", "254", "258", "310"]
USE_PHASE = True; START_FREQ = 10000; END_FREQ = 1000000; N_FREQ_POINTS = 2001
REF_FREQ = np.linspace(START_FREQ, END_FREQ, N_FREQ_POINTS)
N_PCS = 100; rng = np.random.default_rng(42)
os.makedirs(REPORT_DIR, exist_ok=True)
sns.set_style("whitegrid")
plt.rcParams.update({'figure.max_open_warning': 0})

def robust_load_csv(path):
    for var in [{"skiprows": 32}, {"skiprows": 33}, {"skiprows": 1}, {}]:
        try: return pd.read_csv(path, **var)
        except: pass
    return pd.read_csv(path)

def extract_columns(df, use_phase=False):
    cols_map = {c.lower(): c for c in df.columns}
    freq_col = next((cols_map[c] for c in ["frequency", "freq", "f"] if c in cols_map), df.columns[0])
    imp_col = next((cols_map[c] for c in ["trace m (db)", "magnitude", "trace |z|", "|z|"] if c in cols_map), df.columns[1])
    phase_col = next((cols_map[c] for c in ["trace th (deg)", "phase", "angle"] if c in cols_map), None) if use_phase else None
    freq = df[freq_col].values; imp = df[imp_col].values
    phase = df[phase_col].values if (use_phase and phase_col) else None
    return freq, phase, imp

def load_sweep_vector(path, ref_freq=REF_FREQ, use_phase=USE_PHASE):
    df = robust_load_csv(path); freq, phase, imp = extract_columns(df, use_phase)
    if freq[0] > freq[-1]: freq, imp = freq[::-1], imp[::-1]
    if phase is not None: phase = phase[::-1]
    imp_interp = np.interp(ref_freq, freq, imp)
    if use_phase:
        phase_interp = np.zeros_like(ref_freq) if phase is None else np.interp(ref_freq, freq, phase)
        return np.concatenate([phase_interp, imp_interp])
    return imp_interp

def collect_device_files(folder):
    device_files = defaultdict(dict)
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(".csv"): continue
        name = os.path.splitext(fname)[0]
        if "_" not in name: continue
        prefix, idx = name.rsplit("_", 1)
        try: idx_int = int(idx)
        except: continue
        device_files[prefix][idx_int] = os.path.join(folder, fname)
    return device_files

def norm01(x):
    x = np.array(x, dtype=float)
    if x.max() == x.min(): return np.zeros_like(x)
    return (x - x.min()) / (x.max() - x.min())

def compute_f_ratios(X, id_labels, cond_labels, n_pcs=N_PCS):
    id_fs = np.zeros(n_pcs)
    for pc in range(n_pcs):
        groups = [X[id_labels == d, pc] for d in sorted(set(id_labels))]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) > 1: id_fs[pc], _ = f_oneway(*groups)
    hl_fs = np.zeros(n_pcs)
    for pc in range(n_pcs):
        groups = [X[cond_labels == c, pc] for c in sorted(set(cond_labels))]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) > 1: hl_fs[pc], _ = f_oneway(*groups)
    return id_fs, hl_fs

def partition_by_ratio(id_scores, hl_scores):
    eps = 1e-8; ratio = id_scores / (hl_scores + eps)
    ratio_sorted_idx = np.argsort(ratio)[::-1]; ratios_sorted = ratio[ratio_sorted_idx]
    gaps = np.diff(ratios_sorted)
    gap_idx = np.argmax(gaps) if len(gaps) > 0 else 0
    gap_val = (ratios_sorted[gap_idx] + ratios_sorted[gap_idx+1])/2 if len(ratios_sorted) > 1 else 0.5
    id_sub = ratio_sorted_idx[:gap_idx+1]; hl_sub = ratio_sorted_idx[gap_idx+1:]
    id_med = np.median(id_scores); hl_med = np.median(hl_scores)
    id_only = np.where((id_scores > id_med) & (hl_scores <= hl_med))[0]
    hl_only = np.where((id_scores <= id_med) & (hl_scores > hl_med))[0]
    dual = np.where((id_scores > id_med) & (hl_scores > hl_med))[0]
    neither = np.where((id_scores <= id_med) & (hl_scores <= hl_med))[0]
    return id_sub, hl_sub, id_only, hl_only, dual, neither, gap_val

def compute_hsic(X, Y):
    n = X.shape[0]; K = X @ X.T; L = Y @ Y.T
    H = np.eye(n) - np.ones((n, n))/n
    return np.trace((K @ H) @ (L @ H)) / (n-1)**2

def eval_auth_eer(X_raw, X_all, all_devs, all_cond, comps, unique_devs):
    if len(comps) < 2: return np.nan
    enrolled = {}
    for dev in unique_devs:
        mask = np.array([d == dev for d in all_devs[:len(X_raw)]])
        s = X_raw[mask][:, comps]
        if len(s) < 2: continue
        mean = s.mean(axis=0); cov = np.cov(s.T) + np.eye(len(comps))*1e-6
        enrolled[dev] = {'mean': mean, 'inv_cov': inv(cov)}
    auth_gen, auth_imp = [], []
    for i in range(len(X_all)):
        dev = all_devs[i]
        if dev not in enrolled: continue
        diff = X_all[i, comps] - enrolled[dev]['mean']
        auth_gen.append(np.sqrt(diff @ enrolled[dev]['inv_cov'] @ diff))
        other_devs = [d for d in unique_devs if d != dev and d in enrolled]
        if other_devs:
            other = rng.choice(other_devs)
            diff2 = X_all[i, comps] - enrolled[other]['mean']
            auth_imp.append(np.sqrt(diff2 @ enrolled[other]['inv_cov'] @ diff2))
    auth_gen = np.array(auth_gen); auth_imp = np.array(auth_imp)
    if len(auth_gen) < 2 or len(auth_imp) < 2: return np.nan
    thr = np.linspace(0, np.percentile(np.concatenate([auth_gen, auth_imp]), 99), 200)
    far = np.array([np.mean(auth_imp <= t) for t in thr])
    frr = np.array([np.mean(auth_gen > t) for t in thr])
    ei = np.argmin(np.abs(far - frr))
    return (far[ei] + frr[ei]) / 2

def eval_health_eer(X_raw, X_all, all_cond, comps):
    if len(comps) < 2: return np.nan
    bl_mean = X_raw[:, comps].mean(axis=0)
    bl_cov = np.cov(X_raw[:, comps].T) + np.eye(len(comps))*1e-6; bl_inv = inv(bl_cov)
    scores = np.array([np.sqrt((x[comps]-bl_mean) @ bl_inv @ (x[comps]-bl_mean)) for x in X_all])
    normal = scores[all_cond == 'baseline']; abnormal = scores[all_cond != 'baseline']
    if len(normal) < 2 or len(abnormal) < 2: return np.nan
    thr = np.linspace(0, np.percentile(scores, 99), 200)
    far = np.array([np.mean(normal > t) for t in thr])
    tar = np.array([np.mean(abnormal > t) for t in thr])
    ei = np.argmin(np.abs(far - (1-tar)))
    return (far[ei] + (1-tar[ei])) / 2

def eval_cond_deviations(X_raw, X_all, all_cond, comps):
    if len(comps) < 2: return {}
    bl_mean = X_raw[:, comps].mean(axis=0)
    bl_cov = np.cov(X_raw[:, comps].T) + np.eye(len(comps))*1e-6; bl_inv = inv(bl_cov)
    scores = np.array([np.sqrt((x[comps]-bl_mean) @ bl_inv @ (x[comps]-bl_mean)) for x in X_all])
    result = {}
    for ct in sorted(set(all_cond)):
        mask = all_cond == ct
        result[ct] = {'mean': np.mean(scores[mask]), 'std': np.std(scores[mask]), 'n': np.sum(mask)}
    return result

def evaluate_method(method_name, X_proj, n_raw, all_devs, all_cond, unique_devs, elapsed):
    """Evaluate any projection method. X_proj: projected scores, first n_raw = raw baseline."""
    n_pcs = X_proj.shape[1]; X_raw = X_proj[:n_raw]
    id_fs, hl_fs = compute_f_ratios(X_proj, all_devs, all_cond)
    id_s = norm01(id_fs); hl_s = norm01(hl_fs)
    id_sub, hl_sub, id_only, hl_only, dual, neither, gap = partition_by_ratio(id_s, hl_s)
    all_id = sorted(id_only); all_hl = sorted(hl_only)
    if len(all_id) < 2 and len(id_sub) >= 2: all_id = list(id_sub[:2])
    if len(all_hl) < 2 and len(hl_sub) >= 2: all_hl = list(hl_sub[:2])
    if len(all_id) < 2: all_id = list(range(2))
    if len(all_hl) < 2: all_hl = list(range(n_pcs-2, n_pcs))
    hsic_val = compute_hsic(X_proj[:, all_id], X_proj[:, all_hl])
    e1 = eval_auth_eer(X_raw, X_proj, all_devs, all_cond, all_id, unique_devs)
    e2 = eval_health_eer(X_raw, X_proj, all_cond, all_hl)
    hd = eval_cond_deviations(X_raw, X_proj, all_cond, all_hl)
    return {
        'method': method_name, 'n_id': len(all_id), 'n_hl': len(all_hl),
        'n_dual': len(dual), 'n_neither': len(neither),
        'hsic': hsic_val, 'auth_eer': e1, 'health_eer': e2, 'time': elapsed,
        **{f'dev_{k}': v['mean'] for k, v in hd.items()},
    }

# ============================================================================
# MAIN
# ============================================================================
print("=" * 60)
print("SUBSPACE SEPARATION METHOD COMPARISON")
print("=" * 60)

print("\n[1] Loading raw baseline sweeps...")
device_files = collect_device_files(DEVICE_FOLDER)
device_files = {k: v for k, v in device_files.items() if k not in EXCLUDED_DEVICES}
all_sweeps_raw = {}
for dev_id in sorted(device_files.keys()):
    all_sweeps_raw[dev_id] = {}
    for si in sorted(device_files[dev_id].keys()):
        try: all_sweeps_raw[dev_id][si] = load_sweep_vector(device_files[dev_id][si])
        except: pass
print(f"   {len(all_sweeps_raw)} devices, {sum(len(v) for v in all_sweeps_raw.values())} sweeps")

rx, rd, rs = [], [], []
for dev in sorted(all_sweeps_raw.keys()):
    for si in sorted(all_sweeps_raw[dev].keys()):
        rx.append(all_sweeps_raw[dev][si]); rd.append(dev); rs.append(si)
raw_X = np.array(rx); raw_devs = np.array(rd)
unique_devs = sorted(set(raw_devs))

print("\n[2] PCA on baseline sweeps...")
scaler = StandardScaler(); X_scaled = scaler.fit_transform(raw_X)
pca = PCA(); X_pca = pca.fit_transform(X_scaled)
print(f"   {X_pca.shape[1]} total PCs")

print("\n[3] Generating synthetic perturbations...")
engine = PerturbationEngine(PerturbationConfig(
    temperature_levels=[("cool", -10.0), ("warm", 15.0), ("hot", 30.0)],
    aging_levels=[("none", 0.0), ("mild", 0.5), ("moderate", 2.0), ("severe", 5.0)],
    load_levels=[("none", 0.0), ("light", 20.0), ("moderate", 60.0), ("heavy", 100.0)],
))
X_synth, synth_devs, _, meta_list = engine.generate_synthetic_dataset(all_sweeps_raw)
X_synth_pca = pca.transform(scaler.transform(X_synth))
cond_type = np.array([m['condition_type'] for m in meta_list])

X_all = np.vstack([X_pca, X_synth_pca])
all_devs = np.concatenate([raw_devs, synth_devs])
all_cond = np.array(['baseline']*len(raw_devs) + list(cond_type))
n_raw = len(raw_devs)
print(f"   {len(X_synth)} synthetic, {len(X_all)} total")

# ============================================================================
# 4. METHODS
# ============================================================================
results = []

print("\n[4a] PCA...")
t0 = time.time()
X_pcs = X_all[:, :N_PCS]
results.append(evaluate_method('PCA', X_pcs, n_raw, all_devs, all_cond, unique_devs, time.time()-t0))
r = results[-1]; print(f"   PCA: ID={r['n_id']}, HL={r['n_hl']}, Dual={r['n_dual']}, "
      f"HSIC={r['hsic']:.4f}, AuthEER={r['auth_eer']:.4f}, HlthEER={r['health_eer']:.4f}")

print("\n[4b] FastICA...")
t0 = time.time()
X_ica = FastICA(n_components=N_PCS, random_state=42, max_iter=1000, tol=1e-4).fit_transform(X_pcs)
results.append(evaluate_method('FastICA', X_ica, n_raw, all_devs, all_cond, unique_devs, time.time()-t0))
r = results[-1]; print(f"   ICA: ID={r['n_id']}, HL={r['n_hl']}, Dual={r['n_dual']}, "
      f"HSIC={r['hsic']:.4f}, AuthEER={r['auth_eer']:.4f}, HlthEER={r['health_eer']:.4f}")

print("\n[4c] PLS...")
t0 = time.time()
le_dev = LabelEncoder(); le_cond = LabelEncoder()
dev_oh = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
cond_oh = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
Y = np.hstack([dev_oh.fit_transform(le_dev.fit_transform(all_devs).reshape(-1,1)),
               cond_oh.fit_transform(le_cond.fit_transform(all_cond).reshape(-1,1))])
pls = PLSRegression(n_components=N_PCS, max_iter=500, tol=1e-4)
X_pls = pls.fit_transform(X_pcs, Y)[0]
results.append(evaluate_method('PLS', X_pls, n_raw, all_devs, all_cond, unique_devs, time.time()-t0))
r = results[-1]; print(f"   PLS: ID={r['n_id']}, HL={r['n_hl']}, Dual={r['n_dual']}, "
      f"HSIC={r['hsic']:.4f}, AuthEER={r['auth_eer']:.4f}, HlthEER={r['health_eer']:.4f}")

print("\n[4d] PCA (sep F-ratio: ID on raw, HL on synth)...")
t0 = time.time()
id_fs_raw, _ = compute_f_ratios(X_pca[:, :N_PCS], raw_devs, np.zeros(len(raw_devs)))
_, hl_fs_syn = compute_f_ratios(X_synth_pca[:, :N_PCS], synth_devs, cond_type)
id_s_sep = norm01(id_fs_raw); hl_s_sep = norm01(hl_fs_syn)
id_sub_sep, hl_sub_sep, id_only_sep, hl_only_sep, dual_sep, neither_sep, gap_sep = \
    partition_by_ratio(id_s_sep, hl_s_sep)
all_id = sorted(id_only_sep); all_hl = sorted(hl_only_sep)
if len(all_id) < 2 and len(id_sub_sep) >= 2: all_id = list(id_sub_sep[:2])
if len(all_hl) < 2 and len(hl_sub_sep) >= 2: all_hl = list(hl_sub_sep[:2])
if len(all_id) < 2: all_id = list(range(2))
if len(all_hl) < 2: all_hl = list(range(N_PCS-2, N_PCS))
hsic_sep = compute_hsic(X_pcs[:, all_id], X_pcs[:, all_hl])
e1_sep = eval_auth_eer(X_pca[:, :N_PCS], X_pcs, all_devs, all_cond, all_id, unique_devs)
e2_sep = eval_health_eer(X_pca[:, :N_PCS], X_pcs, all_cond, all_hl)
hd_sep = eval_cond_deviations(X_pca[:, :N_PCS], X_pcs, all_cond, all_hl)
res_sep = {'method': 'PCA (sep F-ratio)', 'n_id': len(all_id), 'n_hl': len(all_hl),
    'n_dual': len(dual_sep), 'n_neither': len(neither_sep),
    'hsic': hsic_sep, 'auth_eer': e1_sep, 'health_eer': e2_sep, 'time': time.time()-t0,
    **{f'dev_{k}': v['mean'] for k, v in hd_sep.items()}}
results.append(res_sep)
print(f"   PCA sep: ID={res_sep['n_id']}, HL={res_sep['n_hl']}, Dual={res_sep['n_dual']}, "
      f"HSIC={hsic_sep:.4f}, AuthEER={e1_sep:.4f}, HlthEER={e2_sep:.4f}")

# --- 4e. sisPCA ---
print("\n[4e] sisPCA...")
t0 = time.time()
N_SIS_PRE = 50  # pre-PCA dimension for sisPCA
N_ID_SIS = 10   # identity subspace dim
N_HL_SIS = 10   # health subspace dim
# Pre-reduce with PCA
pca_sis = PCA(n_components=N_SIS_PRE)
X_sis_reduced = pca_sis.fit_transform(scaler.transform(raw_X))
X_syn_sis = pca_sis.transform(scaler.transform(X_synth))
# Combined reduced data
X_sis_all = np.vstack([X_sis_reduced, X_syn_sis])
# Build supervision
le_dev_sis = LabelEncoder(); le_cond_sis = LabelEncoder()
dev_oh_sis = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
cond_oh_sis = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
dev_num_sis = le_dev_sis.fit_transform(all_devs)
cond_num_sis = le_cond_sis.fit_transform(all_cond)
Y_dev_sis = dev_oh_sis.fit_transform(dev_num_sis.reshape(-1, 1))
Y_cond_sis = cond_oh_sis.fit_transform(cond_num_sis.reshape(-1, 1))
# Use subset for training speed (50 devices, ~850 samples)
n_sis_devs = min(50, len(unique_devs))
sis_train_devs = set(unique_devs[:n_sis_devs])
sis_train_mask = np.array([d in sis_train_devs for d in all_devs])
# Training data
X_sis_train = X_sis_all[sis_train_mask]
Y_dev_train = Y_dev_sis[sis_train_mask]
Y_cond_train = Y_cond_sis[sis_train_mask]
# Create SISPCADataset
sis_data = SISPCADataset(
    data=torch.from_numpy(X_sis_train).float(),
    target_supervision_list=[
        Supervision(target_data=Y_dev_train, target_type='categorical', target_name='device_id'),
        Supervision(target_data=Y_cond_train, target_type='categorical', target_name='condition')
    ]
)
# Train with 3 lambda values
for lam_sis in [0.0, 1.0, 10.0]:
    print(f"   Training lam={lam_sis}...", end=" ", flush=True)
    try:
        model_sis = SISPCA(
            dataset=sis_data,
            n_latent_sub=[N_ID_SIS, N_HL_SIS],
            lambda_contrast=lam_sis,
            kernel_subspace='linear',
            solver='eig'
        )
        model_sis.fit(batch_size=len(X_sis_train), max_epochs=3, lr=1.0,
                      early_stopping_patience=None, enable_progress_bar=False,
                      enable_model_summary=False)
        # Project ALL data through learned U
        U_sis = model_sis.U.detach().numpy()
        X_sis_all_proj = X_sis_all @ U_sis  # (n_all, 20)
        Z_id_sis = X_sis_all_proj[:, :N_ID_SIS]
        Z_hl_sis = X_sis_all_proj[:, N_ID_SIS:]
        # Evaluate
        hsic_sis = compute_hsic(Z_id_sis, Z_hl_sis)
        e1_sis = eval_auth_eer(X_sis_reduced @ U_sis[:N_SIS_PRE, :N_ID_SIS],
                               X_sis_all_proj[:, :N_ID_SIS],
                               all_devs, all_cond, list(range(N_ID_SIS)), unique_devs)
        e2_sis = eval_health_eer(X_sis_reduced @ U_sis[:N_SIS_PRE, N_ID_SIS:],
                                 X_sis_all_proj[:, N_ID_SIS:],
                                 all_cond, list(range(N_HL_SIS)))
        hd_sis = eval_cond_deviations(X_sis_reduced @ U_sis[:N_SIS_PRE, N_ID_SIS:],
                                      X_sis_all_proj[:, N_ID_SIS:],
                                      all_cond, list(range(N_HL_SIS)))
        results.append({
            'method': f'sisPCA lam={lam_sis}',
            'n_id': N_ID_SIS, 'n_hl': N_HL_SIS,
            'n_dual': 'fixed', 'n_neither': 'fixed',
            'hsic': hsic_sis, 'auth_eer': e1_sis, 'health_eer': e2_sis,
            'time': time.time() - t0,
            **{f'dev_{k}': v['mean'] for k, v in hd_sis.items()},
        })
        print(f"HSIC={hsic_sis:.4f}, AuthEER={e1_sis:.4f}, HlthEER={e2_sis:.4f}")
    except Exception as e:
        print(f"FAILED: {e}")

# ============================================================================
# 5. RESULTS TABLE
# ============================================================================
print("\n" + "=" * 60)
print("COMPARISON RESULTS")
print("=" * 60)

df = pd.DataFrame(results)

print(f"\n{'Method':<22} {'ID':>4} {'HL':>4} {'Dual':>4} {'None':>4} "
      f"{'HSIC':>10} {'AuthEER':>8} {'HlthEER':>8} {'Time':>6}")
print("-" * 80)
for _, r in df.iterrows():
    def fmt(v, d=2):
        if isinstance(v, str): return v
        if np.isnan(v): return 'N/A'
        return f"{v:.{d}f}"
    print(f"{r['method']:<22} {fmt(r.get('n_id',0)):>4} {fmt(r.get('n_hl',0)):>4} "
          f"{fmt(r.get('n_dual',0)):>4} {fmt(r.get('n_neither',0)):>4} "
          f"{fmt(r['hsic'],4):>10} {fmt(r['auth_eer'],4):>8} {fmt(r['health_eer'],4):>8} "
          f"{fmt(r.get('time',0),1):>6}")

print(f"\n{'Method':<22} {'Baseline':>10} {'Temperature':>12} {'Aging':>10} {'Loading':>10}")
print("-" * 65)
for _, r in df.iterrows():
    print(f"{r['method']:<22} {fmt(r.get('dev_baseline',np.nan),2):>10} "
          f"{fmt(r.get('dev_temperature',np.nan),2):>12} {fmt(r.get('dev_aging',np.nan),2):>10} "
          f"{fmt(r.get('dev_loading',np.nan),2):>10}")

# Figure
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
mnames = list(df['method'])
ax = axes[0]
hs = [r['hsic'] if isinstance(r['hsic'],(int,float)) and not np.isnan(r['hsic']) else 0 for _,r in df.iterrows()]
cs = ['steelblue','salmon','green','gold','purple'][:len(mnames)]
ax.bar(mnames, hs, color=cs, alpha=0.8, edgecolor='black')
ax.set_ylabel('HSIC (lower = better)'); ax.set_title('Subspace Independence')
ax.tick_params(axis='x', rotation=15)

ax = axes[1]
ae = [r['auth_eer'] if isinstance(r['auth_eer'],(int,float)) and not np.isnan(r['auth_eer']) else 0 for _,r in df.iterrows()]
he = [r['health_eer'] if isinstance(r['health_eer'],(int,float)) and not np.isnan(r['health_eer']) else 0 for _,r in df.iterrows()]
x = np.arange(len(mnames)); w = 0.35
ax.bar(x-w/2, ae, w, label='Auth EER', color='steelblue', alpha=0.8, edgecolor='black')
ax.bar(x+w/2, he, w, label='Health EER', color='salmon', alpha=0.8, edgecolor='black')
ax.set_xticks(x); ax.set_xticklabels(mnames, rotation=15)
ax.set_ylabel('EER'); ax.set_title('Authentication vs Health Error'); ax.legend(fontsize=8)

ax = axes[2]
for idx, ct in enumerate(['dev_baseline','dev_temperature','dev_aging','dev_loading']):
    vals = [r.get(ct, np.nan) if isinstance(r.get(ct),(int,float)) and not np.isnan(r.get(ct,np.nan)) else 0 for _,r in df.iterrows()]
    ax.bar(x+(idx-1.5)*0.2, vals, 0.2, label=ct.replace('dev_',''), alpha=0.8, edgecolor='black')
ax.set_xticks(x); ax.set_xticklabels(mnames, rotation=15)
ax.set_ylabel('Mahalanobis Distance'); ax.set_title('Per-Condition Health Deviation'); ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, 'subspace_method_comparison.png'), dpi=150); plt.close()
print(f"\n[V] Figure -> subspace_method_comparison.png")

csv_path = os.path.join(REPORT_DIR, 'subspace_method_comparison.csv')
df.to_csv(csv_path, index=False)
print(f"[V] Table -> subspace_method_comparison.csv")
print("\nDone.")

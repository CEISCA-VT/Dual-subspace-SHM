"""
PAMELA baseline comparison: PCA, FastICA, PLS, sisPCA on standalone PAMELA dataset.
Temperature = identity, damage = condition. Only 30 sweeps — sample-size floor at ~16.7%.
"""
import os, sys, warnings, time
warnings.filterwarnings('ignore')
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
from sispca.data import Supervision, SISPCADataset
from sispca.model import SISPCA

rng = np.random.default_rng(42)
REPORT_DIR = r"./rq_analysis_reports"
os.makedirs(REPORT_DIR, exist_ok=True)
sns.set_style("whitegrid")
plt.rcParams.update({'figure.max_open_warning': 0})

# ======== DATA LOADING (from pamela_analysis.py) ========
print("=" * 60)
print("PAMELA BASELINE COMPARISON: PCA / ICA / PLS / sisPCA")
print("=" * 60)

DATA_DIR = r"./ImpedanceData"
N_FREQ = 2000
REF_FREQ = np.linspace(0, 125000, N_FREQ)

X_list, dev_list, cond_list, meta_list = [], [], [], []
temps = [24, 40, 55, 70, 85, 100]
for temp in temps:
    h = sio.loadmat(f'{DATA_DIR}/HealthyCondition/EMI{temp}H.mat')
    vn = [k for k in h if k.startswith('freq_')][0]
    freq = h[vn].ravel()
    real = h[f'real_{temp}H'].ravel()
    imag = h[f'imag_{temp}H'].ravel()
    mag = np.sqrt(real**2 + imag**2)
    phase = np.degrees(np.arctan2(imag, real))
    mag_i = np.interp(REF_FREQ, freq, mag)
    phase_i = np.interp(REF_FREQ, freq, phase)
    vec = np.concatenate([phase_i, mag_i])
    X_list.append(vec); dev_list.append(f'T{temp}')
    cond_list.append('healthy')
    meta_list.append({'temp': temp, 'damage': 0, 'condition': 'healthy'})
    for dl in range(1, 5):
        fn = f'{DATA_DIR}/DamagedCondition/T{temp}degrees/EMI{temp}D{dl}.mat'
        if not os.path.exists(fn): continue
        d = sio.loadmat(fn)
        real = d[f'real_{temp}D{dl}'].ravel()
        imag = d[f'imag_{temp}D{dl}'].ravel()
        mag = np.sqrt(real**2 + imag**2)
        phase = np.degrees(np.arctan2(imag, real))
        mag_i = np.interp(REF_FREQ, freq, mag)
        phase_i = np.interp(REF_FREQ, freq, phase)
        vec = np.concatenate([phase_i, mag_i])
        X_list.append(vec); dev_list.append(f'T{temp}')
        cond_list.append(f'damage_D{dl}')
        meta_list.append({'temp': temp, 'damage': dl, 'condition': 'damage'})

X_raw = np.array(X_list); dev_labels = np.array(dev_list); cond_labels = np.array(cond_list)
print(f"   {len(X_raw)} samples, {X_raw.shape[1]} features")
unique_devs = sorted(set(dev_labels))
all_cond = np.array(['baseline' if m['condition']=='healthy' else 'damage' for m in meta_list])
all_cond_full = np.array([m['condition'] for m in meta_list])
print(f"   {sum(all_cond=='baseline')} healthy, {sum(all_cond!='baseline')} damaged ({len(temps)} temps x {4} dmg levels)")

# Standardize + PCA pre-reduction
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
N_PRE = min(15, len(X_raw)-1)
pca_pre = PCA(n_components=N_PRE)
X_pca = pca_pre.fit_transform(X_scaled)
print(f"   Reduced to {N_PRE} PCs (var={pca_pre.explained_variance_ratio_.sum():.4f})")

# Supervision
le_dev = LabelEncoder(); le_cond = LabelEncoder()
dev_oh = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
cond_oh = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
dev_num = le_dev.fit_transform(dev_labels)
cond_num = le_cond.fit_transform(all_cond)
Y_dev = dev_oh.fit_transform(dev_num.reshape(-1, 1))
Y_cond = cond_oh.fit_transform(cond_num.reshape(-1, 1))

sis_data = SISPCADataset(
    data=torch.from_numpy(X_pca).float(),
    target_supervision_list=[
        Supervision(target_data=Y_dev, target_type='categorical', target_name='temperature'),
        Supervision(target_data=Y_cond, target_type='categorical', target_name='damage')
    ]
)

normal_mask = all_cond == 'baseline'
abnormal_mask = all_cond != 'baseline'
n_raw = len(X_pca)

# ======== HELPERS ========
def mahalanobis_eer(X_raw, X_all, comps, normal_mask, abnormal_mask):
    if len(comps) < 2: return np.nan
    bl_mean = X_raw[normal_mask][:, comps].mean(axis=0)
    bl_cov = np.cov(X_raw[normal_mask][:, comps].T) + np.eye(len(comps))*1e-6
    bl_inv = inv(bl_cov)
    scores = np.array([np.sqrt((x[comps]-bl_mean) @ bl_inv @ (x[comps]-bl_mean)) for x in X_all])
    nrm = scores[normal_mask]; abn = scores[abnormal_mask]
    if len(nrm) < 2 or len(abn) < 2: return np.nan
    thr = np.linspace(0, np.percentile(scores, 99), 200)
    far = np.array([np.mean(nrm > t) for t in thr])
    tar = np.array([np.mean(abn > t) for t in thr])
    ei = np.argmin(np.abs(far - (1-tar)))
    return (far[ei] + (1-tar[ei])) / 2

def compute_hsic(X, Y):
    n = len(X); K = X @ X.T; L = Y @ Y.T
    H = np.eye(n) - np.ones((n,n))/n
    return np.trace(K @ H @ L @ H) / (n-1)**2

def per_damage_level_eer(X_raw, X_all, comps, meta_list):
    levels = sorted(set(m['damage'] for m in meta_list if m['damage'] > 0))
    res = {}
    for dl in levels:
        d_mask = np.array([m['damage']==dl for m in meta_list])
        n_mask = np.array([m['damage']==0 for m in meta_list])
        if sum(n_mask) < 2 or sum(d_mask) < 2: res[dl] = np.nan
        else: res[dl] = mahalanobis_eer(X_raw, X_all, comps, n_mask, d_mask)
    return res

def per_damage_level_eer_by_temp(X_raw, X_all, comps, meta_list):
    results = {}
    temps = sorted(set(m['temp'] for m in meta_list))
    for dl in [1,2,3,4]:
        eers = []
        for t in temps:
            bl_mask = np.array([m['damage']==0 and m['temp']!=t for m in meta_list])
            ab_mask = np.array([m['damage']==dl and m['temp']==t for m in meta_list])
            if sum(bl_mask) < 2 or sum(ab_mask) < 1: continue
            if len(comps) < 2: continue
            bl_mean = X_raw[bl_mask][:, comps].mean(axis=0)
            bl_cov = np.cov(X_raw[bl_mask][:, comps].T) + np.eye(len(comps))*1e-6
            bl_inv = inv(bl_cov)
            scores = np.array([np.sqrt((x[comps]-bl_mean) @ bl_inv @ (x[comps]-bl_mean)) for x in X_all])
            normal = scores[bl_mask]
            for ab in scores[ab_mask]:
                eers.append(np.mean(normal > ab))
        results[f'D{dl}'] = np.mean(eers) if eers else np.nan
    return results

def compute_f_ratios(X, id_labels, cond_labels, n_pcs=None):
    if n_pcs is None: n_pcs = X.shape[1]
    id_scores, cond_scores = np.zeros(n_pcs), np.zeros(n_pcs)
    for pi in range(n_pcs):
        vals = X[:, pi]
        ss_tot = np.var(vals) * len(vals)
        ss_id = sum(len(vals[id_labels == u]) * (np.mean(vals[id_labels == u]) - np.mean(vals))**2 for u in set(id_labels))
        ss_cond = sum(len(vals[cond_labels == c]) * (np.mean(vals[cond_labels == c]) - np.mean(vals))**2 for c in set(cond_labels))
        id_scores[pi] = ss_id / max(ss_tot - ss_id, 1e-10)
        cond_scores[pi] = ss_cond / max(ss_tot - ss_cond, 1e-10)
    return id_scores, cond_scores

def partition_by_ratio(id_scores, cond_scores):
    ratio = np.where(cond_scores > 0, id_scores / (id_scores + cond_scores + 1e-10), 0.5)
    id_only = np.where(ratio > 0.7)[0]; hl_only = np.where(ratio < 0.3)[0]
    id_sub = np.where(ratio > 0.5)[0]; hl_sub = np.where(ratio < 0.5)[0]
    dual = np.where((ratio >= 0.3) & (ratio <= 0.7))[0]
    neither = np.where((id_scores == 0) & (cond_scores == 0))[0]
    return id_sub, hl_sub, id_only, hl_only, dual, neither, 0

def norm01(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / max(hi - lo, 1e-10)

def evaluate_method(name, X_proj, n_id, n_hl):
    id_comps = list(range(n_id)); hl_comps = list(range(n_id, n_id+n_hl))
    hsic_val = compute_hsic(X_proj[:, id_comps], X_proj[:, hl_comps]) if n_id and n_hl else np.nan
    health_eer = mahalanobis_eer(X_proj, X_proj, hl_comps, normal_mask, abnormal_mask)
    dl_eer = per_damage_level_eer(X_proj, X_proj, hl_comps, meta_list)
    dl_xval = per_damage_level_eer_by_temp(X_proj, X_proj, hl_comps, meta_list)
    res = {'method': name, 'hsic': hsic_val, 'n_id': n_id, 'n_hl': n_hl, 'health_eer': health_eer}
    for dl, v in dl_eer.items(): res[f'health_eer_D{dl}'] = v
    for k, v in dl_xval.items(): res[f'health_eer_{k}_xval'] = v
    return res

# ======== METHODS ========
results = []
mc = X_pca.shape[1]  # rank constraint = number of PCA dims

# [1] PCA — F-ratio split on full PCA space
print(f"\n[1] PCA (F-ratio split)...")
id_fs, hl_fs = compute_f_ratios(X_pca, dev_labels, all_cond)
id_s, hl_s = norm01(id_fs), norm01(hl_fs)
_, _, id_only, hl_only, _, _, _ = partition_by_ratio(id_s, hl_s)
n_id = max(2, len(id_only))
n_hl = max(2, len(hl_only))
while n_id + n_hl > mc:  # respect rank
    if n_hl > 2: n_hl -= 1
    else: n_id -= 1
results.append(evaluate_method('PCA (F-ratio)', X_pca, n_id, n_hl))
r = results[-1]
print(f"   ID={n_id}, HL={n_hl}, HlthEER={r['health_eer']:.4f}, HSIC={r['hsic']:.4f}")

# [2] PCA — hard split (first n_hl as HL, rest as ID)
print(f"[2] PCA (hard split)...")
n_hl2 = min(4, X_pca.shape[1] - 2); n_id2 = X_pca.shape[1] - n_hl2
results.append(evaluate_method('PCA (hard split)', X_pca, n_id2, n_hl2))
r = results[-1]
print(f"   ID={n_id2}, HL={n_hl2}, HlthEER={r['health_eer']:.4f}")

# [3] PCA — sep F-ratio: ID on raw, HL on synth? No synth here.
# Use same F-ratio as [1] but force HL=4, ID=mc-4
print(f"[3] PCA (split 4 HL)...")
n_hl3 = min(4, mc-2); n_id3 = mc - n_hl3
results.append(evaluate_method('PCA (4 HL)', X_pca, n_id3, n_hl3))
r = results[-1]
print(f"   ID={n_id3}, HL={n_hl3}, HlthEER={r['health_eer']:.4f}")

# [4] FastICA
print(f"[4] FastICA...")
t0 = time.time()
n_ica = min(mc - 1, X_pca.shape[1])
X_ica = FastICA(n_components=n_ica, random_state=42, max_iter=2000, tol=1e-4).fit_transform(X_pca)
id_fs_i, hl_fs_i = compute_f_ratios(X_ica, dev_labels, all_cond)
id_s_i, hl_s_i = norm01(id_fs_i), norm01(hl_fs_i)
_, _, id_only_i, hl_only_i, _, _, _ = partition_by_ratio(id_s_i, hl_s_i)
n_id_i = max(2, len(id_only_i))
n_hl_i = max(2, len(hl_only_i))
while n_id_i + n_hl_i > mc: n_hl_i -= 1
results.append(evaluate_method('FastICA', X_ica, n_id_i, n_hl_i))
r = results[-1]; et = time.time()-t0
print(f"   ID={n_id_i}, HL={n_hl_i}, HlthEER={r['health_eer']:.4f} ({et:.1f}s)")

# [5] PLS
print(f"[5] PLS...")
t0 = time.time()
tgt = np.hstack([Y_dev, Y_cond])
n_pls = min(mc - 1, X_pca.shape[1], tgt.shape[1]-1)
pls = PLSRegression(n_components=n_pls, max_iter=500, tol=1e-4)
X_pls = pls.fit_transform(X_pca, tgt)[0]
id_fs_p, hl_fs_p = compute_f_ratios(X_pls, dev_labels, all_cond)
id_s_p, hl_s_p = norm01(id_fs_p), norm01(hl_fs_p)
_, _, id_only_p, hl_only_p, _, _, _ = partition_by_ratio(id_s_p, hl_s_p)
n_id_p = max(2, len(id_only_p))
n_hl_p = max(2, len(hl_only_p))
while n_id_p + n_hl_p > mc: n_hl_p -= 1
results.append(evaluate_method('PLS', X_pls, n_id_p, n_hl_p))
r = results[-1]; et = time.time()-t0
print(f"   ID={n_id_p}, HL={n_hl_p}, HlthEER={r['health_eer']:.4f} ({et:.1f}s)")

# [6] sisPCA
print(f"\n[6] sisPCA...")
N_ID_SIS = 4; N_HL_SIS = 6
for lam in [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0]:
    t0 = time.time()
    try:
        model = SISPCA(dataset=sis_data, n_latent_sub=[N_ID_SIS, N_HL_SIS],
                       lambda_contrast=lam, kernel_subspace='linear', solver='eig')
        model.fit(batch_size=len(X_pca), max_epochs=3, lr=1.0,
                  early_stopping_patience=None, enable_progress_bar=False,
                  enable_model_summary=False)
        U = model.U.detach().numpy()
        X_proj = X_pca @ U
        res = evaluate_method(f'sisPCA lam={lam}', X_proj, N_ID_SIS, N_HL_SIS)
        results.append(res); et = time.time()-t0
        print(f"   lam={lam}: HlthEER={res['health_eer']:.4f}, HSIC={res['hsic']:.4f} ({et:.1f}s)")
    except Exception as e:
        print(f"   lam={lam}: FAILED — {e}")

# ======== RESULTS ========
print("\n" + "=" * 60)
print("PAMELA BASELINE COMPARISON RESULTS")
print("=" * 60)

df = pd.DataFrame(results)
def fmt(v, d=4):
    if isinstance(v, str): return v
    if np.isnan(v): return 'N/A'
    return f"{v:.{d}f}"

# Sort: non-sisPCA first, then sisPCA by λ
order = ['PCA (F-ratio)', 'PCA (hard split)', 'PCA (4 HL)', 'FastICA', 'PLS'] + [f'sisPCA lam={lam}' for lam in [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0]]
df['_order'] = df['method'].apply(lambda x: order.index(x) if x in order else 99)
dfs = df.sort_values('_order').drop(columns='_order')

print(f"\n{'Method':<20} {'HSIC':>10} {'N-ID':>5} {'N-HL':>5} {'HlthEER':>8} ", end='')
for dl in range(1,5): print(f'D{dl}={fmt(0,4):>8}', end='')
print(f"  ", end='')
for dl in range(1,5): print(f'D{dl}xv={fmt(0,4):>8}', end='')
print()
print("-" * (20 + 10 + 5 + 5 + 8 + 8*4 + 10*4))
for _, r in dfs.iterrows():
    print(f"{r['method']:<20} {fmt(r['hsic'],2):>10} {int(r['n_id']):>5} {int(r['n_hl']):>5} {fmt(r['health_eer'],4):>8} ", end='')
    for dl in range(1,5): print(f'{fmt(r.get(f"health_eer_D{dl}",np.nan),4):>8}', end='')
    print(f"  ", end='')
    for dl in range(1,5): print(f'{fmt(r.get(f"health_eer_D{dl}_xval",np.nan),4):>8}', end='')
    print()

# Figure
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
mnames = list(dfs['method'])
palette = ['steelblue','royalblue','navy','green','orange','salmon','firebrick','darkred','purple','brown','pink','gold','teal']
cs = palette[:len(mnames)]

ax = axes[0]
hs = [r['hsic'] if isinstance(r['hsic'],(int,float)) and not np.isnan(r['hsic']) else 0 for _,r in dfs.iterrows()]
ax.bar(mnames, hs, color=cs, alpha=0.8, edgecolor='black')
ax.set_ylabel('HSIC'); ax.set_title('Temp↔Damage Independence')
ax.tick_params(axis='x', rotation=20)

ax = axes[1]
he = [r['health_eer'] if isinstance(r['health_eer'],(int,float)) and not np.isnan(r['health_eer']) else 0 for _,r in dfs.iterrows()]
ax.bar(mnames, he, color=cs, alpha=0.8, edgecolor='black')
ax.set_ylabel('Health EER'); ax.set_title('Damage Detection (overall)')
ax.tick_params(axis='x', rotation=20)

ax = axes[2]
x = np.arange(len(mnames)); w = 0.18
for idx, dl in enumerate(range(1,5)):
    vals = [r.get(f'health_eer_D{dl}',np.nan) if isinstance(r.get(f'health_eer_D{dl}'),(int,float)) and not np.isnan(r.get(f'health_eer_D{dl}',np.nan)) else 0 for _,r in dfs.iterrows()]
    ax.bar(x+(idx-1.5)*w, vals, w, label=f'D{dl}', alpha=0.8, edgecolor='black')
ax.set_xticks(x); ax.set_xticklabels(mnames, rotation=20)
ax.set_ylabel('EER'); ax.set_title('Per-Damage-Level EER'); ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, 'pamela_baseline_comparison.png'), dpi=150); plt.close()
dfs.to_csv(os.path.join(REPORT_DIR, 'pamela_baseline_comparison.csv'), index=False)
print(f"\n[V] CSV + Figure saved to {REPORT_DIR}")
print("Done.")

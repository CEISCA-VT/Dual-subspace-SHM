"""
sisPCA on PAMELA dataset: separate temperature (env confound) from damage (health).
"""
import os, sys, warnings, time
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from scipy.linalg import inv
import scipy.io as sio
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sispca.data import Supervision, SISPCADataset
from sispca.model import SISPCA

REPORT_DIR = r"./rq_analysis_reports"
os.makedirs(REPORT_DIR, exist_ok=True)
sns.set_style("whitegrid")
plt.rcParams.update({'figure.max_open_warning': 0})
rng = np.random.default_rng(42)

# Load all PAMELA .mat files
DATA_DIR = r"./ImpedanceData"
N_FREQ = 2000  # subsample from 32768
REF_FREQ = np.linspace(0, 125000, N_FREQ)

def load_pamela():
    X_list, dev_list, cond_list, meta_list = [], [], [], []
    temps = [24, 40, 55, 70, 85, 100]
    dlevels = {'H': 0, 'D1': 1, 'D2': 2, 'D3': 3, 'D4': 4}
    
    for temp in temps:
        # Healthy
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
        X_list.append(vec)
        dev_list.append(f'T{temp}')
        cond_list.append('healthy')
        meta_list.append({'temp': temp, 'damage': 0, 'condition': 'healthy'})
        
        # Damaged D1-D4
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
            X_list.append(vec)
            dev_list.append(f'T{temp}')
            cond_list.append(f'damage_D{dl}')
            meta_list.append({'temp': temp, 'damage': dl, 'condition': 'damage'})
    
    return np.array(X_list), np.array(dev_list), np.array(cond_list), meta_list

print("=" * 60)
print("PAMELA — sisPCA: Temperature vs Damage Separation")
print("=" * 60)

print("\n[1] Loading PAMELA data...")
X_raw, dev_labels, cond_labels, meta = load_pamela()
print(f"   {len(X_raw)} samples, {4000} features (phase[2000]+mag[2000])")
print(f"   Temperatures: {sorted(set(dev_labels))}")
print(f"   Conditions: {sorted(set(cond_labels))}")

unique_devs = sorted(set(dev_labels))
all_cond = np.array(['baseline' if m['condition']=='healthy' else 'damage' for m in meta])
all_tags = np.array(cond_labels)
all_cond_full = np.array([m['condition'] for m in meta])

print(f"\n   Baseline (healthy): {sum(all_cond=='baseline')}  Abnormal (damaged): {sum(all_cond!='baseline')}")

print("\n[2] Standardize + PCA pre-reduction...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
N_PRE = min(15, len(X_raw)-1)  # rank constraint
pca = PCA(n_components=N_PRE)
X_pca = pca.fit_transform(X_scaled)
print(f"   Reduced to {N_PRE} PCs (var={pca.explained_variance_ratio_.sum():.4f})")

# Encode dev & cond
le_dev = LabelEncoder(); le_cond = LabelEncoder()
dev_oh = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
cond_oh = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
dev_num = le_dev.fit_transform(dev_labels)
cond_num = le_cond.fit_transform(all_cond)
Y_dev = dev_oh.fit_transform(dev_num.reshape(-1, 1))
Y_cond = cond_oh.fit_transform(cond_num.reshape(-1, 1))

# sisPCA dataset
sis_data = SISPCADataset(
    data=torch.from_numpy(X_pca).float(),
    target_supervision_list=[
        Supervision(target_data=Y_dev, target_type='categorical', target_name='temperature'),
        Supervision(target_data=Y_cond, target_type='categorical', target_name='damage')
    ]
)

# -----------------------------------------------------------------------
# Helper: Mahalanobis EER
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

def per_damage_level_eer(X_raw, X_all, comps, meta_list):
    """Damage EER per severity across all temperatures combined."""
    levels = sorted(set(m['damage'] for m in meta_list if m['damage'] > 0))
    results = {}
    for dl in levels:
        d_mask = np.array([m['damage']==dl for m in meta_list])
        n_mask = np.array([m['damage']==0 for m in meta_list])
        if sum(n_mask) < 2 or sum(d_mask) < 2:
            results[dl] = np.nan
        else:
            results[dl] = mahalanobis_eer(X_raw, X_all, comps, n_mask, d_mask)
    return results

def per_damage_level_eer_by_temp(X_raw, X_all, comps, meta_list):
    """Damage EER per severity evaluated at each temp using other temps as baseline."""
    results = {}
    temps = sorted(set(m['temp'] for m in meta_list))
    for dl in [1,2,3,4]:
        key = f'D{dl}'
        eers = []
        for t in temps:
            # Baseline: healthy at ALL temperatures EXCEPT t
            bl_mask = np.array([m['damage']==0 and m['temp']!=t for m in meta_list])
            # Abnormal: damage level dl at temperature t
            ab_mask = np.array([m['damage']==dl and m['temp']==t for m in meta_list])
            if sum(bl_mask) < 2 or sum(ab_mask) < 1:
                continue
            scores = np.zeros(len(X_all))
            if len(comps) < 2: continue
            bl_mean = X_raw[bl_mask][:, comps].mean(axis=0)
            bl_cov = np.cov(X_raw[bl_mask][:, comps].T) + np.eye(len(comps))*1e-6
            bl_inv = inv(bl_cov)
            scores = np.array([np.sqrt((x[comps]-bl_mean) @ bl_inv @ (x[comps]-bl_mean)) for x in X_all])
            normal = scores[bl_mask]
            for ab in scores[ab_mask]:
                eers.append(np.mean(normal > ab))
        results[key] = np.mean(eers) if eers else np.nan
    return results

# -----------------------------------------------------------------------
print("\n[3] sisPCA training (lambda sweep)...")
N_ID = 4   # temperature subspace
N_HL = 6   # damage subspace
n_raw = len(X_pca)

results = []
normal_mask = all_cond == 'baseline'
abnormal_mask = all_cond != 'baseline'

for lam in [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0]:
    print(f"\n   lam={lam}...", end=" ", flush=True)
    try:
        model = SISPCA(
            dataset=sis_data,
            n_latent_sub=[N_ID, N_HL],
            lambda_contrast=lam,
            kernel_subspace='linear',
            solver='eig'
        )
        model.fit(batch_size=len(X_pca), max_epochs=3, lr=1.0,
                  early_stopping_patience=None, enable_progress_bar=False,
                  enable_model_summary=False)
        
        U = model.U.detach().numpy()  # (N_PRE, N_ID+N_HL)
        X_proj = X_pca @ U
        
        # Hard split
        Z_id = X_proj[:, :N_ID]; Z_hl = X_proj[:, N_ID:]
        hsic_val = compute_hsic(Z_id, Z_hl)
        
        # Health EER (damage detection in health subspace)
        health_eer = mahalanobis_eer(X_pca, X_proj, list(range(N_ID, N_ID+N_HL)),
                                      normal_mask, abnormal_mask)
        
        # Per-damage-level EER (across all temps)
        hl_comps = list(range(N_ID, N_ID+N_HL))
        dl_eer = per_damage_level_eer(X_pca, X_proj, hl_comps, meta)
        dl_eer_xval = per_damage_level_eer_by_temp(X_pca, X_proj, hl_comps, meta)
        
        res = {
            'method': f'sisPCA lam={lam}', 'n_temp': N_ID, 'n_damage': N_HL,
            'hsic': hsic_val, 'health_eer': health_eer,
        }
        for dl, v in dl_eer.items():
            res[f'health_eer_D{dl}'] = v
        for dlk, v in dl_eer_xval.items():
            res[f'health_eer_{dlk}_xval'] = v
        results.append(res)
        
        dl_str = ', '.join(f'D{dl}={dl_eer[dl]:.4f}' for dl in sorted(dl_eer.keys()))
        xv_str = ', '.join(f'{k}={v:.4f}' for k,v in dl_eer_xval.items() if not np.isnan(v))
        print(f"HSIC={hsic_val:.4f}, HlthEER={health_eer:.4f}, "
              f"ByDmg: {dl_str}, Xval: {xv_str}")
    
    except Exception as e:
        print(f"FAILED: {e}")

# -----------------------------------------------------------------------
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
df = pd.DataFrame(results)

def fmt(v, d=4):
    if isinstance(v, str): return v
    if np.isnan(v): return 'N/A'
    return f"{v:.{d}f}"

print(f"\n{'Method':<20} {'HSIC':>10} {'HlthEER':>10} ", end='')
for dl in range(1,5): print(f'D{dl}={fmt(0,4):>8}', end='')
print(f"  ", end='')
for dl in range(1,5): print(f'D{dl}xv={fmt(0,4):>8}', end='')
print()
print("-" * (20 + 10 + 10 + 8*4 + 10*4))

for _, r in df.iterrows():
    print(f"{r['method']:<20} {fmt(r['hsic'],4):>10} {fmt(r['health_eer'],4):>10} ", end='')
    for dl in range(1,5):
        print(f'{fmt(r.get(f"health_eer_D{dl}",np.nan),4):>8}', end='')
    print(f"  ", end='')
    for dl in range(1,5):
        print(f'{fmt(r.get(f"health_eer_D{dl}_xval",np.nan),4):>8}', end='')
    print()

# Save CSV
csv_path = os.path.join(REPORT_DIR, 'pamela_results.csv')
df.to_csv(csv_path, index=False)
print(f"\n[V] CSV -> {csv_path}")

# Figure
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
mnames = list(df['method'])
palette = ['steelblue','salmon','green','gold','purple','cyan','magenta']
cs = palette[:len(mnames)]

ax = axes[0]
hs = [r['hsic'] if isinstance(r['hsic'],(int,float)) and not np.isnan(r['hsic']) else 0 for _,r in df.iterrows()]
ax.bar(mnames, hs, color=cs, alpha=0.8, edgecolor='black')
ax.set_ylabel('HSIC (lower = better)'); ax.set_title('Temp↔Damage Independence')
ax.tick_params(axis='x', rotation=15)

ax = axes[1]
he = [r['health_eer'] if isinstance(r['health_eer'],(int,float)) and not np.isnan(r['health_eer']) else 0 for _,r in df.iterrows()]
ax.bar(mnames, he, color=cs, alpha=0.8, edgecolor='black')
ax.set_ylabel('Health EER'); ax.set_title('Damage Detection (overall)')
ax.tick_params(axis='x', rotation=15)

ax = axes[2]
x = np.arange(len(mnames)); w = 0.18
for idx, dl in enumerate(range(1,5)):
    vals = [r.get(f'health_eer_D{dl}',np.nan) if isinstance(r.get(f'health_eer_D{dl}'),(int,float)) and not np.isnan(r.get(f'health_eer_D{dl}',np.nan)) else 0 for _,r in df.iterrows()]
    ax.bar(x+(idx-1.5)*w, vals, w, label=f'D{dl}', alpha=0.8, edgecolor='black')
ax.set_xticks(x); ax.set_xticklabels(mnames, rotation=15)
ax.set_ylabel('EER'); ax.set_title('Per-Damage-Level EER'); ax.legend(fontsize=8)

plt.tight_layout()
fig_path = os.path.join(REPORT_DIR, 'pamela_results.png')
plt.savefig(fig_path, dpi=150); plt.close()
print(f"[V] Figure -> {fig_path}")
print("\nDone.")

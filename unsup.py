"""
FINAL.py

Single vs Multi Sweep Autoencoder (Parametric t-SNE) Authentication
-------------------------------------------------------------------

Generates IDs using a PyTorch Autoencoder from device sweep CSVs.
Performs authentication, computes success %, Hamming distances, and writes a
Final_Results.md with embedded plots and explanations.

Outputs in REPORT_DIR:
 - single_sweep_metrics.csv
 - multi_sweep_metrics.csv
 - single_combined.png
 - multi_combined.png
 - comparison_summary.csv
 - device_debug_metrics.csv
 - Final_Results.md
"""

import os
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------
# CONFIG
# ---------------------------
DEVICE_FOLDER = r"./01_master_dataset"    # folder with CSV sweeps
REPORT_DIR    = r"./01_256reports"        # folder for outputs

PREFERRED_REG_INDEX_SINGLE = 1
PREFERRED_AUTH_INDEX_SINGLE = 2
PREFERRED_MULTI_TRAIN_INDICES = list(range(4, 7))
PREFERRED_MULTI_AUTH_INDEX = 5

USE_PHASE     = True
ID_BIT_LENGTH = 128
START_FREQ    = 10000
END_FREQ      = 1000000
N_FREQ_POINTS = 2001
REF_FREQ      = np.linspace(START_FREQ, END_FREQ, N_FREQ_POINTS)

# ---------------------------
# CUSTOM CONTROLS
# ---------------------------
EXCLUDED_DEVICES = ["201", "253", "254", "258", "310"]          
HAMMING_AUTH_THRESHOLD = 25    

os.makedirs(REPORT_DIR, exist_ok=True)

# ---------------------------
# PYTORCH AUTOENCODER ARCHITECTURE
# ---------------------------
class TSNEEncoder(nn.Module):
    def __init__(self, input_features, bit_length):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_features, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, bit_length) 
        )

    def forward(self, x):
        return self.net(x)

def compute_joint_probabilities(X, target_perplexity=30.0, tol=1e-5, max_iter=50):
    num_samples = X.shape[0]
    perplexity = min(target_perplexity, num_samples - 2)
    if perplexity < 1: 
        perplexity = 1
        
    sum_X = torch.sum(X*X, dim=1, keepdim=True)
    D = sum_X + sum_X.T - 2 * torch.mm(X, X.T)
    D = torch.clamp(D, min=0.0)
    
    target_entropy = np.log(perplexity)
    P = torch.zeros((num_samples, num_samples), dtype=torch.float32)
    
    for i in range(num_samples):
        beta_min, beta_max, beta = -float('inf'), float('inf'), 1.0  
        dists = D[i, :]
        mask = torch.ones_like(dists)
        mask[i] = 0 
        
        for _ in range(max_iter):
            shifted_dists = dists - torch.max(dists * mask)
            exps = torch.exp(-shifted_dists * beta) * mask
            sum_exps = torch.sum(exps) + 1e-8
            
            prob = exps / sum_exps
            entropy = -torch.sum(prob * torch.log(prob + 1e-8))
            
            if torch.abs(entropy - target_entropy) < tol: break
            if entropy > target_entropy:
                beta_min = beta
                beta = beta * 2 if beta_max == float('inf') else (beta + beta_max) / 2
            else:
                beta_max = beta
                beta = beta / 2 if beta_min == -float('inf') else (beta + beta_min) / 2
                
        P[i, :] = prob
        
    P = (P + P.T) / (2.0 * num_samples)
    return torch.clamp(P, min=1e-12)

def tsne_kl_loss(P, Y):
    sum_Y = torch.sum(Y*Y, dim=1, keepdim=True)
    D_Y = sum_Y + sum_Y.T - 2 * torch.mm(Y, Y.T)
    D_Y = torch.clamp(D_Y, min=0.0)
    
    Q = 1.0 / (1.0 + D_Y)
    Q = Q * (1.0 - torch.eye(Y.shape[0], device=Y.device)) 
    Q = Q / (torch.sum(Q) + 1e-8)
    Q = torch.clamp(Q, min=1e-12)
    
    return torch.sum(P * torch.log(P / Q))

# ---------------------------
# Utilities
# ---------------------------

def robust_load_csv_try_variants(path):
    for var in [{"skiprows":32}, {"skiprows":33}, {"skiprows":1}, {}]:
        try:
            df = pd.read_csv(path, **var)
            return df
        except: continue
    return pd.read_csv(path)

def extract_columns_from_df(df, use_phase=False):
    cols_map = {c.lower(): c for c in df.columns}
    freq_col = next((cols_map[c] for c in ["frequency", "freq", "f"] if c in cols_map), df.columns[0])
    imp_col  = next((cols_map[c] for c in ["trace |z| (ohm)", "impedance", "trace |z|", "|z|", "imp", "z"] if c in cols_map), df.columns[1])
    phase_col = None
    if use_phase:
        phase_col = next((cols_map[c] for c in ["trace th (deg)", "phase", "angle", "th"] if c in cols_map), None)
    freq  = df[freq_col].values
    imp   = df[imp_col].values
    phase = df[phase_col].values if (use_phase and phase_col) else None
    return freq, phase, imp

def load_sweep_vector(path, ref_freq=REF_FREQ, use_phase=USE_PHASE):
    df = robust_load_csv_try_variants(path)
    freq, phase, imp = extract_columns_from_df(df, use_phase)
    if freq[0] > freq[-1]:
        freq, imp = freq[::-1], imp[::-1]
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

def choose_network_components(X_rows, X_cols, desired_bits=ID_BIT_LENGTH):
    return min(desired_bits, X_cols) 

def binary_from_projection(proj, bits):
    return ''.join('1' if x > 0 else '0' for x in proj[:bits])

def hamming_distance(a, b):
    if a is None or b is None: return None
    if len(a) != len(b):
        L = max(len(a), len(b))
        a = a.ljust(L, "0"); b = b.ljust(L, "0")
    return sum(x != y for x, y in zip(a, b))

def export_device_debug_data(report_dir, single_ids, multi_ids, single_results, multi_results):
    device_list = sorted(set(single_ids.keys()) | set(multi_ids.keys()))
    records = []
    single_map = {r["Expected"]: r for r in single_results}
    multi_map = {r["Expected"]: r for r in multi_results}

    for dev in device_list:
        rec = {"Device": dev, "Single_ID": single_ids.get(dev), "Multi_ID": multi_ids.get(dev)}
        s_res, m_res = single_map.get(dev, {}), multi_map.get(dev, {})
        rec.update({
            "Single_Intra_Hamming": s_res.get("Intra_Hamming"),
            "Single_Match": s_res.get("Match"),
            "Single_Predicted": s_res.get("Predicted"),
            "Multi_Intra_Hamming": m_res.get("Intra_Hamming"),
            "Multi_Match": m_res.get("Match"),
            "Multi_Predicted": m_res.get("Predicted"),
        })
        records.append(rec)

    out_path = os.path.join(report_dir, "device_debug_metrics.csv")
    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f" Device debug data written to: {out_path}")

# ---------------------------
# Registration / Training
# ---------------------------

def build_single_sweep_ids(device_files, reg_index):
    train_vecs, device_order = [], []
    for dev, files in sorted(device_files.items()):
        if reg_index in files:
            vec = load_sweep_vector(files[reg_index])
            train_vecs.append(vec); device_order.append(dev)
            
    if not train_vecs: raise RuntimeError(f"No reg sweeps at index {reg_index}")
    X = np.vstack(train_vecs)
    n_samples, n_features = X.shape
    n_comp = choose_network_components(n_samples, n_features)
    
    scaler = StandardScaler().fit(X)
    X_tensor = torch.tensor(scaler.transform(X), dtype=torch.float32)
    P = compute_joint_probabilities(X_tensor, target_perplexity=30.0)
    
    ae_model = TSNEEncoder(input_features=n_features, bit_length=n_comp)
    optimizer = optim.Adam(ae_model.parameters(), lr=0.005)
    
    print("[Single-Sweep] Training Autoencoder...")
    for epoch in range(250):
        optimizer.zero_grad()
        Y = ae_model(X_tensor)
        loss = tsne_kl_loss(P, Y)
        loss.backward()
        optimizer.step()
        
    with torch.no_grad():
        Xp = ae_model(X_tensor).numpy()
        
    binary_ids = {device_order[i]: binary_from_projection(Xp[i], n_comp) for i in range(len(device_order))}
    return device_order, binary_ids, {
        "scaler": scaler, "ae_model": ae_model, "ref_freq": REF_FREQ, 
        "use_phase": USE_PHASE, "bit_length": n_comp
    }

def build_multi_sweep_ids(device_files, train_indices):
    X, labels = [], []
    for dev, files in sorted(device_files.items()):
        for idx in train_indices:
            if idx in files:
                vec = load_sweep_vector(files[idx]); X.append(vec); labels.append(dev)
                
    if not X: raise RuntimeError("No multi-sweep data")
    X = np.vstack(X); n_samples, n_features = X.shape
    n_comp = choose_network_components(n_samples, n_features)
    
    scaler = StandardScaler().fit(X)
    X_tensor = torch.tensor(scaler.transform(X), dtype=torch.float32)
    P = compute_joint_probabilities(X_tensor, target_perplexity=30.0)
    
    ae_model = TSNEEncoder(input_features=n_features, bit_length=n_comp)
    optimizer = optim.Adam(ae_model.parameters(), lr=0.005)
    
    print("[Multi-Sweep] Training Autoencoder...")
    for epoch in range(250):
        optimizer.zero_grad()
        Y = ae_model(X_tensor)
        loss = tsne_kl_loss(P, Y)
        loss.backward()
        optimizer.step()
        
    with torch.no_grad():
        Xp = ae_model(X_tensor).numpy()
        
    binary_ids = {}
    for dev in sorted(set(labels)):
        idxs = [i for i, l in enumerate(labels) if l == dev]
        mean_proj = np.mean(Xp[idxs, :], axis=0)
        binary_ids[dev] = binary_from_projection(mean_proj, n_comp)
        
    return binary_ids, {
        "scaler": scaler, "ae_model": ae_model, "ref_freq": REF_FREQ, 
        "use_phase": USE_PHASE, "bit_length": n_comp
    }

# ---------------------------
# Authentication
# ---------------------------
def authenticate_files(model, binary_ids, device_files, test_index, threshold=None, excluded=None):
    excluded = set(excluded or [])
    scaler, ae_model, bit_length = model["scaler"], model["ae_model"], model["bit_length"]
    results = []
    flags = defaultdict(list)
    intra, inter = [], []

    ae_model.eval() # Set model to evaluation mode
    
    for dev, files in sorted(device_files.items()):
        if dev in excluded or test_index not in files:
            continue

        vec = load_sweep_vector(files[test_index], ref_freq=model["ref_freq"], use_phase=model["use_phase"])
        vec_scaled = scaler.transform(vec.reshape(1, -1))
        
        with torch.no_grad():
            vec_tensor = torch.tensor(vec_scaled, dtype=torch.float32)
            proj = ae_model(vec_tensor).numpy()[0]
            
        gen_bin = binary_from_projection(proj, bit_length)
        reg_bin = binary_ids.get(dev)
        intra_d = hamming_distance(gen_bin, reg_bin)
        
        if intra_d is not None:
            intra.append(intra_d)

        for o_dev, o_bin in binary_ids.items():
            if o_dev != dev:
                inter.append(hamming_distance(gen_bin, o_bin))

        best, min_d = None, None
        for o_dev, o_bin in binary_ids.items():
            d = hamming_distance(gen_bin, o_bin)
            if min_d is None or d < min_d:
                min_d, best = d, o_dev

        match = (best == dev)
        threshold_pass = (threshold is None) or (intra_d is not None and intra_d <= threshold)
        success = match and threshold_pass

        results.append({
            "File": os.path.basename(files[test_index]),
            "Expected": dev,
            "Predicted": best,
            "Intra_Hamming": intra_d,
            "Within_Threshold": threshold_pass,
            "Match": match,
            "Authenticated": success
        })
        flags[dev].append(success)

    return results, flags, intra, inter

# ---------------------------
# Plot helpers
# ---------------------------

def plot_combined_pdf(intra, inter, title, outpath):
    plt.figure(figsize=(6, 4))

    intra = np.array(intra)
    inter = np.array(inter)

    if len(intra) == 0 or len(inter) == 0:
        return

    max_val = max(inter.max(), intra.max()) if len(inter) and len(intra) else 0
    bins = np.linspace(0, max_val, 25)

    inter_color = '#1f77b4'   # blue
    intra_color = '#d62728'   # red

    plt.hist(inter, bins=bins, density=True, alpha=0.6, label="Inter-HD", edgecolor='black', linewidth=0.8, color=inter_color)
    plt.hist(intra, bins=bins, density=True, alpha=0.6, label="Intra-HD", edgecolor='black', linewidth=0.8, color=intra_color)

    plt.axvline(inter.mean(), linestyle='--', linewidth=2, color=inter_color, label=f'Inter Mean = {inter.mean():.2f}')
    plt.axvline(intra.mean(), linestyle='--', linewidth=2, color=intra_color, label=f'Intra Mean = {intra.mean():.2f}')

    plt.xlabel("Hamming Distance")
    plt.ylabel("Probability Density")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)

    plt.tight_layout()
    plt.savefig(outpath, dpi=600, bbox_inches='tight')
    plt.close()

# ---------------------------
# Main
# ---------------------------

def run_all_tests(folder):
    device_files = collect_device_files(folder)
    print("Devices Found:", len(device_files))

    # --- Single Sweep Execution ---
    order, bin_ids, model = build_single_sweep_ids(device_files, PREFERRED_REG_INDEX_SINGLE)
    single_results, flags_s, intra_s, inter_s = authenticate_files(
        model, bin_ids, device_files, PREFERRED_AUTH_INDEX_SINGLE,
        threshold=HAMMING_AUTH_THRESHOLD, excluded=EXCLUDED_DEVICES
    )

    pd.DataFrame(single_results).to_csv(os.path.join(REPORT_DIR, "single_sweep_metrics.csv"), index=False)
    plot_combined_pdf(intra_s, inter_s, "Hamming Distance Distribution (Single-Sweep Autoencoder)", os.path.join(REPORT_DIR, "single_combined.png"))

    total_reg_s = len(order)
    success_s = sum(1 for d in order if all(flags_s[d]))
    rate_s = success_s / total_reg_s * 100 if total_reg_s else 0

    # --- Multi Sweep Execution ---
    bin_ids_m, model_m = build_multi_sweep_ids(device_files, PREFERRED_MULTI_TRAIN_INDICES)
    multi_results, flags_m, intra_m, inter_m = authenticate_files(
        model_m, bin_ids_m, device_files, PREFERRED_MULTI_AUTH_INDEX,
        threshold=HAMMING_AUTH_THRESHOLD, excluded=EXCLUDED_DEVICES
    )

    pd.DataFrame(multi_results).to_csv(os.path.join(REPORT_DIR, "multi_sweep_metrics.csv"), index=False)
    plot_combined_pdf(intra_m, inter_m, "Hamming Distance Distribution (Multi-Sweep Autoencoder)", os.path.join(REPORT_DIR, "multi_combined.png"))

    total_reg_m = len(bin_ids_m)
    success_m = sum(1 for d in bin_ids_m if all(flags_m[d]))
    rate_m = success_m / total_reg_m * 100 if total_reg_m else 0

    # --- Metrics & Markdown ---
    pd.DataFrame([
        {"Scenario": "Single", "Registered": total_reg_s, "Success": success_s, "Rate%": rate_s, "MeanIntra": np.mean(intra_s)},
        {"Scenario": "Multi", "Registered": total_reg_m, "Success": success_m, "Rate%": rate_m, "MeanIntra": np.mean(intra_m)},
    ]).to_csv(os.path.join(REPORT_DIR, "comparison_summary.csv"), index=False)

    export_device_debug_data(REPORT_DIR, bin_ids, bin_ids_m, single_results, multi_results)

    with open(os.path.join(REPORT_DIR, "Final_Results.md"), "w") as f:
        f.write("# Final Results Report (Autoencoder / Parametric t-SNE)\n\n")
        f.write("## What are intra vs inter Hamming graphs?\n")
        f.write("- **Intra-device distances**: Hamming distance between a regenerated ID (at authentication) and its *own* registered ID.\n")
        f.write("  These show how stable/reproducible each device’s ID is over time.\n")
        f.write("- **Inter-device distances**: Hamming distances between a regenerated ID and *all other devices’* registered IDs.\n")
        f.write("  These show how well-separated the devices are (uniqueness).\n\n")
        f.write("Ideally: intra distances are low (close to 0), while inter distances are high (close to half the ID length).\n\n")

        f.write("## Single-sweep Execution\n")
        f.write(f"- Devices registered: **{total_reg_s}**\n- Successfully authenticated: **{success_s}**\n")
        f.write(f"- Success rate: **{rate_s:.2f}%**\n- Mean intra Hamming: **{np.mean(intra_s):.2f}**\n\n")
        f.write("### Plots\n")
        f.write("![Single Combined Distribution](single_combined.png)\n\n")

        f.write("## Multi-sweep Execution\n")
        f.write(f"- Devices registered: **{total_reg_m}**\n- Successfully authenticated: **{success_m}**\n")
        f.write(f"- Success rate: **{rate_m:.2f}%**\n- Mean intra Hamming: **{np.mean(intra_m):.2f}**\n\n")
        f.write("### Plots\n")
        f.write("![Multi Combined Distribution](multi_combined.png)\n\n")

    print(" Reports written to", REPORT_DIR)

if __name__ == "__main__":
    run_all_tests(DEVICE_FOLDER)
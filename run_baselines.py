"""
Dual-Subspace SHM — Baseline Experiments
=========================================
Runs two independent baseline evaluations:

  1. PAMELA standalone (30 sweeps, 1 sensor, real damage)
  2. Cohort + synthetic perturbations (300 sensors, 1500 sweeps)

Outputs: CSV results + figures saved to ./rq_analysis_reports/
"""
import subprocess, sys, time

baselines = [
    ("PAMELA standalone", "pamela_baseline_comparison.py"),
    ("Cohort + synthetic perturbations", "subspace_comparison.py"),
]

for name, script in baselines:
    print("\n" + "=" * 70)
    print(f"BASELINE: {name}")
    print("=" * 70)
    t0 = time.time()
    result = subprocess.run([sys.executable, script], capture_output=False)
    elapsed = time.time() - t0
    print(f"\n[{script}] finished in {elapsed:.0f}s with return code {result.returncode}")

print("\n" + "=" * 70)
print("ALL BASELINES COMPLETE")
print("=" * 70)
print("Outputs in ./rq_analysis_reports/")
print("  pamela_baseline_comparison.csv / .png")
print("  subspace_method_comparison.csv / .png")

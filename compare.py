import os
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

PYTHON_DIR = "python_reports"
MATLAB_DIR = "matlab_reports"
OUTPUT_DIR = "comparison_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def is_debug_file(filename):
    return filename.lower().startswith("device_debug_metrics")


def compare_dataframes(df_py, df_mat):

    result = {
        "rows_python": len(df_py),
        "rows_matlab": len(df_mat),
        "same_row_count": len(df_py) == len(df_mat),
        "same_columns": False,
        "numeric_columns_checked": 0,
        "string_columns_checked": 0,
        "numeric_max_abs_diff": 0.0,
        "numeric_mean_abs_diff": 0.0,
        "string_match_pct": np.nan,
    }

    py_cols = list(df_py.columns)
    mat_cols = list(df_mat.columns)

    result["same_columns"] = py_cols == mat_cols

    common_cols = [c for c in py_cols if c in mat_cols]

    numeric_diffs = []
    string_match_rates = []

    n_rows = min(len(df_py), len(df_mat))

    df_py = df_py.iloc[:n_rows]
    df_mat = df_mat.iloc[:n_rows]

    for col in common_cols:

        try:
            py_num = pd.to_numeric(df_py[col])
            mat_num = pd.to_numeric(df_mat[col])

            diff = np.abs(py_num - mat_num)

            numeric_diffs.extend(diff.dropna().tolist())

            result["numeric_columns_checked"] += 1

        except Exception:

            py_str = df_py[col].astype(str).fillna("")
            mat_str = df_mat[col].astype(str).fillna("")

            matches = (py_str == mat_str).mean()

            string_match_rates.append(matches)

            result["string_columns_checked"] += 1

    if numeric_diffs:
        result["numeric_max_abs_diff"] = float(np.max(numeric_diffs))
        result["numeric_mean_abs_diff"] = float(np.mean(numeric_diffs))

    if string_match_rates:
        result["string_match_pct"] = (
            float(np.mean(string_match_rates)) * 100.0
        )

    return result


def save_row_level_differences(df_py, df_mat, filename):

    common_cols = [c for c in df_py.columns if c in df_mat.columns]

    n_rows = min(len(df_py), len(df_mat))

    diffs = []

    for idx in range(n_rows):

        for col in common_cols:

            a = df_py.iloc[idx][col]
            b = df_mat.iloc[idx][col]

            if pd.isna(a) and pd.isna(b):
                continue

            if str(a) != str(b):
                diffs.append(
                    {
                        "row": idx,
                        "column": col,
                        "python": a,
                        "matlab": b,
                    }
                )

    if diffs:
        pd.DataFrame(diffs).to_csv(
            os.path.join(
                OUTPUT_DIR,
                filename.replace(".csv", "_differences.csv"),
            ),
            index=False,
        )


def create_comparison_plots():

    records = []

    for fname in os.listdir(PYTHON_DIR):

        if not fname.endswith(".csv"):
            continue

        if fname.startswith("device_debug_metrics"):
            continue

        if not fname.startswith("comparison_summary"):
            continue

        py_path = os.path.join(PYTHON_DIR, fname)
        mat_path = os.path.join(MATLAB_DIR, fname)

        if not os.path.exists(mat_path):
            continue

        bit_match = re.search(r"(\d+)bit", fname)

        if not bit_match:
            continue

        bits = int(bit_match.group(1))

        py_df = pd.read_csv(py_path)
        mat_df = pd.read_csv(mat_path)

        for scenario in ["Single", "Multi"]:

            py_row = py_df[
                py_df["Scenario"].str.lower()
                == scenario.lower()
            ]

            mat_row = mat_df[
                mat_df["Scenario"].str.lower()
                == scenario.lower()
            ]

            if len(py_row) == 0 or len(mat_row) == 0:
                continue

            records.append({
                "Bits": bits,
                "Scenario": scenario,

                "Python_Rate":
                    float(py_row["Rate%"].iloc[0]),

                "MATLAB_Rate":
                    float(mat_row["Rate_Pct"].iloc[0])
                    if "Rate_Pct" in mat_row.columns
                    else float(mat_row["Rate%"].iloc[0]),

                "Python_Intra":
                    float(py_row["MeanIntra"].iloc[0]),

                "MATLAB_Intra":
                    float(mat_row["MeanIntra"].iloc[0]),

                "Python_Inter":
                    float(py_row["MeanInter"].iloc[0]),

                "MATLAB_Inter":
                    float(mat_row["MeanInter"].iloc[0]),

                "Python_Separation":
                    float(py_row["Separation"].iloc[0]),

                "MATLAB_Separation":
                    float(mat_row["Separation"].iloc[0]),
            })

    if not records:
        print("No comparison summary files found.")
        return

    df = pd.DataFrame(records)

    for scenario in ["Single", "Multi"]:

        sub = df[df["Scenario"] == scenario]
        sub = sub.sort_values("Bits")

        # ---------------------------------
        # Success Rate Plot
        # ---------------------------------

        plt.figure(figsize=(8, 5))

        plt.plot(
            sub["Bits"],
            sub["Python_Rate"],
            marker="o",
            linewidth=2,
            label="Python",
        )

        plt.plot(
            sub["Bits"],
            sub["MATLAB_Rate"],
            marker="s",
            linewidth=2,
            label="MATLAB",
        )

        plt.title(
            f"{scenario} Sweep Authentication Rate"
        )

        plt.xlabel("ID Length (bits)")
        plt.ylabel("Authentication Rate (%)")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                OUTPUT_DIR,
                f"{scenario.lower()}_rate_comparison.png"
            ),
            dpi=300,
        )

        plt.close()

        # ---------------------------------
        # Mean Intra Hamming Plot
        # ---------------------------------

        plt.figure(figsize=(8, 5))

        plt.plot(
            sub["Bits"],
            sub["Python_Intra"],
            marker="o",
            linewidth=2,
            label="Python",
        )

        plt.plot(
            sub["Bits"],
            sub["MATLAB_Intra"],
            marker="s",
            linewidth=2,
            label="MATLAB",
        )

        plt.title(
            f"{scenario} Sweep Mean Intra Hamming"
        )

        plt.xlabel("ID Length (bits)")
        plt.ylabel("Mean Intra Hamming Distance")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                OUTPUT_DIR,
                f"{scenario.lower()}_intra_comparison.png"
            ),
            dpi=300,
        )

        plt.close()


        plt.figure(figsize=(8, 5))

        plt.plot(
            sub["Bits"],
            sub["Python_Inter"],
            marker="o",
            linewidth=2,
            label="Python",
        )

        plt.plot(
            sub["Bits"],
            sub["MATLAB_Inter"],
            marker="s",
            linewidth=2,
            label="MATLAB",
        )

        plt.title(
            f"{scenario} Sweep Mean Inter Hamming"
        )

        plt.xlabel("ID Length (bits)")
        plt.ylabel("Mean Inter Hamming Distance")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                OUTPUT_DIR,
                f"{scenario.lower()}_inter_comparison.png"
            ),
            dpi=300,
        )

        plt.close()

        plt.figure(figsize=(8, 5))

        plt.plot(
            sub["Bits"],
            sub["Python_Separation"],
            marker="o",
            linewidth=2,
            label="Python",
        )

        plt.plot(
            sub["Bits"],
            sub["MATLAB_Separation"],
            marker="s",
            linewidth=2,
            label="MATLAB",
        )

        plt.title(
            f"{scenario} Sweep Separation"
        )

        plt.xlabel("ID Length (bits)")
        plt.ylabel("MeanInter - MeanIntra")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                OUTPUT_DIR,
                f"{scenario.lower()}_separation_comparison.png"
            ),
            dpi=300,
        )

        plt.close()
        
        plt.figure(figsize=(8, 5))

        plt.plot(
            sub["Bits"],
            sub["Python_Separation"],
            marker="o",
            linewidth=2,
            label="Python",
        )

        plt.plot(
            sub["Bits"],
            sub["MATLAB_Separation"],
            marker="s",
            linewidth=2,
            label="MATLAB",
        )

        plt.title(
            f"{scenario} Sweep Separation"
        )

        plt.xlabel("ID Length (bits)")
        plt.ylabel("MeanInter - MeanIntra")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                OUTPUT_DIR,
                f"{scenario.lower()}_separation_comparison.png"
            ),
            dpi=300,
        )

        plt.close()
        # ---------------------------------
        # Difference Plot
        # ---------------------------------

        plt.figure(figsize=(8, 5))

        rate_diff = np.abs(
            sub["Python_Rate"]
            - sub["MATLAB_Rate"]
        )

        intra_diff = np.abs(
            sub["Python_Intra"]
            - sub["MATLAB_Intra"]
        )

        x = np.arange(len(sub))

        width = 0.35

        plt.bar(
            x - width / 2,
            rate_diff,
            width,
            label="Rate Difference",
        )

        plt.bar(
            x + width / 2,
            intra_diff,
            width,
            label="Intra-HD Difference",
        )

        plt.xticks(
            x,
            [str(v) for v in sub["Bits"]]
        )

        plt.xlabel("ID Length (bits)")
        plt.ylabel("Absolute Difference")
        plt.title(
            f"{scenario} MATLAB vs Python Difference"
        )

        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                OUTPUT_DIR,
                f"{scenario.lower()}_difference.png"
            ),
            dpi=300,
        )

        plt.close()

    print("Comparison plots generated.")

summary_rows = []

python_files = {
    f
    for f in os.listdir(PYTHON_DIR)
    if f.endswith(".csv") and not is_debug_file(f)
}

matlab_files = {
    f
    for f in os.listdir(MATLAB_DIR)
    if f.endswith(".csv") and not is_debug_file(f)
}

all_files = sorted(python_files | matlab_files)

for fname in all_files:

    py_exists = fname in python_files
    mat_exists = fname in matlab_files

    if not py_exists or not mat_exists:

        summary_rows.append(
            {
                "file": fname,
                "status": "missing",
                "python_exists": py_exists,
                "matlab_exists": mat_exists,
            }
        )

        continue

    print(f"Comparing {fname}")

    py_path = os.path.join(PYTHON_DIR, fname)
    mat_path = os.path.join(MATLAB_DIR, fname)

    try:

        df_py = pd.read_csv(py_path)
        df_mat = pd.read_csv(mat_path)

        stats = compare_dataframes(df_py, df_mat)

        save_row_level_differences(
            df_py,
            df_mat,
            fname,
        )

        stats["file"] = fname
        stats["status"] = "ok"

        summary_rows.append(stats)

    except Exception as e:

        summary_rows.append(
            {
                "file": fname,
                "status": f"error: {e}",
            }
        )

summary_df = pd.DataFrame(summary_rows)

summary_path = os.path.join(
    OUTPUT_DIR,
    "comparison_summary.csv",
)

summary_df.to_csv(summary_path, index=False)
create_comparison_plots()
print()
print("=" * 60)
print("COMPARISON COMPLETE")
print("=" * 60)
print(summary_df)
print()
print("Summary written to:")
print(summary_path)
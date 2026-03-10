# compute_enrichment_difference.py
#
# For each time window, computes the difference in motif enrichment:
#   neural_labels - unfiltered
#
# Output table columns:
#   1. frac_positive   : fraction of motifs where the difference is > 0 (per loop)
#   2. count_11        : number of 1-1 cells for that loop (consistent across motifs)
#   3. <motif_1> ...   : per-motif enrichment differences

import os
import numpy as np
import pandas as pd

from utils import print_timestamp


WINDOWS = ['06-08', '10-12', '14-16']


def compute_difference_for_window(window: str) -> None:
    neural_dir      = f"results/training_data/neural_labels/hrs{window}"
    unfiltered_dir  = f"results/training_data/unfiltered/hrs{window}"
    output_dir      = f"results/EDA/enrichement_difference"
    os.makedirs(output_dir, exist_ok=True)

    # --- load enrichment tables ---
    neural_path     = os.path.join(neural_dir,     f"motif_enrichment_hrs{window}.csv")
    unfiltered_path = os.path.join(unfiltered_dir, f"motif_enrichment_hrs{window}.csv")

    print_timestamp(f"Window {window}: loading enrichment tables...")
    neural_df     = pd.read_csv(neural_path,     index_col=0)
    unfiltered_df = pd.read_csv(unfiltered_path, index_col=0)

    # --- load count_11 (take from neural; value is loop-level so either file works) ---
    count_path = os.path.join(neural_dir, f"count11_hrs{window}.csv")
    print_timestamp(f"Window {window}: loading count_11 table...")
    count_df = pd.read_csv(count_path, index_col=0)
    # All motif columns hold the same value per row — take the first column
    count_11 = count_df.iloc[:, 0].rename("count_11")

    # --- align indices / columns (defensive) ---
    shared_loops  = neural_df.index.intersection(unfiltered_df.index)
    shared_motifs = neural_df.columns.intersection(unfiltered_df.columns)

    neural_df     = neural_df.loc[shared_loops, shared_motifs]
    unfiltered_df = unfiltered_df.loc[shared_loops, shared_motifs]
    count_11      = count_11.loc[shared_loops]

    # --- compute difference ---
    diff_df = neural_df - unfiltered_df  # (n_loops, n_motifs)

    # --- fraction of motifs with positive difference (ignoring NaN) ---
    frac_positive = (diff_df > 0).sum(axis=1) / diff_df.notna().sum(axis=1)
    frac_positive.name = "frac_positive"

    # --- assemble output: frac_positive | count_11 | motif columns ---
    result_df = pd.concat([frac_positive, count_11, diff_df], axis=1)

    out_path = os.path.join(output_dir, f"motif_enrichment_difference_hrs{window}.csv")
    result_df.to_csv(out_path)
    print_timestamp(f"Window {window}: saved difference table to {out_path}")


def main():
    for window in WINDOWS:
        compute_difference_for_window(window)
    print_timestamp("Computed enrichment differences for all three windows.")


if __name__ == "__main__":
    main()
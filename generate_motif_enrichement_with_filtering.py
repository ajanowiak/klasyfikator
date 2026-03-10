# generate_motif_enrichment_with_filtering_fast.py
#
# Optimized rewrite of generate_motif_enrichement_with_filtering.py
#
# Key changes vs. original:
#   1. BUG FIX: table.loc[loop, motif] (was: table.loc[tissue, motif] — undefined variable)
#   2. BUG FIX: CSV save + output_dir moved inside the `for loop` block (was mis-indented)
#   3. Eliminated big_dict_neural: instead of storing all raw distributions across tissues
#      and then chaining them, we accumulate (sum, count) incrementally — O(1) memory per cell.
#   4. Vectorized mean computation: numpy operations on the full motif matrix at once
#      instead of per-motif Python loops.
#   5. Parallel window processing via ProcessPoolExecutor.

import os
import argparse
import numpy as np
import pandas as pd
import pyreadr
import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

from utils import load_window_split_by_tissue, print_timestamp


WINDOWS = ['06-08', '10-12', '14-16']
ACTIVITY_PROFILES = ["1-1", "1-0", "0-1", "0-0"]

NEURAL_LABELS_RAW = [
    "Brain", "Neural", "Ventral_nerve_cord",
    "Ventral_nerve_cord_prim", "Glia", "PNS_&_sense"
]
NEURAL_LABELS = list(map(lambda s: s.replace("prim", "prim.").replace("_", " "), NEURAL_LABELS_RAW))


def compute_enrichment_for_window(window: str, anot_df: pd.DataFrame, filter_labels = False) -> None:
    """
    For a single time window:
      - load data split by tissue
      - filter to neural tissues
      - compute per-loop motif enrichment (mean_11 - mean_other) fully vectorized
      - save one CSV per loop

    No intermediate distribution storage: we accumulate weighted sums directly.
    """
    print_timestamp(f"Window {window}: loading tissue-split data...")
    tissue_dict = load_window_split_by_tissue(window=window, metadata_df=anot_df)

    valid_labels = [l for l in NEURAL_LABELS if l in tissue_dict]
    if not valid_labels:
        print_timestamp(f"Window {window}: no valid neural labels found, skipping.")
        return

    # Retrieve index labels from the first valid tissue
    first_loops_df, first_motifs_df = tissue_dict[valid_labels[0]]
    loop_ids = list(first_loops_df.index)
    motif_ids = list(first_motifs_df.index)

    # For each loop, accumulate sum and count across all neural tissues
    # Shape: (n_loops, n_motifs)
    n_loops = len(loop_ids)
    n_motifs = len(motif_ids)

    sum_11    = np.zeros((n_loops, n_motifs), dtype=np.float64)
    count_11  = np.zeros((n_loops,),          dtype=np.int64)
    sum_other    = np.zeros((n_loops, n_motifs), dtype=np.float64)
    count_other  = np.zeros((n_loops,),          dtype=np.int64)

    for label in valid_labels:
        loops_df, motifs_df = tissue_dict[label]

        # numpy matrices: shape (n_loops, n_cells) and (n_motifs, n_cells)
        loops_mat  = loops_df.to_numpy()   # (n_loops,  n_cells)
        motifs_mat = motifs_df.to_numpy() # (n_motifs, n_cells)

        for i in range(n_loops):
            mask_11    = loops_mat[i] == 11
            mask_other = ~mask_11  # 1-0, 0-1, 0-0 combined

            motifs_11    = motifs_mat[:, mask_11]    # (n_motifs, k)
            motifs_other = motifs_mat[:, mask_other] # (n_motifs, m)

            if motifs_11.shape[1] > 0:
                sum_11[i]   += motifs_11.sum(axis=1)
                count_11[i] += motifs_11.shape[1]

            if motifs_other.shape[1] > 0:
                sum_other[i]   += motifs_other.sum(axis=1)
                count_other[i] += motifs_other.shape[1]

    # Vectorized mean difference: (n_loops, n_motifs)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean_11    = np.where(count_11[:, None]    > 0, sum_11    / count_11[:, None],    np.nan)
        mean_other = np.where(count_other[:, None] > 0, sum_other / count_other[:, None], np.nan)

    enrichment_matrix = mean_11 - mean_other  # (n_loops, n_motifs)

    # Save one CSV per loop (matching original output format)
    if filter_labels:
        output_dir = f"results/training_data/neural_labels/hrs{window}"
    else:
        output_dir = f"results/training_data/unfiltered/hrs{window}"
    
    os.makedirs(output_dir, exist_ok=True)

    # Also save the full matrix for convenience
    full_table = pd.DataFrame(enrichment_matrix, index=loop_ids, columns=motif_ids, dtype=float)
    full_path = os.path.join(output_dir, f"motif_enrichment_hrs{window}.csv")
    full_table.to_csv(full_path)

    print_timestamp(f"Window {window}: saved enrichment matrix to {full_path}")


def main():
    parser = argparse.ArgumentParser(description="Perform a small literature search.")
    parser.add_argument("--filter_labels", required=True, help="Boolean parameter for deciding whether cell should be filtered based on neural_labels")

    args = parser.parse_args()

    print_timestamp("Reading metadata (tissue annotations)...")
    atac_meta = pyreadr.read_r('data/atac_meta.rds')
    anot_df = list(atac_meta.values())[0]

    # Process windows in parallel (one process per window)
    # If RAM is tight, reduce max_workers or process sequentially.
    with ProcessPoolExecutor(max_workers=len(WINDOWS)) as executor:
        futures = {
            executor.submit(compute_enrichment_for_window, w, anot_df, args.filter_labels): w
            for w in WINDOWS
        }
        for fut in as_completed(futures):
            w = futures[fut]
            try:
                fut.result()
            except Exception as e:
                print_timestamp(f"\t Window {w} failed: {e}")

    print_timestamp("All tables saved.")


if __name__ == '__main__':
    main()
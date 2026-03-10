# generate_motif_enrichement_with_filtering.py

# this is effectively a newer implementation of generate_data.py

import os
import numpy as np
import pandas as pd
import pyreadr
import datetime
import itertools

from utils import *
from generate_tissue_annotation_chromvar_distribution_plots import distributions



def main():

    windows = ['06-08', '10-12', '14-16']
    # loop_ids = ['L21', 'L32', 'L222', 'L400']
    # motif_ids = ['M4676-1.02', 'M2013-1.02', 'M4913-1.02', 'M4962-1.02', 'M4982-1.02', 'M2061-1.02']
    activity_profiles = ["1-1", "1-0", "0-1", "0-0"]


    # -------------- compute all distributions into one dictionary (stratified by tissue) -------------------
    big_dict_stratified = {w: {} for w in windows}

    print_timestamp("Reading metadata (tissue annotations)...")
    # read tissue annotations of each cell
    atac_meta = pyreadr.read_r('data/atac_meta.rds') # also works for RData
    anot_df = list(atac_meta.values())[0]

    print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Reading the motif lookup table...")
    # motif_lookup = pd.read_csv("data/motif_names.tsv", sep="\t")

    for window in windows[::-1]:
        print_timestamp(f"\t Computing distributions for window {window}\n")

        tissue_dict = load_window_split_by_tissue(
            window=window,
            metadata_df=anot_df
        )

        big_dict_stratified[window] = {}

        for tissue, (loops_df, motifs_df) in tissue_dict.items():

            big_dict_stratified[window][tissue] = distributions(
                loop_ids = list(loops_df.index),
                motif_ids = list(motifs_df.index),
                loops_df = loops_df,
                motifs_df = motifs_df
            )

    print_timestamp(f"\t Distributions computed.\n")
    print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\t Distributions computed.\n")


    neural_labels = [
        "Brain", "Neural", "Ventral_nerve_cord",
        "Ventral_nerve_cord_prim", "Glia", "PNS_&_sense"
    ]

    # this could've been a list comprehension but isn't
    neural_labels = list(map(lambda s: s.replace("prim", "prim.").replace("_", " "), neural_labels))
    motif_ids = motifs_df.index
    loop_ids = loops_df.index

    big_dict_neural = {w:{
        l : {
            m : {
                profile : [] for profile in activity_profiles
            } for m in motif_ids
        } for l in loop_ids
    } for w in windows}

    for window in windows:
        valid_labels = list(set(big_dict_stratified[window].keys()) & set(neural_labels))
        if not valid_labels:
            continue
        for loop in loop_ids:
            for motif in motif_ids:
                for profile in activity_profiles:
                    # unify the distributions of different tissues in neural_label
                    big_dict_neural[window][loop][motif][profile] = list(itertools.chain(*[big_dict_stratified[window][label][loop][motif][profile] for label in valid_labels]))

    # -------------- compute and subtract the means -------------------

    for window in windows:
        output_dir = f"results/training_data/neural_labels/hrs{window}"
        os.makedirs(output_dir, exist_ok=True)

        # tissues = big_dict_stratified[window].keys()

        for loop in loop_ids:

            table = pd.DataFrame(index=list(loop_ids), columns=list(motif_ids), dtype=float)

            for motif in motif_ids:

                entry = big_dict_neural[window][loop][motif]

                arr_11 = entry["1-1"]
                arr_other = np.concatenate([
                    entry[p] for p in activity_profiles[1:]
                    if len(entry[p]) > 0
                ]) if any(len(entry[p]) > 0 for p in activity_profiles[1:]) else np.array([])

                if len(arr_11) > 0 and len(arr_other) > 0:
                    mean_11 = np.mean(arr_11)
                    mean_other = np.mean(arr_other)
                    table.loc[tissue, motif] = mean_11 - mean_other
                else:
                    table.loc[tissue, motif] = np.nan

    # -------------- save csv -------------------
    out_path = os.path.join(output_dir, f"motif_enrichment_hrs{window}.csv")
    table.to_csv(out_path)

    print("All tables saved.")

if __name__ == '__main__':
    main()
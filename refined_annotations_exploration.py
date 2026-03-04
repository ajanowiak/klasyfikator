# refined_annotations_exploration.py
# Creates refined_annotation x motifs table for each combination of (loop, window) with mean_11 - mean_other values

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
    loop_ids = ['L21', 'L32', 'L222', 'L400']
    motif_ids = ['M4676-1.02', 'M2013-1.02', 'M4913-1.02', 'M4962-1.02', 'M4982-1.02', 'M2061-1.02']
    activity_profiles = ["1-1", "1-0", "0-1", "0-0"]

    # firstly, the distributions need to be computed
    big_dict_stratified = {w: {} for w in windows}

    print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Reading metadata (tissue annotations)...")
    atac_meta = pyreadr.read_r('data/atac_meta.rds') # also works for RData
    anot_df = list(atac_meta.values())[0]

    print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Reading the motif lookup table...")
    motif_lookup = pd.read_csv("data/motif_names.tsv", sep="\t")
    motif_name_map = dict(zip(motif_lookup["id"], motif_lookup["name"]))

    for window in windows[::-1]:
        print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\t Computing distributions for window {window}\n")

        tissue_dict = load_window_split_by_tissue(
            window=window,
            metadata_df=anot_df
        )

        big_dict_stratified[window] = {}

        for tissue, (loops_df, motifs_df) in tissue_dict.items():

            big_dict_stratified[window][tissue] = distributions(
                loop_ids,
                motif_ids,
                loops_df,
                motifs_df
            )

    output_dir = "results/EDA/refined_annotations"
    os.makedirs(output_dir, exist_ok=True)

    for window in windows:

        tissues = big_dict_stratified[window].keys()

        for loop in loop_ids:
            columns = pd.MultiIndex.from_tuples(
                [(m, motif_name_map.get(m, "NA")) for m in motif_ids],
                names=["motif_id", "motif_name"]
            )

            # Prepare table: rows = tissues, cols = motifs
            table = pd.DataFrame(index=list(tissues), columns=columns, dtype=float)

            for tissue in tissues:

                for motif in motif_ids:

                    entry = big_dict_stratified[window][tissue][loop][motif]

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

            # Save CSV
            out_path = os.path.join(output_dir, f"{loop}_hrs{window}.csv")
            table.to_csv(out_path)

    print("All tables saved.")

if __name__ == "__main__":
    main()
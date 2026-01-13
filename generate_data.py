#!/usr/bin/env python3

import os
import tqdm
import time
import json
import datetime
import numpy as np
import pandas as pd
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed


def convert_2D(data_3D: dict)-> pd.DataFrame:
    """ For each (loop, motif) pair: compute the difference in mean chromVAR scores between populations of cells '1-1' and 'other'. 

    Args:  
        data_3D (dict): output of process_window_parallel()

    Returns:
        data_diff: a DataFrame with loops (rows), motifs (columns) and "1-1" - "other" difference (values).
    """
    data_2D = {}
    for loop_id, motif_dict in data_3D.items():
        data_2D[loop_id] = {}
        for motif_id, profile_dict in motif_dict.items():
            mean_11 = profile_dict["1-1"]
            mean_other = profile_dict["other"]
            data_2D[loop_id][motif_id] = np.subtract(mean_11, mean_other)

    data_diff = pd.DataFrame.from_dict(data_2D, orient='index')
    return data_diff

def process_loop(loop_id, loops_df, motifs_df):
    """Process a single loop_id — runs in a worker process."""
    result = {}  # Change to dict to store motif_id as key

    loop_values = loops_df.loc[loop_id]
    cells_11 = loop_values[loop_values == 11].index
    cells_other = loop_values.index.difference(cells_11)

    for motif_id in motifs_df.index:
        motif_values = motifs_df.loc[motif_id]
        mean_11 = motif_values.loc[cells_11].mean() if len(cells_11) > 0 else np.nan
        mean_other = motif_values.loc[cells_other].mean() if len(cells_other) > 0 else np.nan
        result[motif_id] = {"1-1": mean_11, "other": mean_other}

    return loop_id, result

def load_window(window: str):
    """ Load data and drop missing values while preserving loop and motif labels.

    Args:
        window (str): timepoint to process (eg. 'hrs06-08')

    Returns:
        Two cleaned DataFrames: loops_df, motifs_df (with labels as index/columns)
    """
    loops_path = f"data/new_time/hrs{window}_NNv1_time_matrix_loops.tsv"
    motifs_path = f"data/new_time/hrs{window}_NNv1_time_matrix_motifs.tsv"

    print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\t\t Loading loops...\n")
    loops_df = pd.read_csv(loops_path, sep='\t', index_col=0)  # First column is loop ID
    print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\t\t Loading motifs...\n")
    motifs_df = pd.read_csv(motifs_path, sep='\t', index_col=0)  # First column is motif ID

    # Convert data to numeric, preserving labels
    print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\t\t Converting loops to numeric and dropping NaN values...\n")
    loops_df = loops_df.apply(pd.to_numeric, errors='coerce')
    loops_df.dropna(axis=1, inplace=True)  # Drop columns (cells) that are all NaN
    # loops_df.dropna(axis=0, how='all', inplace=True)  # Drop rows (loops) that are all NaN

    print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\t\t Converting motifs to numeric and dropping NaN values...\n")
    motifs_df = motifs_df.apply(pd.to_numeric, errors='coerce')
    motifs_df.dropna(axis=1, inplace=True)  # Drop columns (cells) that are all NaN
    # motifs_df.dropna(axis=0, how='all', inplace=True)  # Drop rows (motifs) that are all NaN
    
    # Keep only common cells (columns)
    common = list(set(loops_df.columns) & set(motifs_df.columns))
    loops_df = loops_df[common]
    motifs_df = motifs_df[common]
    
    print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\t\t Done\n")
    return loops_df, motifs_df

def process_window_parallel(window: str, n_workers=6)-> defaultdict:
    """Parallel computation of data_3D using multiple processes."""
    loops_df, motifs_df = load_window(window)
    
    data_3D = defaultdict(lambda: defaultdict(dict))
    print(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}\t\tReal business begins\n")

    # Submit tasks
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_loop, loop_id, loops_df, motifs_df): loop_id for loop_id in loops_df.index}

        for fut in tqdm.tqdm(as_completed(futures), total=len(futures)):
            loop_id, result = fut.result()
            data_3D[loop_id] = result

    return data_3D

def main():
    #   os.makedirs(OUT_DIR, exist_ok=True)
    windows = ['06-08', '10-12', '14-16']

    for window in windows:
        os.makedirs(f"notebook_results/hrs{window}", exist_ok=True)

        data_3D = process_window_parallel(window, n_workers=8) # UWAGA!!
        json.dump(data_3D, open(f"notebook_results/hrs{window}/data_3D_hrs{window}.json", 'w+'))

        data_diff = convert_2D(data_3D)
        data_diff.to_csv(f"notebook_results/hrs{window}/data_diff_hrs{window}.csv")

if __name__ == '__main__':
    main()
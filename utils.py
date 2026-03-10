# utils.py
# these imports are not supposed to execute (they fix annoing text highlighting in my IDE)
import os
import pandas as pd
import numpy as np
from collections import defaultdict
from collections.abc import Callable
import tqdm
import datetime
from scipy.stats import wasserstein_distance
from concurrent.futures import ProcessPoolExecutor, as_completed

def print_timestamp(message):
    print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {message}")

###     MODEL TRAINING UTILS
def compose_windows(tissue, windows=["06-08", "10-12", "14-16"]):
    """
    TIME-AGNOSTIC CLASSIFIER
    concatenate window-specific DataFrames and generate a `composite` vector for stratification
    """
    Xs, ys = [], []
    for idx, w in enumerate(windows):
        curr_X = pd.read_csv(f"data/training/hrs{w}/data_diff_hrs{w}.csv", index_col=0)
        curr_y = pd.read_csv(f"data/training/hrs{w}/y_{tissue}.csv", index_col=0).iloc[:, 0]
        curr_X['window'] = idx

        Xs.append(curr_X)
        ys.append(curr_y)

    y_new = pd.concat(ys, axis=0)
    X_new = pd.concat(Xs, axis=0)

    composite = pd.Categorical(list(zip(X_new['window'], y_new))).codes
    num_values = len(pd.Series(composite).unique())
    # print(f"Created a composite vector with {num_values} distinct values")

    X_new.drop('window', axis=1, inplace=True) # we don't want to use 'window' for prediction

    return X_new, y_new, composite

def make_names_dict():
    # FIXME: move this to some kind of config
    """
    MODEL TRAINING (metrics plots)
    Dictionary for f-string construction in model paths and in text on plots
    """
    model_names = ['RandomForestClassifier', 'SVC', 'LogisticRegression', 'XGBClassifier']
    model_names_dict = {name:{'full':'', 'short':''} for name in model_names}

    model_names_dict['RandomForestClassifier']['full'] = 'Random Forest'
    model_names_dict['RandomForestClassifier']['short'] = 'RF'

    model_names_dict['SVC']['full'] = 'Support Vector Machine'
    model_names_dict['SVC']['short'] = 'SVM'

    model_names_dict['LogisticRegression']['full'] = 'Logistic Regression'
    model_names_dict['LogisticRegression']['short'] = 'LR'

    model_names_dict['XGBClassifier']['full'] = 'XGBoost'
    model_names_dict['XGBClassifier']['short'] = 'XGB'

    return model_names_dict

###     MOTIF ENRICHEMENT DATA PREPARATION

def load_window(window: str):
    """ 
    Load and clean data (drop missing values while preserving loop and motif labels).

    Args:
        window (str): timepoint to process (eg. 'hrs06-08')

    Returns:
        Two cleaned DataFrames: loops_df, motifs_df (with labels as index/columns)
    """
    loops_path = f"data/new_time/hrs{window}_NNv1_time_matrix_loops.tsv"
    motifs_path = f"data/new_time/hrs{window}_NNv1_time_matrix_motifs.tsv"

    print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\t\t Loading data...\n")
    loops_df = pd.read_csv(loops_path, sep='\t', index_col=0)  # First column is loop ID
    motifs_df = pd.read_csv(motifs_path, sep='\t', index_col=0)  # First column is motif ID

    print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\t\t Converting DataFrames to numeric and dropping NaN values...\n")
    loops_df = loops_df.apply(pd.to_numeric, errors='coerce')
    loops_df.dropna(axis=1, inplace=True)  
    motifs_df = motifs_df.apply(pd.to_numeric, errors='coerce')
    motifs_df.dropna(axis=1, inplace=True) 
    
    # Keep only common cells (columns)
    common = list(set(loops_df.columns) & set(motifs_df.columns))
    loops_df = loops_df[common]
    motifs_df = motifs_df[common]
    
    return loops_df, motifs_df
def distributions(
    loop_ids: list[str],
    motif_ids: list[str],
    loops_df: pd.DataFrame,
    motifs_df: pd.DataFrame,
) -> dict:
    """
    Returns the entire z-score distributions for all activity profiles (lists of values)
    Args:
        loop_ids (list[str]): Loop indentifiers (like 'L417')
        motif_ids (list[str]): motif indentifiers (like 'M0111-1.02')
        loops_df, motifs_df (pd.DataFrame): dataframes loaded by load_window()

    Returns:
        result (dict): a dictionary with 
    """

    # Pre-extract numpy matrices once
    loops_mat = loops_df.loc[loop_ids].to_numpy()
    motifs_mat = motifs_df.loc[motif_ids].to_numpy()

    result = {}

    for i, loop_id in enumerate(loop_ids):

        loop_values = loops_mat[i]   # 1D array (cells)

        # Precompute masks ONCE
        mask_11 = loop_values == 11
        mask_10 = loop_values == 10
        mask_01 = loop_values == 1
        mask_00 = loop_values == 0

        result[loop_id] = {}

        for j, motif_id in enumerate(motif_ids):

            motif_values = motifs_mat[j]  # 1D array (cells)

            result[loop_id][motif_id] = {
                "1-1": motif_values[mask_11] if mask_11.any() else np.array([]),
                "1-0": motif_values[mask_10] if mask_10.any() else np.array([]),
                "0-1": motif_values[mask_01] if mask_01.any() else np.array([]),
                "0-0": motif_values[mask_00] if mask_00.any() else np.array([]),
            }

    return result
        

def load_window_split_by_tissue(window: str, metadata_df: pd.DataFrame):
    """
    returns a dictionary of dataframes
    """
    loops_path = f"data/new_time/hrs{window}_NNv1_time_matrix_loops.tsv"
    motifs_path = f"data/new_time/hrs{window}_NNv1_time_matrix_motifs.tsv"

    loops_df = pd.read_csv(loops_path, sep="\t", index_col=0)
    motifs_df = pd.read_csv(motifs_path, sep="\t", index_col=0)

    # numeric cleanup
    loops_df = loops_df.apply(pd.to_numeric, errors="coerce").dropna(axis=1)
    motifs_df = motifs_df.apply(pd.to_numeric, errors="coerce").dropna(axis=1)

    # align once
    common_cells = loops_df.columns.intersection(motifs_df.columns)
    loops_df = loops_df[common_cells]
    motifs_df = motifs_df[common_cells]

    assert loops_df.columns.equals(motifs_df.columns), "Loops_df and motifs_df have different columns (or the same columns in different order). This will interfere with downstream masking of the dataframes."

    # group cells by tissue
    grouped = {}
    for label, submeta in metadata_df.groupby("refined_annotation"):
        cells = submeta.index.intersection(common_cells)
        if len(cells) > 0:
            grouped[label] = (
                loops_df[cells],
                motifs_df[cells]
            )

    return grouped

# neural_labels = [
#     "Brain", "Neural", "Ventral_nerve_cord",
#     "Ventral_nerve_cord_prim", "Glia", "PNS_&_sense"
# ]

# neural_cells = set(
#     metadata_df.index[
#         metadata_df["refined_annotation"].isin(neural_labels)
#     ]
# )

def distributions(loop_ids: list[str], motif_ids: list[str], loops_df: pd.DataFrame, motifs_df: pd.DataFrame) -> dict[dict]:
    """
    Returns whole z-score distributions for all activity profiles (lists of values)
    Args:
        loop_ids (list[str]): Loop indentifiers (like 'L417')
        motif_ids (list[str]): motif indentifiers (like 'M0111-1.02')
        loops_df, motifs_df (pd.DataFrame): dataframes loaded by load_window()

    Returns:
        result (dict): a dictionary with 
    """
    result = {loop: {motif: {"1-1": [], "1-0": [], "0-1": [], "0-0": []} for motif in motif_ids} for loop in loop_ids}
    for loop_id in loop_ids:
        loop_values = loops_df.loc[loop_id]
        cells_11 = loop_values[loop_values == 11].index
        cells_10 = loop_values[loop_values == 10].index
        cells_01 = loop_values[loop_values == 1].index
        cells_00 = loop_values[loop_values == 0].index

        for motif_id in motif_ids:
            motif_values = motifs_df.loc[motif_id]

            result[loop_id][motif_id]["1-1"] = motif_values.loc[cells_11] if len(cells_11) > 0 else np.nan
            result[loop_id][motif_id]["1-0"] = motif_values.loc[cells_10] if len(cells_10) > 0 else np.nan
            result[loop_id][motif_id]["0-1"] = motif_values.loc[cells_01] if len(cells_01) > 0 else np.nan
            result[loop_id][motif_id]["0-0"] = motif_values.loc[cells_00] if len(cells_00) > 0 else np.nan
            
    return result

def difference_of_means(loop_id: str, loops_df: pd.DataFrame, motifs_df: pd.DataFrame) -> tuple[str, dict]:
    # niech to po prostu zwraca różnicę
    """
    MOTIF ENRICHEMENT DATA PREPARATION
    The basic loop_processing_function.
    Averages chromVAR scores in groups 1-1 and other for all motifs for a given loop and returns the difference.
    
    Args:
        loop_id (str): Loop indentifier (like 'L417')
        loops_df, motifs_df (pd.DataFrame): dataframes loaded by load_winodow()

    Returns:
        loop_id (str): Loop indentifier (like 'L417')   # this is good for parallel execution???
        result (dict): a dictionary with motif names as keys and the difference between mean z-scores in cell populations as values.
    """
    result = {}  # Change to dict to store motif_id as key

    loop_values = loops_df.loc[loop_id]
    cells_11 = loop_values[loop_values == 11].index
    cells_other = loop_values.index.difference(cells_11)

    for motif_id in motifs_df.index:
        motif_values = motifs_df.loc[motif_id]
        mean_11 = np.mean(motif_values.loc[cells_11]) if len(cells_11) > 0 else np.nan
        mean_other = np.mean(motif_values.loc[cells_other]) if len(cells_other) > 0 else np.nan
        result[motif_id] = np.subtract(mean_11, mean_other)

    return loop_id, result

def loop_wasserstein(loop_id: str, loops_df: pd.DataFrame, motifs_df: pd.DataFrame) -> tuple[str, dict]:
    """
    Alternative loop_processing_function.
    Computes the wasserstein disctance chromVAR scores in groups 1-1 and other for all motifs for a given loop.
    
    Args:
        loop_id (str): Loop indentifier (like 'L417')
        loops_df, motifs_df (pd.DataFrame): dataframes loaded by load_window()

    Returns:
        loop_id (str): Loop indentifier (like 'L417')   # this is good for parallel execution???
        result (dict): a dictionary with motif names as keys and the Wasserstein distance between z-score distributions in cell populations as values.
    """
    result = {}  # Change to dict to store motif_id as key

    loop_values = loops_df.loc[loop_id]
    cells_11 = loop_values[loop_values == 11].index
    cells_other = loop_values.index.difference(cells_11)

    for motif_id in motifs_df.index:
        motif_values = motifs_df.loc[motif_id]
        # mean_11 = np.mean(motif_values.loc[cells_11]) if len(cells_11) > 0 else np.nan
        # mean_other = np.mean(motif_values.loc[cells_other]) if len(cells_other) > 0 else np.nan
        # result[motif_id] = np.subtract(mean_11, mean_other)
        one_one_values = motif_values.loc[cells_11] if len(cells_11) > 0 else np.nan
        other_values = motif_values.loc[cells_other] if len(cells_other) > 0 else np.nan
        result[motif_id] = wasserstein_distance(one_one_values, other_values)

    return loop_id, result

def motif_enrichement_parallel(window: str, loop_processing_function: Callable = difference_of_means, n_workers=6)-> pd.DataFrame: 
    """
    Parallel computation of motif enrichement data

    Args:
        window (str): timepoint to process (eg. 'hrs06-08')
        loop_processing_function (Callable): a function computing some kind of distance between motif score distribution in "1-1" and "other".

    Returns:
        motif_enrichments (pd.DataFrame):  a DataFrame with loops (rows), motifs (columns) and "1-1" - "other" difference (values).
    """

    loops_df, motifs_df = load_window(window)
    
    # data_3D = defaultdict(lambda: defaultdict(dict)) # previous implemenation
    all_data = defaultdict(dict)

    # Submit tasks
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(loop_processing_function, loop_id, loops_df, motifs_df): loop_id for loop_id in loops_df.index}

        for fut in tqdm.tqdm(as_completed(futures), total=len(futures)):
            loop_id, result = fut.result()
            all_data[loop_id] = result

    motif_enrichments = pd.DataFrame.from_dict(all_data, orient='index')
    return motif_enrichments

def process_loop(loop_id, loops_df, motifs_df):
    """
    OBSOLETE!
    The basic loop_processing_function.
    Averages chromVAR scores in groups 1-1 and other for all motifs for a given loop.
    
    Args:
        loop_id (str): Loop indentifier (like 'L417')
        loops_df, motifs_df (pd.DataFrame): dataframes loaded by load_winodow()

    Returns:
        loop_id (str): Loop indentifier (like 'L417')   # this is good for parallel execution???
        result (dict): a two-level dictionary where
            keys are motif names
            values are dictionaries of average score in populations '1-1' and 'other'
    """
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

def process_window_parallel(window: str, loop_processing_function=process_loop, n_workers=6)-> defaultdict: 
    # AJk 21.02.26: Czy ten defaultdict jest tutaj potrzebny?
    """
    OBSOLETE!
    Parallel computation of data_3D using multiple processes.    

    Args:
        window (str): timepoint to process (eg. 'hrs06-08')

    Returns:
        data_3D (defaultdict): A three-level dictionary
    """
    
    loops_df, motifs_df = load_window(window)
    
    data_3D = defaultdict(lambda: defaultdict(dict))
    # print(f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}\t\tReal business begins\n")

    # Submit tasks
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(loop_processing_function, loop_id, loops_df, motifs_df): loop_id for loop_id in loops_df.index}

        for fut in tqdm.tqdm(as_completed(futures), total=len(futures)):
            loop_id, result = fut.result()
            data_3D[loop_id] = result

    return data_3D

def convert_2D(data_3D: dict)-> pd.DataFrame:
    """ OBSOLETE!
    For each (loop, motif) pair: compute the difference in mean chromVAR scores between populations of cells '1-1' and 'other'. 

    Args:  
        data_3D (dict): output of process_window_parallel()

    Returns:
        data_diff: a DataFrame with loops (rows), motifs (columns) and "1-1" - "other" difference (values).
        (data_diff is later reffered to as "motif enrichement matrix")
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



def main():
    print("Utilities module :P - use this for imports only")

if __name__ == "__main__":
    main()
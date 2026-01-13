import pandas as pd

def compose_windows(tissue, windows=["06-08", "10-12", "14-16"]):
    """
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
    print(f"Created a composite vector with {num_values} distinct values")

    X_new.drop('window', axis=1, inplace=True) # we don't want to use 'window' for prediction

    return X_new, y_new, composite

def main():
    print("Utilities module :P - this is for imports only")

if __name__ == "__main__":
    main()
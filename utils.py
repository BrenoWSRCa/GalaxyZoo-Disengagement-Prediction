from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from sklearn.metrics import classification_report
from tqdm import tqdm


def get_arrays(df: pd.DataFrame, window_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This method returns three arrays corresponding to the lines being predicted, the windows and their labels
    Args:
        df: dataset with columns ('id', 'num_hit', 'disengage')
        window_size: size of the sliding window

    Returns:
        - array with index of lines from *df* corresponding to each window
        - array of feature windows, each line is a window
        - array of labels for each feature window

    """
    ignored = 0
    features = []
    labels = []
    idxs = df['id'].unique()
    l_numbers = []
    for idx in idxs:
        df_slice = df.loc[df['id'] == idx, :]
        if df_slice[['num_hit']].values.shape[0] < window_size:
            ignored += 1
            continue
        f = window_stack(df_slice[['num_hit']].values, window_size)
        l = df_slice[['disengage']].values[(window_size - 1):, :]

        filter = ~pd.isnull(l).flatten()
        f = f[filter, :]
        l = l[filter, :]
        n = df_slice.iloc[(window_size - 1):, :].index[filter]
        if l.shape[0] == 0:
            ignored += 1
            continue
        features.append(f)
        labels.append(l)
        l_numbers.append(n.values.reshape(-1, 1))
    # Sanity check
    print(f'Discarded samples ratio {100 * ignored / len(idxs):6.2f}%')
    return np.vstack(l_numbers).flatten(), np.vstack(features), np.vstack(labels)


def window_stack(a: np.ndarray, width: int, step_size: int = 1) -> np.ndarray:
    """
    Creates window stacks given parameters
    Args:
        a: array to be split
        width: window size
        step_size: step size between beginning of two consecutive windows

    Returns:
        Array with stacked windows
    """
    return np.vstack([a[i: i + width: step_size, 0] for i in range(0, a.shape[0] - width + 1)])


def report_model(y_test: np.ndarray,
                 pred: np.ndarray,
                 sample_weight: np.ndarray,
                 name,
                 window_size='--',
                 clf_table=None) -> None:
    """
    Prints report and updates *clf_table*.
    Note: changes *clf_table* in place.
    Args:
        y_test: array of test labels
        pred: array of predictions
        sample_weight: array of sample weights
        name: name of the predictor
        window_size: value to insert in the column window_size of *clf_table*
        clf_table: pandas dataframe to store results
    """
    report = classification_report(y_test, pred, sample_weight=sample_weight)
    print(f'{name}:')
    print(report)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    acc = metrics.accuracy_score(y_test, pred)
    if clf_table is not None:
        clf_table.loc[(name, window_size), ['accuracy', 'AUC']] = acc, auc


def prepare_sequence(df: pd.DataFrame, nan_replacement: int = 1) -> Tuple[List[torch.tensor], List[torch.tensor]]:
    """
    Prepare training sequence for RNN models
    Args:
        df:
        nan_replacement:

    Returns:
        - List of feature tensors
        - List of label tensors
    """
    # print(df.describe())
    features = []
    labels = []

    for (idx, session), df_slice in tqdm(df.groupby(by=['id', 'session']), desc='Preparing dataset'):
        # print(f'ID: {idx}, SESSION: {session}')
        features.append(torch.tensor(df_slice['num_hit'].values, dtype=torch.long))
        label = df_slice['disengage'].values
        label[np.isnan(label)] = nan_replacement
        labels.append(torch.tensor(label, dtype=torch.long))
    return features, labels


def calc_session_boundaries(df_slice: pd.DataFrame, df: pd.DataFrame, max_session_gap: pd.Timedelta) -> None:
    """
    Changes *df* inplace. Adds session and disengagement values.
    Args:
        df_slice:
        df:
        max_session_gap
    """
    df.loc[df_slice.index, 'session'] = (df_slice.stamp - df_slice.stamp.shift(1) > max_session_gap).cumsum() + 1


def expand_session(df: pd.DataFrame, prediction_window, step_size) -> pd.DataFrame:
    """
    Creates a new dataset by adding empty rows and disengagement label.
    There are several contributions at the same minute performed by the same id. So I count those contributions.
    Args:
        df: pandas DataFrame with columns 'id', 'session' and 'stamp'

    Returns:
        DataFrame with the expanded sessions.
    """

    stamp_counts_df = df.groupby('stamp')['stamp'].count()
    disengagement_df = pd.Series(stamp_counts_df.index).apply(lambda x: df['stamp'].max() - x < prediction_window)
    disengagement_df.index = stamp_counts_df.index
    index = np.arange(df['stamp'].min(), df['stamp'].max() + step_size / 2, step_size)
    stamp_counts = np.where(np.isin(index, stamp_counts_df.index), stamp_counts_df[index], 0)
    disengagement = np.where(np.isin(index, stamp_counts_df.index), disengagement_df[index], np.nan).astype(np.float16)

    return pd.DataFrame({'id': df['id'].iloc[0],
                         'session': df['session'].iloc[0],
                         'stamp': index,
                         'num_hit': stamp_counts,
                         'disengage': disengagement})

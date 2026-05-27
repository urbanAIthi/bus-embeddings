from typing import List, Tuple

import pandas as pd


def generate_split_boundaries(
    start_time: pd.Timestamp, end_time: pd.Timestamp, val_size: float, n_splits: int
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Generate temporal cross-validation split boundaries.

    Creates n_splits with expanding training windows and fixed-size validation/test periods.

    Args:
        start_time: Start of the overall time range
        end_time: End of the overall time range
        val_size: Fraction of each test segment to use for validation (0 to 1)
        n_splits: Number of splits to generate

    Returns:
        List of (train_start, train_end, val_end, test_end) tuples for each split.
        Each split uses:
        - Training: start_time to train_end (expanding window)
        - Validation: train_end to val_end (fixed size per split)
        - Test: val_end to test_end (fixed size per split)
    """
    split_size = (end_time - start_time) / (n_splits + 1)
    val_size_absolute = split_size * val_size
    test_size_absolute = split_size * (1 - val_size)

    splits = []
    for i in range(n_splits):
        train_end = start_time + (i + 1) * split_size
        val_end = train_end + val_size_absolute
        test_end = val_end + test_size_absolute
        splits.append((start_time, train_end, val_end, test_end))

    return splits


import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# validation.py dosyasını bul
project_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_dir / "src"))

from validation import validate_pm25_window


# Geçerli 24 saatlik örnek
@pytest.fixture
def valid_window():

    timestamps = pd.date_range(
        "2016-07-24 00:00",
        periods=24,
        freq="h"
    )

    values = np.arange(24, dtype=float) + 40

    return timestamps, values


# TEST 1 — 23 saatlik veri
def test_short_window(valid_window):

    timestamps, values = valid_window

    with pytest.raises(ValueError):
        validate_pm25_window(
            timestamps[:23],
            values[:23]
        )


# TEST 2 — NaN ve Inf
def test_nan_inf(valid_window):

    timestamps, values = valid_window

    for invalid_value in [np.nan, np.inf, -np.inf]:

        invalid_values = values.copy()
        invalid_values[5] = invalid_value

        with pytest.raises(ValueError):
            validate_pm25_window(
                timestamps,
                invalid_values
            )


# TEST 3 — Zaman boşluğu
def test_timestamp_gap(valid_window):

    timestamps, values = valid_window

    gap_timestamps = list(timestamps)

    gap_timestamps[12:] = [
        t + pd.Timedelta(hours=1)
        for t in gap_timestamps[12:]
    ]

    with pytest.raises(ValueError):
        validate_pm25_window(
            gap_timestamps,
            values
        )

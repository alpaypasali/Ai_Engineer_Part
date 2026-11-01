import numpy as np
import pandas as pd


def validate_pm25_window(timestamps, values, sequence_length=24):

    # 1. GİRDİ UZUNLUĞU KONTROLÜ
    if len(values) != sequence_length:
        raise ValueError(
            f"Beklenen {sequence_length} PM2.5 ölçümü, "
            f"gelen {len(values)} ölçüm."
        )

    if len(timestamps) != sequence_length:
        raise ValueError(
            f"Beklenen {sequence_length} zaman damgası, "
            f"gelen {len(timestamps)} zaman damgası."
        )

    # 2. NaN / Inf KONTROLÜ
    try:
        values = np.asarray(values, dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise ValueError(
            "PM2.5 değerleri sayısal olmalıdır."
        ) from e

    if values.ndim != 1:
        raise ValueError(
            "PM2.5 değerleri tek boyutlu olmalıdır."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "PM2.5 değerleri NaN veya Inf içeremez."
        )

    # 3. ARDIŞIK SAAT KONTROLÜ
    try:
        timestamps = pd.DatetimeIndex(
            pd.to_datetime(timestamps, errors="coerce")
        )
    except (ValueError, TypeError) as e:
        raise ValueError(
            "Geçersiz zaman damgası biçimi."
        ) from e

    if timestamps.isna().any():
        raise ValueError(
            "Zaman damgaları eksik veya geçersiz olamaz."
        )

    time_differences = timestamps[1:] - timestamps[:-1]

    if not (time_differences == pd.Timedelta(hours=1)).all():
        raise ValueError(
            "Zaman damgaları ardışık ve saatlik olmalıdır. "
            "Eksik, tekrarlanan veya sırasız saat tespit edildi."
        )

    # Bütün kontroller başarılı
    return True
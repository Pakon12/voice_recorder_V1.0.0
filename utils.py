import numpy as np
from typing import Tuple

def calculate_db(data: np.ndarray) -> float:
    """Calculate decibel level of audio data."""
    rms = np.sqrt(np.mean(data.astype(float) ** 2))
    return 20 * np.log10(rms + 1e-9)

def normalize_level(db: float, threshold: float) -> float:
    """Normalize dB level to 0-1 range."""
    normalized = (db - threshold) / 40
    return max(0, min(1, normalized))
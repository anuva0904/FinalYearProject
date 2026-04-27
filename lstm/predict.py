import numpy as np
import os
from config import DATA_PROC, LSTM_DIR, STOPS, N
from utils.preprocess import base_travel_time, get_rush_factor

_model        = None
_scaler_min   = None
_scaler_scale = None


def _load_model():
    global _model, _scaler_min, _scaler_scale
    if _model is not None:
        return
    from tensorflow.keras.models import load_model as _lm
    path = os.path.join(LSTM_DIR, "best_model.keras")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"LSTM model not found at {path}. Run: python lstm/train_lstm.py")
    _model        = _lm(path)
    _scaler_min   = np.load(os.path.join(DATA_PROC, "scaler_min.npy"))
    _scaler_scale = np.load(os.path.join(DATA_PROC, "scaler_scale.npy"))


def build_time_matrix(hour=8, is_fog=0, is_rain=0):
    """
    Loads real-world travel time matrix built from OpenStreetMap.
    """

    path = os.path.join(DATA_PROC, "real_time_matrix.npy")

    if not os.path.exists(path):
        raise FileNotFoundError(
            "Real travel time matrix not found. Run:\n"
            "python real_data/build_travel_time_matrix.py"
        )

    print(f"[OSM] Loading real travel time matrix for hour {hour}")

    matrix = np.load(path)

    # Simulate dynamic traffic impact based on time of day
    np.random.seed(hour)
    if hour in [8, 9, 17, 18, 19]:
        multiplier = np.random.uniform(1.3, 1.6, size=matrix.shape)
    else:
        multiplier = np.random.uniform(0.8, 1.0, size=matrix.shape)
        
    np.fill_diagonal(multiplier, 1.0)
    matrix = matrix * multiplier

    return matrix
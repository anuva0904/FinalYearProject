import numpy as np
import pandas as pd
import os
from config import STOPS, N, DATA_RAW, DATA_PROC, SIM_DAYS, SEQ_LEN, RANDOM_SEED

np.random.seed(RANDOM_SEED)


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in km."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def base_travel_time(i, j, speed_kmh=25.0):
    """Free-flow travel time in minutes between stop i and stop j."""
    dist = haversine(STOPS[i]["lat"], STOPS[i]["lon"],
                     STOPS[j]["lat"], STOPS[j]["lon"])
    return max(1.0, (dist / speed_kmh) * 60.0)


def _rush_multiplier(hour, dow):
    if 8 <= hour <= 10:
        m = np.random.uniform(1.6, 2.2)
    elif 17 <= hour <= 20:
        m = np.random.uniform(1.5, 2.0)
    elif hour >= 22 or hour <= 5:
        m = np.random.uniform(0.5, 0.8)
    else:
        m = np.random.uniform(0.9, 1.2)
    if dow >= 5:
        m *= 0.75
    return m


def generate_traffic_data():
    os.makedirs(DATA_RAW, exist_ok=True)
    timestamps = pd.date_range("2024-01-01", periods=SIM_DAYS * 96, freq="15min")
    rng = np.random.default_rng(RANDOM_SEED)
    records = []

    # Pre-compute all (i,j) pairs
    pairs = [(i, j) for i in range(N) for j in range(N) if i != j]

    for ts in timestamps:
        hour = ts.hour
        dow  = ts.dayofweek
        rush = _rush_multiplier(hour, dow)

        # Sample 40 random pairs per timestamp
        idxs = rng.choice(len(pairs), size=40, replace=False)
        for idx in idxs:
            i, j = pairs[idx]
            base  = base_travel_time(i, j)
            noise = rng.normal(0, 0.05 * base)
            t     = max(1.0, base * rush + noise)
            records.append({
                "timestamp":       ts,
                "stop_from":       i,
                "stop_to":         j,
                "travel_time_min": round(t, 2),
                "hour":            hour,
                "day_of_week":     dow,
            })

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(DATA_RAW, "traffic_data.csv"), index=False)
    print(f"[preprocess] Traffic: {len(df):,} rows → data/raw/traffic_data.csv")
    return df


def generate_weather_data():
    os.makedirs(DATA_RAW, exist_ok=True)
    timestamps = pd.date_range("2024-01-01", periods=SIM_DAYS * 96, freq="15min")
    rng = np.random.default_rng(RANDOM_SEED + 1)
    records = []
    for ts in timestamps:
        hour = ts.hour
        temp = float(rng.normal(15 if 10 <= hour <= 18 else 8, 3))
        fog  = int(hour <= 8 and rng.random() < 0.30)
        rain = int(rng.random() < 0.05)
        records.append({"timestamp": ts, "temperature_c": round(temp, 1),
                         "is_fog": fog, "is_rain": rain})
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(DATA_RAW, "weather_data.csv"), index=False)
    print(f"[preprocess] Weather: {len(df):,} rows → data/raw/weather_data.csv")
    return df


def save_bus_stops():
    os.makedirs(DATA_RAW, exist_ok=True)
    df = pd.DataFrame(STOPS)
    df.to_csv(os.path.join(DATA_RAW, "bus_stops.csv"), index=False)
    print(f"[preprocess] {N} bus stops → data/raw/bus_stops.csv")
    return df


def build_feature_sequences():
    """
    Merge traffic + weather → engineer features → build LSTM sequences.
    Saves X.npy (n, SEQ_LEN, N_FEATURES) and y.npy (n,) to data/processed/.
    """
    traffic = pd.read_csv(os.path.join(DATA_RAW, "traffic_data.csv"),
                          parse_dates=["timestamp"])
    weather = pd.read_csv(os.path.join(DATA_RAW, "weather_data.csv"),
                          parse_dates=["timestamp"])

    traffic = traffic.sort_values("timestamp").reset_index(drop=True)
    weather = weather.sort_values("timestamp").reset_index(drop=True)
    df = pd.merge_asof(traffic, weather, on="timestamp")

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["day_of_week"] / 7)

    feature_cols = ["travel_time_min", "hour_sin", "hour_cos",
                    "dow_sin", "dow_cos", "is_fog", "is_rain"]
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    X_all, y_all = [], []
    for (_, _), grp in df.groupby(["stop_from", "stop_to"]):
        grp  = grp.sort_values("timestamp").reset_index(drop=True)
        vals = grp[feature_cols].values.astype(np.float32)
        if len(vals) <= SEQ_LEN:
            continue
        for k in range(SEQ_LEN, len(vals)):
            X_all.append(vals[k - SEQ_LEN: k])
            y_all.append(vals[k, 0])

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.float32)

    os.makedirs(DATA_PROC, exist_ok=True)
    np.save(os.path.join(DATA_PROC, "X.npy"), X)
    np.save(os.path.join(DATA_PROC, "y.npy"), y)
    print(f"[preprocess] Sequences: X={X.shape}, y={y.shape} → data/processed/")
    return X, y


def get_rush_factor(hour, is_fog=0, is_rain=0):
    """Deterministic rush multiplier for inference (no randomness)."""
    if 8 <= hour <= 10:   m = 1.90
    elif 17 <= hour <= 20: m = 1.70
    elif hour >= 22 or hour <= 5: m = 0.60
    else: m = 1.05
    if is_fog:  m *= 1.30
    if is_rain: m *= 1.20
    return m


if __name__ == "__main__":
    save_bus_stops()
    generate_traffic_data()
    generate_weather_data()
    build_feature_sequences()

# config.py  — single source of truth for all constants
import os

# ── Paths ────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_RAW      = os.path.join(BASE_DIR, "data", "raw")
DATA_PROC     = os.path.join(BASE_DIR, "data", "processed")
LSTM_DIR      = os.path.join(BASE_DIR, "lstm")
DQN_DIR       = os.path.join(BASE_DIR, "dqn")
RESULTS_DIR   = os.path.join(BASE_DIR, "results")

# ── Bus network ──────────────────────────────────────────────────
STOPS = [
    {"id": 0,  "name": "Kashmere Gate",       "lat": 28.6671, "lon": 77.2281},
    {"id": 1,  "name": "Red Fort",            "lat": 28.6562, "lon": 77.2410},
    {"id": 2,  "name": "Chandni Chowk",       "lat": 28.6506, "lon": 77.2334},
    {"id": 3,  "name": "New Delhi Station",   "lat": 28.6420, "lon": 77.2195},
    {"id": 4,  "name": "Connaught Place",     "lat": 28.6289, "lon": 77.2065},
    {"id": 5,  "name": "India Gate",          "lat": 28.6129, "lon": 77.2295},
    {"id": 6,  "name": "Lajpat Nagar",        "lat": 28.5700, "lon": 77.2440},
    {"id": 7,  "name": "Saket",               "lat": 28.5245, "lon": 77.2066},
    {"id": 8,  "name": "Hauz Khas",           "lat": 28.5494, "lon": 77.2001},
    {"id": 9,  "name": "IIT Delhi",           "lat": 28.5459, "lon": 77.1926},
    {"id": 10, "name": "Dwarka Sec 21",       "lat": 28.5530, "lon": 77.0588},
    {"id": 11, "name": "Janakpuri",           "lat": 28.6219, "lon": 77.0839},
    {"id": 12, "name": "Rajouri Garden",      "lat": 28.6489, "lon": 77.1219},
    {"id": 13, "name": "Karol Bagh",          "lat": 28.6527, "lon": 77.1900},
    {"id": 14, "name": "Patel Nagar",         "lat": 28.6511, "lon": 77.1700},
    {"id": 15, "name": "Rohini Sec 18",       "lat": 28.7357, "lon": 77.1067},
    {"id": 16, "name": "Pitampura",           "lat": 28.7005, "lon": 77.1317},
    {"id": 17, "name": "Netaji Subhash Place","lat": 28.6928, "lon": 77.1531},
    {"id": 18, "name": "Azadpur",             "lat": 28.7100, "lon": 77.1800},
    {"id": 19, "name": "GTB Nagar",           "lat": 28.7017, "lon": 77.2065},
    {"id": 20, "name": "Shahdara",            "lat": 28.6718, "lon": 77.2888},
    {"id": 21, "name": "Dilshad Garden",      "lat": 28.6817, "lon": 77.3208},
    {"id": 22, "name": "Anand Vihar",         "lat": 28.6469, "lon": 77.3151},
    {"id": 23, "name": "Laxmi Nagar",         "lat": 28.6322, "lon": 77.2777},
    {"id": 24, "name": "Noida Sec 15",        "lat": 28.5853, "lon": 77.3210},
]
N = len(STOPS)

# ── Data generation ──────────────────────────────────────────────
SIM_DAYS      = 60
SEQ_LEN       = 10
N_FEATURES    = 7   # travel_time, hour_sin, hour_cos, dow_sin, dow_cos, is_fog, is_rain
RANDOM_SEED   = 42

# ── LSTM ─────────────────────────────────────────────────────────
LSTM_UNITS    = 50
LSTM_EPOCHS   = 100
LSTM_BATCH    = 32
LSTM_LR       = 0.001
LSTM_PATIENCE = 10

# ── GA ───────────────────────────────────────────────────────────
GA_POP        = 200
GA_GENS       = 100
GA_CX_PROB    = 0.7
GA_MUT_PROB   = 0.2
GA_LAMBDA1    = 0.3
GA_LAMBDA2    = 0.2

# ── DQN ──────────────────────────────────────────────────────────
DQN_TIMESTEPS = 150_000
DQN_LR        = 1e-3
DQN_GAMMA     = 0.9
DQN_EPS_START = 0.9
DQN_EPS_END   = 0.1
DQN_EPS_FRAC  = 0.5
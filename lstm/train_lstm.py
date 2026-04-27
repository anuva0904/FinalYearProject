import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from config import DATA_PROC, LSTM_DIR, RESULTS_DIR
from config import LSTM_UNITS, LSTM_EPOCHS, LSTM_BATCH, LSTM_LR, LSTM_PATIENCE, SEQ_LEN


def build_model(seq_len, n_features):
    model = Sequential([
        LSTM(LSTM_UNITS, return_sequences=True,
             input_shape=(seq_len, n_features)),
        Dropout(0.2),
        LSTM(LSTM_UNITS, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation="relu"),
        Dense(1),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LSTM_LR),
        loss="mse",
        metrics=["mae"],
    )
    return model


def train():
    os.makedirs(LSTM_DIR,    exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    X = np.load(os.path.join(DATA_PROC, "X.npy"))
    y = np.load(os.path.join(DATA_PROC, "y.npy"))
    print(f"[LSTM] Loaded X={X.shape}, y={y.shape}")

    # Scale only the target (y); X features are already normalised cyclically
    scaler = MinMaxScaler()
    y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()

    np.save(os.path.join(DATA_PROC, "scaler_min.npy"),   scaler.data_min_)
    np.save(os.path.join(DATA_PROC, "scaler_scale.npy"), scaler.scale_)

    # Chronological split (70 / 30) — no shuffle to avoid data leakage
    split = int(len(X) * 0.70)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y_scaled[:split], y_scaled[split:]
    print(f"[LSTM] Train={len(X_train)}, Test={len(X_test)}")

    model_path = os.path.join(LSTM_DIR, "best_model.keras")
    callbacks  = [
        EarlyStopping(patience=LSTM_PATIENCE, restore_best_weights=True, verbose=1),
        ModelCheckpoint(model_path, save_best_only=True, verbose=0),
    ]

    
    model   = build_model(X.shape[1], X.shape[2])
    history = model.fit(
        X_train, y_train,
        epochs=LSTM_EPOCHS,
        batch_size=LSTM_BATCH,
        validation_split=0.15,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Evaluation ────────────────────────────────────────────────
    y_pred_s = model.predict(X_test, verbose=0).flatten()
    y_pred   = y_pred_s / scaler.scale_[0] + scaler.data_min_[0]
    y_true   = y_test   / scaler.scale_[0] + scaler.data_min_[0]
    mae      = mean_absolute_error(y_true, y_pred)
    print(f"[LSTM] Test MAE: {mae:.3f} minutes")

    # ── Ablation: effect of weather features ─────────────────────
    X_nw         = X.copy()
    X_nw[:, :, 5] = 0.0  # zero out is_fog
    X_nw[:, :, 6] = 0.0  # zero out is_rain
    Xnw_tr, Xnw_te = X_nw[:split], X_nw[split:]

    model_nw = build_model(X.shape[1], X.shape[2])
    model_nw.fit(Xnw_tr, y_train, epochs=30, batch_size=LSTM_BATCH,
                 validation_split=0.15, verbose=0)
    y_pred_nw = (model_nw.predict(Xnw_te, verbose=0).flatten()
                 / scaler.scale_[0] + scaler.data_min_[0])
    mae_nw    = mean_absolute_error(y_true, y_pred_nw)
    improv    = (mae_nw - mae) / mae_nw * 100
    print(f"[LSTM] MAE without weather: {mae_nw:.3f} min  "
          f"| Improvement: {improv:.1f}%")

    # ── Plots ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"],     label="Train")
    axes[0].plot(history.history["val_loss"], label="Val")
    axes[0].set_title("LSTM training loss"); axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE"); axes[0].legend()

    axes[1].scatter(y_true[:300], y_pred[:300], alpha=0.35, s=8)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[1].plot(lims, lims, "r--", lw=1)
    axes[1].set_title("Predicted vs Actual")
    axes[1].set_xlabel("Actual (min)"); axes[1].set_ylabel("Predicted (min)")

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "lstm_performance.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[LSTM] Plot saved → {plot_path}")

    return mae


if __name__ == "__main__":
    train()
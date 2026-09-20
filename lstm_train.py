# ============================================================
# SmartFlow - Multi-Output LSTM Training
# ============================================================

import os
import pickle

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from config import (
    BASE_FILE,
    LSTM_MODEL,
    SCALER_FILE,
    TIME_STEPS,
)

from traffic_features import calculate_traffic_load


# ============================================================
# CHECK HISTORICAL FILE
# ============================================================

if not os.path.exists(BASE_FILE):
    raise FileNotFoundError(
        f"Historical data file not found: {BASE_FILE}"
    )


# ============================================================
# LOAD DATA
# ============================================================

print("========================================")
print("LOADING HISTORICAL DATA")
print("========================================")

df = pd.read_csv(BASE_FILE)

print(f"Loaded {len(df)} historical rows.")


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "cars",
    "motorcycles",
    "buses",
    "trucks",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}\n"
        f"Available columns: {list(df.columns)}"
    )


# ============================================================
# CLEAN DATA
# ============================================================

for column in required_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    ).fillna(0)


# ============================================================
# TOTAL VEHICLES
# ============================================================

df["total_vehicles"] = (
    df["cars"]
    + df["motorcycles"]
    + df["buses"]
    + df["trucks"]
)


# ============================================================
# WEIGHTED TRAFFIC LOAD
# ============================================================

df["traffic_load"] = df.apply(
    lambda row: calculate_traffic_load(
        cars=row["cars"],
        motorcycles=row["motorcycles"],
        buses=row["buses"],
        trucks=row["trucks"],
    ),
    axis=1,
)


print("\nFirst few calculated values:")
print(
    df[
        [
            "cars",
            "motorcycles",
            "buses",
            "trucks",
            "total_vehicles",
            "traffic_load",
        ]
    ].head()
)


# ============================================================
# CHECK DATA SIZE
# ============================================================

if len(df) <= TIME_STEPS:
    raise ValueError(
        f"Not enough historical data.\n"
        f"TIME_STEPS = {TIME_STEPS}\n"
        f"Rows available = {len(df)}\n"
        f"Need at least {TIME_STEPS + 1} rows."
    )


# ============================================================
# INPUT / TARGET
# ============================================================

input_values = df[
    ["traffic_load"]
].values

target_values = df[
    ["total_vehicles", "traffic_load"]
].values


# ============================================================
# SCALING
# ============================================================

input_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

scaled_input = input_scaler.fit_transform(
    input_values
)

scaled_targets = target_scaler.fit_transform(
    target_values
)


# ============================================================
# CREATE SEQUENCES
# ============================================================

X = []
y = []

for i in range(TIME_STEPS, len(df)):

    X.append(
        scaled_input[
            i - TIME_STEPS:i,
            0
        ]
    )

    y.append(
        scaled_targets[i]
    )


X = np.array(X)
y = np.array(y)


# LSTM input shape:
# samples, time steps, features

X = X.reshape(
    X.shape[0],
    X.shape[1],
    1
)


print("\n========================================")
print("TRAINING DATA")
print("========================================")

print("X shape:", X.shape)
print("y shape:", y.shape)


# ============================================================
# MODEL
# ============================================================

model = Sequential(
    [
        LSTM(
            64,
            return_sequences=True,
            input_shape=(TIME_STEPS, 1),
        ),

        Dropout(0.2),

        LSTM(32),

        Dropout(0.2),

        Dense(
            16,
            activation="relu",
        ),

        # Two outputs:
        #
        # 1. total vehicles
        # 2. traffic load

        Dense(2),
    ]
)


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"],
)


model.summary()


# ============================================================
# TRAIN
# ============================================================

print("\n========================================")
print("STARTING LSTM TRAINING")
print("========================================")

history = model.fit(
    X,
    y,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    verbose=1,
)


# ============================================================
# SAVE MODEL
# ============================================================

model.save(LSTM_MODEL)


# ============================================================
# SAVE SCALERS
# ============================================================

scalers = {
    "input_scaler": input_scaler,
    "target_scaler": target_scaler,
}

with open(
    SCALER_FILE,
    "wb",
) as file:

    pickle.dump(
        scalers,
        file,
    )


# ============================================================
# COMPLETED
# ============================================================

print("\n========================================")
print("LSTM TRAINING COMPLETED")
print("========================================")

print(f"Model saved: {LSTM_MODEL}")
print(f"Scalers saved: {SCALER_FILE}")

print("\nModel outputs:")
print("1. Future total vehicles")
print("2. Future traffic load")

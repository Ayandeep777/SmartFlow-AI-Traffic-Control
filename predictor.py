# ============================================================
# SmartFlow - LSTM Prediction Pipeline
# ============================================================

import os
import pickle

import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model

from config import (
    BASE_FILE,
    LSTM_MODEL,
    SCALER_FILE,
    TIME_STEPS,
)

from traffic_features import calculate_traffic_load

from traffic_classifier import (
    calculate_thresholds,
    classify_traffic,
)

from optimizer import pso_optimize


# ============================================================
# LOAD MODEL
# ============================================================

if not os.path.exists(LSTM_MODEL):
    raise FileNotFoundError(
        f"LSTM model not found: {LSTM_MODEL}\n"
        f"Run lstm_train.py first."
    )

if not os.path.exists(SCALER_FILE):
    raise FileNotFoundError(
        f"Scaler file not found: {SCALER_FILE}\n"
        f"Run lstm_train.py first."
    )

print("Loading LSTM model...")

model = load_model(
    LSTM_MODEL
)

print("LSTM model loaded.")


# ============================================================
# LOAD SCALERS
# ============================================================

with open(
    SCALER_FILE,
    "rb",
) as file:

    scalers = pickle.load(file)


input_scaler = scalers["input_scaler"]

target_scaler = scalers["target_scaler"]


# ============================================================
# LOAD HISTORICAL DATA
# ============================================================

if not os.path.exists(BASE_FILE):
    raise FileNotFoundError(
        f"Historical data file not found: {BASE_FILE}"
    )

historical_df = pd.read_csv(
    BASE_FILE
)


# ============================================================
# CLEAN HISTORICAL DATA
# ============================================================

required_columns = [
    "cars",
    "motorcycles",
    "buses",
    "trucks",
]

for column in required_columns:

    if column not in historical_df.columns:
        raise ValueError(
            f"Required column '{column}' "
            f"not found in {BASE_FILE}"
        )

    historical_df[column] = pd.to_numeric(
        historical_df[column],
        errors="coerce",
    ).fillna(0)


# ============================================================
# CALCULATE HISTORICAL TRAFFIC LOAD
# ============================================================

historical_df["traffic_load"] = historical_df.apply(
    lambda row: calculate_traffic_load(
        cars=row["cars"],
        motorcycles=row["motorcycles"],
        buses=row["buses"],
        trucks=row["trucks"],
    ),
    axis=1,
)


# ============================================================
# CALCULATE TRAFFIC THRESHOLDS
# ============================================================

thresholds = calculate_thresholds(
    historical_df["traffic_load"].values
)


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_from_live(live_loads):

    """
    Predict future traffic using the latest
    TIME_STEPS weighted traffic-load observations.

    Returns:
        predicted_total_vehicles
        predicted_traffic_load
        traffic_class
        low_threshold
        high_threshold
        signal
    """

    # --------------------------------------------------------
    # Validate live data
    # --------------------------------------------------------

    if live_loads is None:
        raise ValueError(
            "Live traffic data is None."
        )

    if len(live_loads) < TIME_STEPS:
        raise ValueError(
            f"Need at least {TIME_STEPS} "
            f"live observations. "
            f"Currently available: {len(live_loads)}"
        )


    # --------------------------------------------------------
    # Latest TIME_STEPS observations
    # --------------------------------------------------------

    latest_loads = np.asarray(
        live_loads[-TIME_STEPS:],
        dtype=float,
    ).reshape(
        -1,
        1,
    )


    # --------------------------------------------------------
    # Remove invalid values
    # --------------------------------------------------------

    latest_loads = np.nan_to_num(
        latest_loads,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


    # --------------------------------------------------------
    # Scale input
    # --------------------------------------------------------

    scaled_loads = input_scaler.transform(
        latest_loads
    )


    # --------------------------------------------------------
    # LSTM input
    # Shape:
    # (batch_size, time_steps, features)
    # --------------------------------------------------------

    X_live = scaled_loads.reshape(
        1,
        TIME_STEPS,
        1,
    )


    # --------------------------------------------------------
    # LSTM prediction
    # --------------------------------------------------------

    prediction = model.predict(
        X_live,
        verbose=0,
    )


    # --------------------------------------------------------
    # Convert prediction to numpy
    # --------------------------------------------------------

    prediction = np.asarray(
        prediction
    )


    # --------------------------------------------------------
    # Validate model output
    # --------------------------------------------------------

    if prediction.ndim != 2:
        raise ValueError(
            "Unexpected LSTM output shape: "
            f"{prediction.shape}"
        )

    if prediction.shape[1] < 2:
        raise ValueError(
            "LSTM model must return two outputs: "
            "total vehicles and traffic load."
        )


    # --------------------------------------------------------
    # Inverse scaling
    # --------------------------------------------------------

    prediction_original = (
        target_scaler.inverse_transform(
            prediction
        )
    )


    # --------------------------------------------------------
    # Predicted total vehicles
    # --------------------------------------------------------

    predicted_total_vehicles = max(
        0.0,
        float(
            prediction_original[0][0]
        ),
    )


    # --------------------------------------------------------
    # Predicted traffic load
    # --------------------------------------------------------

    predicted_traffic_load = max(
        0.0,
        float(
            prediction_original[0][1]
        ),
    )


    # --------------------------------------------------------
    # Traffic classification
    # --------------------------------------------------------

    traffic_class = classify_traffic(
        predicted_traffic_load,
        thresholds,
    )


    # --------------------------------------------------------
    # PSO signal optimization
    # --------------------------------------------------------

    signal = pso_optimize(
        predicted_traffic_load
    )


    # --------------------------------------------------------
    # RETURN RESULT
    # --------------------------------------------------------

    return {
        "predicted_total_vehicles":
            predicted_total_vehicles,

        "predicted_traffic_load":
            predicted_traffic_load,

        "traffic_class":
            traffic_class,

        "low_threshold":
            float(thresholds["low"]),

        "high_threshold":
            float(thresholds["high"]),

        "signal":
            signal,
    }

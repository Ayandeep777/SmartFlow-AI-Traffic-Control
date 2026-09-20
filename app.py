# ============================================================
# SmartFlow - Live Adaptive Traffic Control Dashboard
# ============================================================

import csv
import os
import time

from collections import deque
from datetime import datetime

import cv2
import pandas as pd
import streamlit as st

from ultralytics import YOLO

from config import (
    CAMERA_URL,
    YOLO_MODEL,
    YOLO_CONFIDENCE,
    TRACKER_CONFIG,
    LIVE_FILE,
    DETECTION_INTERVAL,
    TIME_STEPS,
)

from predictor import predict_from_live

from signal_controller import SignalController

from traffic_features import calculate_traffic_load

from vehicle_tracker import VehicleTracker


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="SMARTFLOW",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# DASHBOARD STYLE
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1rem;
        padding-bottom: 1.5rem;
        max-width: 1500px;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 14px;
        padding: 14px;
        background: rgba(128,128,128,0.035);
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 700;
    }

    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin-top: 15px;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "camera" not in st.session_state:
    st.session_state.camera = None

if "yolo_model" not in st.session_state:
    st.session_state.yolo_model = None

if "vehicle_tracker" not in st.session_state:
    st.session_state.vehicle_tracker = VehicleTracker()

if "controller" not in st.session_state:
    st.session_state.controller = SignalController()

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "cycle_number" not in st.session_state:
    st.session_state.cycle_number = 0

if "last_detection_time" not in st.session_state:
    st.session_state.last_detection_time = 0.0

if "current_counts" not in st.session_state:
    st.session_state.current_counts = {
        "car": 0,
        "motorcycle": 0,
        "bus": 0,
        "truck": 0,
    }

if "current_new_counts" not in st.session_state:
    st.session_state.current_new_counts = {
        "car": 0,
        "motorcycle": 0,
        "bus": 0,
        "truck": 0,
    }

if "current_active_counts" not in st.session_state:
    st.session_state.current_active_counts = {
        "car": 0,
        "motorcycle": 0,
        "bus": 0,
        "truck": 0,
    }

if "current_ids" not in st.session_state:
    st.session_state.current_ids = []

if "live_values" not in st.session_state:
    st.session_state.live_values = deque(
        maxlen=1000
    )

if "traffic_history" not in st.session_state:
    st.session_state.traffic_history = []

if "live_log" not in st.session_state:
    st.session_state.live_log = []


# ============================================================
# SAFE CONVERSION FUNCTIONS
# ============================================================

def safe_float(
    value,
    default=0.0,
):

    try:
        return float(value)

    except (
        TypeError,
        ValueError,
    ):
        return float(default)


def safe_int(
    value,
    default=0,
):

    try:
        return int(
            round(
                float(value)
            )
        )

    except (
        TypeError,
        ValueError,
    ):
        return int(default)


# ============================================================
# LOAD YOLO
# ============================================================

def load_yolo():

    if st.session_state.yolo_model is None:

        st.session_state.yolo_model = YOLO(
            YOLO_MODEL
        )

    return st.session_state.yolo_model


# ============================================================
# OPEN CAMERA
# ============================================================

def open_camera():

    if st.session_state.camera is None:

        cap = cv2.VideoCapture(
            CAMERA_URL
        )

        if not cap.isOpened():

            st.session_state.camera = None

            return None

        try:

            cap.set(
                cv2.CAP_PROP_BUFFERSIZE,
                1,
            )

        except Exception:
            pass

        st.session_state.camera = cap

    return st.session_state.camera


# ============================================================
# YOLO + BOT-SORT
# ============================================================

def detect_and_track(frame):

    model = load_yolo()

    results = model.track(
        frame,
        persist=True,
        tracker=TRACKER_CONFIG,
        conf=YOLO_CONFIDENCE,
        verbose=False,
    )

    tracker = (
        st.session_state.vehicle_tracker
    )

    empty_result = {
        "new_counts": {
            "car": 0,
            "motorcycle": 0,
            "bus": 0,
            "truck": 0,
        },

        "new_ids": [],

        "current_ids": [],

        "active_counts": {
            "car": 0,
            "motorcycle": 0,
            "bus": 0,
            "truck": 0,
        },

        "total_counts": {
            "car": 0,
            "motorcycle": 0,
            "bus": 0,
            "truck": 0,
        },
    }

    if not results:
        return empty_result

    latest_result = empty_result

    for result in results:

        if result.boxes is None:
            continue

        latest_result = (
            tracker.process_tracks(
                result.boxes
            )
        )

    return latest_result


# ============================================================
# INITIALIZE LIVE CSV
# ============================================================

def initialize_live_csv():

    if os.path.exists(LIVE_FILE):
        return

    with open(
        LIVE_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "cycle_number",
                "timestamp",
                "phase",
                "cars",
                "motorcycles",
                "buses",
                "trucks",
                "active_vehicles",
                "new_vehicles",
                "unique_vehicles",
                "traffic_load",
            ]
        )


# ============================================================
# APPEND LIVE CSV
# ============================================================

def append_live_csv(
    active_counts,
    traffic_load,
    phase,
    cycle_number,
    new_vehicles,
    unique_vehicles,
):

    initialize_live_csv()

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    active_total = (
        active_counts["car"]
        + active_counts["motorcycle"]
        + active_counts["bus"]
        + active_counts["truck"]
    )

    with open(
        LIVE_FILE,
        "a",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                cycle_number,
                timestamp,
                phase,
                active_counts["car"],
                active_counts["motorcycle"],
                active_counts["bus"],
                active_counts["truck"],
                active_total,
                new_vehicles,
                unique_vehicles,
                traffic_load,
            ]
        )


# ============================================================
# TRAFFIC LIGHT
# ============================================================

def render_traffic_light(phase):

    phase = str(
        phase
    ).upper()

    red_active = (
        phase == "RED"
    )

    yellow_active = (
        phase == "YELLOW"
    )

    green_active = (
        phase == "GREEN"
    )

    red_color = (
        "#ff2b2b"
        if red_active
        else "#3a3a3a"
    )

    yellow_color = (
        "#ffd21f"
        if yellow_active
        else "#3a3a3a"
    )

    green_color = (
        "#25d94f"
        if green_active
        else "#3a3a3a"
    )

    red_shadow = (
        "0 0 28px rgba(255,0,0,0.9)"
        if red_active
        else "none"
    )

    yellow_shadow = (
        "0 0 28px rgba(255,210,0,0.9)"
        if yellow_active
        else "none"
    )

    green_shadow = (
        "0 0 28px rgba(0,255,80,0.9)"
        if green_active
        else "none"
    )

    html = f"""
    <div style="
        width:100%;
        min-height:270px;
        display:flex;
        flex-direction:column;
        align-items:center;
        justify-content:center;
        font-family:Arial,sans-serif;
    ">

        <div style="
            width:105px;
            background:#171717;
            border-radius:24px;
            padding:15px 10px;
            border:3px solid #303030;
        ">

            <div style="
                width:62px;
                height:62px;
                border-radius:50%;
                margin:10px auto;
                background:{red_color};
                box-shadow:{red_shadow};
            "></div>

            <div style="
                width:62px;
                height:62px;
                border-radius:50%;
                margin:10px auto;
                background:{yellow_color};
                box-shadow:{yellow_shadow};
            "></div>

            <div style="
                width:62px;
                height:62px;
                border-radius:50%;
                margin:10px auto;
                background:{green_color};
                box-shadow:{green_shadow};
            "></div>

        </div>

        <div style="
            margin-top:14px;
            font-size:19px;
            font-weight:700;
        ">
            {phase}
        </div>

    </div>
    """

    st.html(html)


# ============================================================
# LIVE CAMERA DISPLAY
# ============================================================

def render_live_camera():

    html = f"""
    <div style="
        width:100%;
        display:flex;
        justify-content:center;
        align-items:center;
        overflow:hidden;
    ">

        <img
            src="{CAMERA_URL}"
            style="
                width:100%;
                max-height:520px;
                object-fit:contain;
                border-radius:14px;
                background:#111;
            "
        >

    </div>
    """

    st.html(html)


# ============================================================
# AI DECISION
# ============================================================

def make_ai_decision():

    if (
        len(
            st.session_state.live_values
        )
        < TIME_STEPS
    ):
        return False

    try:

        result = predict_from_live(
            list(
                st.session_state.live_values
            )
        )

        if not isinstance(
            result,
            dict,
        ):

            st.error(
                "predict_from_live() did not "
                "return a dictionary."
            )

            return False

        required_keys = [
            "predicted_total_vehicles",
            "predicted_traffic_load",
            "traffic_class",
            "low_threshold",
            "high_threshold",
            "signal",
        ]

        missing_keys = [
            key
            for key in required_keys
            if key not in result
        ]

        if missing_keys:

            st.error(
                "Predictor returned missing keys: "
                + ", ".join(missing_keys)
            )

            return False

        st.session_state.prediction_result = (
            result
        )

        return True

    except Exception as exc:

        st.error(
            f"LSTM / PSO error: {exc}"
        )

        return False


# ============================================================
# START FIRST CYCLE
# ============================================================

def start_first_cycle():

    if (
        len(
            st.session_state.live_values
        )
        < TIME_STEPS
    ):
        return False

    if (
        st.session_state.cycle_number
        != 0
    ):
        return False

    if not make_ai_decision():
        return False

    result = (
        st.session_state.prediction_result
    )

    signal = result["signal"]

    green = safe_int(
        signal["green"],
        10,
    )

    yellow = safe_int(
        signal["yellow"],
        5,
    )

    red = safe_int(
        signal["red"],
        10,
    )

    try:

        st.session_state.controller.start_cycle(
            green=green,
            yellow=yellow,
            red=red,
            cycle_number=1,
        )

        st.session_state.cycle_number = 1

        return True

    except Exception as exc:

        st.error(
            f"Could not start Cycle 1: {exc}"
        )

        return False


# ============================================================
# START NEXT CYCLE
# ============================================================

def start_next_cycle():

    if (
        len(
            st.session_state.live_values
        )
        < TIME_STEPS
    ):
        return False

    if not make_ai_decision():
        return False

    result = (
        st.session_state.prediction_result
    )

    signal = result["signal"]

    green = safe_int(
        signal["green"],
        10,
    )

    yellow = safe_int(
        signal["yellow"],
        5,
    )

    red = safe_int(
        signal["red"],
        10,
    )

    next_cycle = (
        st.session_state.cycle_number
        + 1
    )

    try:

        st.session_state.controller.start_cycle(
            green=green,
            yellow=yellow,
            red=red,
            cycle_number=next_cycle,
        )

        st.session_state.cycle_number = (
            next_cycle
        )

        return True

    except Exception as exc:

        st.error(
            f"Could not start Cycle "
            f"{next_cycle}: {exc}"
        )

        return False


# ============================================================
# RESET SESSION
# ============================================================

def reset_session():

    if st.session_state.camera is not None:

        try:
            st.session_state.camera.release()
        except Exception:
            pass

    st.session_state.camera = None

    st.session_state.yolo_model = None

    st.session_state.vehicle_tracker = (
        VehicleTracker()
    )

    st.session_state.controller = (
        SignalController()
    )

    st.session_state.prediction_result = None

    st.session_state.cycle_number = 0

    st.session_state.last_detection_time = 0.0

    st.session_state.current_counts = {
        "car": 0,
        "motorcycle": 0,
        "bus": 0,
        "truck": 0,
    }

    st.session_state.current_new_counts = {
        "car": 0,
        "motorcycle": 0,
        "bus": 0,
        "truck": 0,
    }

    st.session_state.current_active_counts = {
        "car": 0,
        "motorcycle": 0,
        "bus": 0,
        "truck": 0,
    }

    st.session_state.current_ids = []

    st.session_state.live_values = deque(
        maxlen=1000
    )

    st.session_state.traffic_history = []

    st.session_state.live_log = []


# ============================================================
# INITIALIZE
# ============================================================

initialize_live_csv()

controller = (
    st.session_state.controller
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("SMARTFLOW")

    st.write(
        "Adaptive Traffic Control System"
    )

    st.divider()

    st.write(
        f"YOLO: `{YOLO_MODEL}`"
    )

    st.write(
        f"Tracker: `{TRACKER_CONFIG}`"
    )

    st.write(
        f"Detection interval: "
        f"`{DETECTION_INTERVAL}s`"
    )

    st.write(
        f"LSTM window: `{TIME_STEPS}`"
    )

    st.divider()

    if st.button(
        "Reset Live Session",
        width="stretch",
    ):

        reset_session()

        st.rerun()


# ============================================================
# HEADER
# ============================================================

st.title("🚦 SMARTFLOW")

st.caption(
    "Live AI-Based Adaptive Traffic Control System"
)


# ============================================================
# CAMERA + SIGNAL
# ============================================================

camera_col, signal_col = st.columns(
    [2.25, 1],
    gap="large",
)


# ============================================================
# LIVE CAMERA
# ============================================================

with camera_col:

    st.markdown(
        '<div class="section-title">'
        'LIVE CAMERA'
        '</div>',
        unsafe_allow_html=True,
    )

    render_live_camera()


# ============================================================
# CURRENT SIGNAL
# ============================================================

with signal_col:

    st.markdown(
        '<div class="section-title">'
        'CURRENT SIGNAL'
        '</div>',
        unsafe_allow_html=True,
    )

    render_traffic_light(
        controller.phase
    )

    s1, s2 = st.columns(2)

    with s1:

        st.metric(
            "PHASE",
            controller.phase,
        )

    with s2:

        st.metric(
            "TIME REMAINING",
            f"{controller.remaining()} sec",
        )


# ============================================================
# YOLO DETECTION
# ============================================================

cap = open_camera()

if cap is None:

    st.warning(
        "YOLO camera connection could not "
        "be opened. Check CAMERA_URL in config.py."
    )

else:

    now = time.time()

    should_detect = (
        now
        - st.session_state.last_detection_time
        >= DETECTION_INTERVAL
    )

    if should_detect:

        ok, frame = cap.read()

        if (
            not ok
            or frame is None
        ):

            try:
                cap.release()
            except Exception:
                pass

            st.session_state.camera = None

        else:

            try:

                tracking_result = (
                    detect_and_track(frame)
                )

                new_counts = (
                    tracking_result[
                        "new_counts"
                    ]
                )

                active_counts = (
                    tracking_result[
                        "active_counts"
                    ]
                )

                total_counts = (
                    tracking_result[
                        "total_counts"
                    ]
                )

                current_ids = (
                    tracking_result[
                        "current_ids"
                    ]
                )


                # --------------------------------------------
                # SAVE COUNTS
                # --------------------------------------------

                st.session_state.current_counts = (
                    total_counts
                )

                st.session_state.current_new_counts = (
                    new_counts
                )

                st.session_state.current_active_counts = (
                    active_counts
                )

                st.session_state.current_ids = (
                    current_ids
                )


                # --------------------------------------------
                # ACTIVE TOTAL
                # --------------------------------------------

                active_total = (
                    active_counts["car"]
                    + active_counts["motorcycle"]
                    + active_counts["bus"]
                    + active_counts["truck"]
                )


                # --------------------------------------------
                # NEW TOTAL
                # --------------------------------------------

                new_total = (
                    new_counts["car"]
                    + new_counts["motorcycle"]
                    + new_counts["bus"]
                    + new_counts["truck"]
                )


                # --------------------------------------------
                # WEIGHTED TRAFFIC LOAD
                # --------------------------------------------

                traffic_load = (
                    calculate_traffic_load(
                        cars=active_counts["car"],
                        motorcycles=active_counts[
                            "motorcycle"
                        ],
                        buses=active_counts["bus"],
                        trucks=active_counts[
                            "truck"
                        ],
                    )
                )


                # --------------------------------------------
                # ADD TO LSTM WINDOW
                # --------------------------------------------

                st.session_state.live_values.append(
                    traffic_load
                )


                # --------------------------------------------
                # UNIQUE VEHICLES
                # --------------------------------------------

                unique_total = len(
                    st.session_state
                    .vehicle_tracker
                    .counted_ids
                )


                # --------------------------------------------
                # CURRENT PHASE
                # --------------------------------------------

                if controller.running:

                    observation_phase = (
                        controller.phase
                    )

                else:

                    observation_phase = "WARMUP"


                # --------------------------------------------
                # SAVE CSV
                # --------------------------------------------

                append_live_csv(
                    active_counts=active_counts,
                    traffic_load=traffic_load,
                    phase=observation_phase,
                    cycle_number=(
                        st.session_state.cycle_number
                    ),
                    new_vehicles=new_total,
                    unique_vehicles=unique_total,
                )


                # --------------------------------------------
                # GRAPH HISTORY
                # --------------------------------------------

                st.session_state.traffic_history.append(
                    {
                        "timestamp":
                            datetime.now(),

                        "traffic_load":
                            traffic_load,

                        "active_vehicles":
                            active_total,

                        "new_vehicles":
                            new_total,
                    }
                )

                if (
                    len(
                        st.session_state.traffic_history
                    )
                    > 1000
                ):

                    st.session_state.traffic_history = (
                        st.session_state
                        .traffic_history[-1000:]
                    )


                # --------------------------------------------
                # LIVE LOG
                # --------------------------------------------

                st.session_state.live_log.append(
                    {
                        "cycle":
                            st.session_state.cycle_number,

                        "timestamp":
                            datetime.now().strftime(
                                "%H:%M:%S"
                            ),

                        "phase":
                            observation_phase,

                        "active":
                            active_total,

                        "new":
                            new_total,

                        "unique":
                            unique_total,

                        "cars":
                            active_counts["car"],

                        "motorcycles":
                            active_counts[
                                "motorcycle"
                            ],

                        "buses":
                            active_counts["bus"],

                        "trucks":
                            active_counts[
                                "truck"
                            ],

                        "traffic_load":
                            round(
                                traffic_load,
                                2,
                            ),
                    }
                )

                if (
                    len(
                        st.session_state.live_log
                    )
                    > 300
                ):

                    st.session_state.live_log = (
                        st.session_state
                        .live_log[-300:]
                    )

            except Exception as exc:

                st.error(
                    f"YOLO / BoT-SORT error: {exc}"
                )

        st.session_state.last_detection_time = now


# ============================================================
# LSTM WARM-UP
# ============================================================

live_count = len(
    st.session_state.live_values
)


# ============================================================
# FIRST CYCLE
# ============================================================

if (
    live_count >= TIME_STEPS
    and st.session_state.cycle_number == 0
):

    start_first_cycle()

    controller = (
        st.session_state.controller
    )


# ============================================================
# UPDATE SIGNAL
# ============================================================

controller.update()


# ============================================================
# NEXT CYCLE
# ============================================================

if (
    controller.phase == "COMPLETED"
    and live_count >= TIME_STEPS
):

    start_next_cycle()

    controller = (
        st.session_state.controller
    )


# ============================================================
# CURRENT ACTIVE COUNTS
# ============================================================

active_counts = (
    st.session_state.current_active_counts
)

new_counts = (
    st.session_state.current_new_counts
)


# ============================================================
# ACTIVE TOTAL
# ============================================================

active_total = (
    active_counts["car"]
    + active_counts["motorcycle"]
    + active_counts["bus"]
    + active_counts["truck"]
)


# ============================================================
# NEW TOTAL
# ============================================================

new_total = (
    new_counts["car"]
    + new_counts["motorcycle"]
    + new_counts["bus"]
    + new_counts["truck"]
)


# ============================================================
# UNIQUE TOTAL
# ============================================================

unique_total = len(
    st.session_state
    .vehicle_tracker
    .counted_ids
)


# ============================================================
# CURRENT TRAFFIC LOAD
# ============================================================

current_load = (
    calculate_traffic_load(
        cars=active_counts["car"],
        motorcycles=active_counts[
            "motorcycle"
        ],
        buses=active_counts["bus"],
        trucks=active_counts[
            "truck"
        ],
    )
)


# ============================================================
# PREDICTION DATA
# ============================================================

prediction_result = (
    st.session_state.prediction_result
)

if prediction_result:

    predicted_total_vehicles = safe_float(
        prediction_result[
            "predicted_total_vehicles"
        ]
    )

    predicted_load = safe_float(
        prediction_result[
            "predicted_traffic_load"
        ]
    )

    traffic_class = str(
        prediction_result[
            "traffic_class"
        ]
    ).upper()

    low_threshold = safe_float(
        prediction_result[
            "low_threshold"
        ]
    )

    high_threshold = safe_float(
        prediction_result[
            "high_threshold"
        ]
    )

    signal = prediction_result[
        "signal"
    ]

else:

    predicted_total_vehicles = 0.0

    predicted_load = 0.0

    traffic_class = "WAITING"

    low_threshold = 0.0

    high_threshold = 0.0

    signal = None


# ============================================================
# AI WARM-UP STATUS
# ============================================================

st.divider()

if live_count < TIME_STEPS:

    remaining = (
        TIME_STEPS - live_count
    )

    st.info(
        f"AI warm-up: {live_count}/"
        f"{TIME_STEPS} observations collected. "
        f"{remaining} more required before "
        f"the first LSTM prediction."
    )

else:

    st.success(
        f"LSTM window ready: "
        f"{live_count} observations available."
    )


# ============================================================
# TRAFFIC STATUS
# ============================================================

st.markdown(
    '<div class="section-title">'
    'TRAFFIC STATUS'
    '</div>',
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)

with m1:

    st.metric(
        "ACTIVE VEHICLES",
        active_total,
    )

with m2:

    st.metric(
        "NEW VEHICLES",
        new_total,
    )

with m3:

    st.metric(
        "WEIGHTED LOAD",
        f"{current_load:.2f}",
    )

with m4:

    st.metric(
        "TRAFFIC CLASS",
        traffic_class,
    )


# ============================================================
# TRACKING SUMMARY
# ============================================================

t1, t2, t3 = st.columns(3)

with t1:

    st.metric(
        "UNIQUE VEHICLES",
        unique_total,
    )

with t2:

    st.metric(
        "ACTIVE TRACKS",
        len(
            st.session_state.current_ids
        ),
    )

with t3:

    st.metric(
        "LSTM WINDOW",
        f"{live_count}/{TIME_STEPS}",
    )


# ============================================================
# LSTM PREDICTION
# ============================================================

st.markdown(
    '<div class="section-title">'
    'LSTM PREDICTION'
    '</div>',
    unsafe_allow_html=True,
)

l1, l2 = st.columns(2)

with l1:

    st.metric(
        "PREDICTED VEHICLES",
        f"{predicted_total_vehicles:.2f}",
    )

with l2:

    st.metric(
        "PREDICTED TRAFFIC LOAD",
        f"{predicted_load:.2f}",
    )


# ============================================================
# THRESHOLDS
# ============================================================

if prediction_result:

    th1, th2 = st.columns(2)

    with th1:

        st.metric(
            "LOW THRESHOLD",
            f"{low_threshold:.2f}",
        )

    with th2:

        st.metric(
            "HIGH THRESHOLD",
            f"{high_threshold:.2f}",
        )


# ============================================================
# VEHICLE BREAKDOWN
# ============================================================

st.markdown(
    '<div class="section-title">'
    'LIVE VEHICLE BREAKDOWN'
    '</div>',
    unsafe_allow_html=True,
)

v1, v2, v3, v4 = st.columns(4)

with v1:

    st.metric(
        "CARS",
        active_counts["car"],
    )

with v2:

    st.metric(
        "MOTORCYCLES",
        active_counts[
            "motorcycle"
        ],
    )

with v3:

    st.metric(
        "BUSES",
        active_counts["bus"],
    )

with v4:

    st.metric(
        "TRUCKS",
        active_counts["truck"],
    )


# ============================================================
# PSO SIGNAL
# ============================================================

st.markdown(
    '<div class="section-title">'
    'PSO OPTIMIZED SIGNAL'
    '</div>',
    unsafe_allow_html=True,
)

if signal:

    green = safe_int(
        signal["green"],
        10,
    )

    yellow = safe_int(
        signal["yellow"],
        5,
    )

    red = safe_int(
        signal["red"],
        10,
    )

    cycle = safe_int(
        signal["cycle"],
        green + yellow + red,
    )

    cost = safe_float(
        signal.get(
            "cost",
            0,
        )
    )

    p1, p2, p3, p4, p5 = st.columns(5)

    with p1:

        st.metric(
            "GREEN",
            f"{green} sec",
        )

    with p2:

        st.metric(
            "YELLOW",
            f"{yellow} sec",
        )

    with p3:

        st.metric(
            "RED",
            f"{red} sec",
        )

    with p4:

        st.metric(
            "CYCLE",
            f"{cycle} sec",
        )

    with p5:

        st.metric(
            "PSO COST",
            f"{cost:.2f}",
        )

else:

    st.info(
        "PSO signal will be generated "
        "after the first 30 observations."
    )


# ============================================================
# CURRENT CYCLE
# ============================================================

st.markdown(
    '<div class="section-title">'
    'CURRENT CYCLE'
    '</div>',
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)

with c1:

    st.metric(
        "CYCLE NUMBER",
        st.session_state.cycle_number,
    )

with c2:

    st.metric(
        "CURRENT PHASE",
        controller.phase,
    )

with c3:

    st.metric(
        "TIME REMAINING",
        f"{controller.remaining()} sec",
    )


# ============================================================
# LIVE TRAFFIC GRAPH
# ============================================================

st.markdown(
    '<div class="section-title">'
    'LIVE TRAFFIC'
    '</div>',
    unsafe_allow_html=True,
)

if st.session_state.traffic_history:

    graph_df = pd.DataFrame(
        st.session_state.traffic_history
    )

    graph_df = graph_df.set_index(
        "timestamp"
    )

    st.line_chart(
        graph_df[
            [
                "traffic_load",
                "active_vehicles",
                "new_vehicles",
            ]
        ],
        width="stretch",
    )

else:

    st.info(
        "Waiting for traffic observations..."
    )


# ============================================================
# LIVE LOG
# ============================================================

st.markdown(
    '<div class="section-title">'
    'LIVE TRAFFIC LOG'
    '</div>',
    unsafe_allow_html=True,
)

if st.session_state.live_log:

    log_df = pd.DataFrame(
        st.session_state.live_log
    )

    st.dataframe(
        log_df,
        width="stretch",
        hide_index=True,
    )

else:

    st.info(
        "No observations recorded yet."
    )


# ============================================================
# SYSTEM STATUS
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">'
    'SYSTEM STATUS'
    '</div>',
    unsafe_allow_html=True,
)

s1, s2, s3, s4 = st.columns(4)

with s1:

    st.write(
        "**YOLO:** "
        + (
            "READY"
            if st.session_state.yolo_model
            else "WAITING"
        )
    )

with s2:

    st.write(
        "**TRACKING:** "
        + (
            "ACTIVE"
            if len(
                st.session_state.current_ids
            ) > 0
            else "WAITING"
        )
    )

with s3:

    st.write(
        f"**LSTM:** "
        f"{live_count}/{TIME_STEPS}"
    )

with s4:

    st.write(
        f"**SIGNAL:** "
        f"{controller.phase}"
    )


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "SmartFlow | YOLO11n + BoT-SORT + "
    "LSTM + Traffic Classification + "
    "PSO Adaptive Signal Control"
)


# ============================================================
# AUTO REFRESH
# ============================================================

time.sleep(
    max(
        0.1,
        DETECTION_INTERVAL,
    )
)

st.rerun()

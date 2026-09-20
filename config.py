# ============================================================
# SmartFlow - Configuration
# ============================================================

# ------------------------------------------------------------
# CAMERA
# ------------------------------------------------------------

CAMERA_IP = "10.10.114.109"
CAMERA_URL = f"http://{CAMERA_IP}:8080/video"


# ------------------------------------------------------------
# YOLO / TRACKING
# ------------------------------------------------------------

YOLO_MODEL = "yolo11n.pt"
YOLO_CONFIDENCE = 0.35

# Ultralytics BoT-SORT tracker
TRACKER_CONFIG = "botsort.yaml"


# ------------------------------------------------------------
# DETECTION / OBSERVATION TIMING
# ------------------------------------------------------------

# Record one traffic observation approximately every second
DETECTION_INTERVAL = 1.0


# ------------------------------------------------------------
# FILES
# ------------------------------------------------------------

BASE_FILE = "SmartFlow_Traffic_TimeSeries.csv"
LIVE_FILE = "SmartFlow_Live_Traffic.csv"

LSTM_MODEL = "traffic_lstm.keras"
SCALER_FILE = "traffic_scaler.pkl"


# ------------------------------------------------------------
# LSTM
# ------------------------------------------------------------

# Number of previous traffic observations used by LSTM
TIME_STEPS = 30


# ------------------------------------------------------------
# TRAFFIC WEIGHTS
# ------------------------------------------------------------

MOTORCYCLE_WEIGHT = 0.5
CAR_WEIGHT = 1.0
BUS_WEIGHT = 3.0
TRUCK_WEIGHT = 3.5


# ------------------------------------------------------------
# SIGNAL TIMING
# ------------------------------------------------------------

YELLOW_TIME = 5
RED_TIME = 10

# PSO searches for green time between these values
MIN_GREEN_TIME = 10
MAX_GREEN_TIME = 60


# ------------------------------------------------------------
# PARTICLE SWARM OPTIMIZATION
# ------------------------------------------------------------

PSO_PARTICLES = 20
PSO_ITERATIONS = 30

PSO_INERTIA = 0.7
PSO_COGNITIVE = 1.5
PSO_SOCIAL = 1.5


# ------------------------------------------------------------
# PSO COST FUNCTION
# ------------------------------------------------------------

# Cost =
#
# Waiting Cost
# + Queue Cost
# + Under-Service Penalty
# + Excessive-Green Penalty

WAITING_COST_WEIGHT = 1.0
QUEUE_COST_WEIGHT = 1.0
UNDER_SERVICE_PENALTY = 2.0
EXCESSIVE_GREEN_PENALTY = 1.0


# ------------------------------------------------------------
# TRAFFIC SERVICE RATE
# ------------------------------------------------------------

# Approximate amount of weighted traffic load that can
# be served per second during green.
#
# This is a model parameter and can later be calibrated
# using real traffic discharge observations.

SERVICE_RATE = 0.5

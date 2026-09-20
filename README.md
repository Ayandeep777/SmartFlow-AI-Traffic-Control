# SMARTFLOW — AI-Based Adaptive Traffic Signal Control System

SMARTFLOW is an AI-based adaptive traffic signal control system that combines **computer vision, time-series forecasting, traffic-load modeling, and Particle Swarm Optimization (PSO)** to dynamically determine traffic signal timings.

The system uses a smartphone as an IP camera, **YOLO11n with BoT-SORT tracking** for real-time vehicle detection, a weighted traffic-load model to represent heterogeneous vehicles, a **multi-output LSTM** to forecast future traffic conditions, and **PSO** to optimize the green signal duration.

---

## Project Objective

The objective of SMARTFLOW is to develop an adaptive traffic signal system that can respond to changing traffic conditions instead of relying on fixed signal timings.

The system continuously:

1. Captures live traffic video.
2. Detects and tracks vehicles.
3. Calculates the current traffic load.
4. Uses the latest traffic observations for LSTM forecasting.
5. Classifies the predicted traffic condition.
6. Uses PSO to determine an optimal green duration.
7. Applies the optimized timing to the next signal cycle.
8. Continues collecting new observations and adapts the following cycle.

---

# System Architecture

```text
                 SMARTPHONE IP CAMERA
                         │
                         │ Wi-Fi / HTTP
                         ▼
                  ┌───────────────┐
                  │    OpenCV     │
                  │ Video Capture │
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │    YOLO11n    │
                  │ Vehicle Detect│
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │  BoT-SORT     │
                  │   Tracking    │
                  └───────┬───────┘
                          │
                          ▼
                  Vehicle Counts
                          │
                          ▼
                  Traffic Load
                          │
                          ▼
                  ┌───────────────┐
                  │     LSTM      │
                  │   Forecasting │
                  └───────┬───────┘
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
    Future Vehicle Count       Future Traffic Load
                                      │
                                      ▼
                             Traffic Classification
                                      │
                                      ▼
                               ┌─────────────┐
                               │     PSO     │
                               │ Optimization│
                               └──────┬──────┘
                                      │
                                      ▼
                             Optimal Green Time
                                      │
                                      ▼
                            Signal Controller
                                      │
                                      ▼
                              Next Signal Cycle
```

---

# Technologies Used

| Technology                  | Purpose                                  |
| --------------------------- | ---------------------------------------- |
| Python                      | Main programming language                |
| OpenCV                      | Camera/video acquisition                 |
| YOLO11n                     | Vehicle detection                        |
| BoT-SORT                    | Vehicle tracking                         |
| Pandas                      | Traffic data processing                  |
| NumPy                       | Numerical computation                    |
| TensorFlow / Keras          | LSTM forecasting                         |
| Scikit-learn                | Data scaling                             |
| Particle Swarm Optimization | Signal timing optimization               |
| Streamlit                   | Real-time dashboard                      |
| CSV                         | Historical and live traffic data storage |

---

# Vehicle Detection

SMARTFLOW uses **YOLO11n** to detect the following vehicle classes:

* Cars
* Motorcycles
* Buses
* Trucks

BoT-SORT tracking is used to maintain vehicle identities across frames and reduce repeated counting of the same vehicle.

The detection pipeline is:

```text
Camera Frame
     ↓
YOLO11n Detection
     ↓
BoT-SORT Tracking
     ↓
Vehicle Classes
     ↓
Vehicle Counts
```

The system creates approximately one traffic observation per second using:

```python
DETECTION_INTERVAL = 1.0
```

---

# Traffic Load Calculation

Different vehicle types have different approximate effects on traffic flow.

SMARTFLOW therefore uses a weighted traffic-load model.

```text
Traffic Load =
    Cars × 1.0
  + Motorcycles × 0.5
  + Buses × 3.0
  + Trucks × 3.5
```

The current weights are:

| Vehicle    | Weight |
| ---------- | -----: |
| Motorcycle |    0.5 |
| Car        |    1.0 |
| Bus        |    3.0 |
| Truck      |    3.5 |

For example:

```text
Cars = 5
Motorcycles = 2
Buses = 1
Trucks = 0

Traffic Load
= (5 × 1.0)
+ (2 × 0.5)
+ (1 × 3.0)
+ (0 × 3.5)

= 9.0
```

This weighted value is used as the primary traffic-load signal for forecasting and optimization.

---

# LSTM Traffic Forecasting

SMARTFLOW uses a **multi-output Long Short-Term Memory (LSTM)** model.

The model is trained using historical traffic observations.

The LSTM uses:

```text
Previous 30 Traffic-Load Observations
                  ↓
                 LSTM
                  ↓
        ┌─────────┴─────────┐
        ▼                   ▼
Future Total          Future Traffic
Vehicles              Load
```

The configured sequence length is:

```python
TIME_STEPS = 30
```

Therefore, the model requires 30 observations before the first live prediction can be generated.

---

# Rolling Live Prediction

After the initial 30 observations have been collected, SMARTFLOW continuously uses the latest 30 live observations.

For example:

```text
Observations 1–30
       ↓
    Prediction 1

Observations 40–69
       ↓
    Prediction 2

Observations 79–108
       ↓
    Prediction 3
```

The exact window advances according to the observations collected during each signal cycle.

The LSTM is trained initially and then used for inference during live operation.

---

# Traffic Classification

The predicted traffic load is classified into:

* LOW
* MEDIUM
* HIGH

The classification thresholds are calculated from the historical traffic-load distribution rather than being arbitrarily selected.

Conceptually:

```text
Historical Traffic Loads
          ↓
   Calculate Thresholds
          ↓
 ┌────────┼─────────┐
 ▼        ▼         ▼
 LOW    MEDIUM      HIGH
```

This allows the classification to reflect the characteristics of the historical dataset.

---

# PSO-Based Signal Optimization

Particle Swarm Optimization is used to determine the green signal duration.

### Input

The numerical traffic load predicted by the LSTM.

### Decision Variable

Green signal duration:

```text
10 seconds ≤ G ≤ 60 seconds
```

### Objective

Minimize overall traffic cost by balancing:

* Waiting cost
* Queue / remaining traffic cost
* Under-service penalty
* Excessive green-time penalty

The optimization uses:

```text
Particles = 20
Iterations = 30
```

Each particle represents a candidate green duration.

The particles update their positions using standard PSO velocity and position updates while searching toward:

* Their personal best solution.
* The global best solution.

The final solution is:

```text
G*
```

where `G*` is the optimized green duration.

---

# Signal Timing

The signal controller uses:

```text
Green  →  Yellow  →  Red
```

The timings are:

```text
Green  = PSO optimized
Yellow = 5 seconds
Red    = 10 seconds
```

Therefore:

```text
Total Cycle
=
Optimized Green
+ 5
+ 10
```

For example, if PSO determines:

```text
Green = 24 seconds
```

then:

```text
Green  = 24 s
Yellow = 5 s
Red    = 10 s

Total Cycle = 39 seconds
```

---

# Adaptive Cycle Operation

SMARTFLOW does not retrain the LSTM after every signal cycle.

Instead:

```text
Initial 30 observations
          ↓
       LSTM
          ↓
     Prediction
          ↓
        PSO
          ↓
   Signal Cycle 1
          ↓
New observations collected
          ↓
Latest 30 observations
          ↓
       LSTM
          ↓
     Prediction
          ↓
        PSO
          ↓
   Signal Cycle 2
          ↓
        ...
```

This allows the system to continuously adapt to changing traffic conditions while keeping the trained model fixed during operation.

---

# Project Structure

```text
SmartFlow-AI-Traffic-Control/
│
├── README.md
├── CAMERA_SETUP.md
├── requirements.txt
├── .gitignore
│
├── app.py
├── config.py
├── vehicle_tracker.py
├── traffic_features.py
├── traffic_classifier.py
├── lstm_train.py
├── predictor.py
├── optimizer.py
├── signal_controller.py
│
├── test_predictor.py
├── test_live_csv.py
│
├── SmartFlow_Traffic_TimeSeries.csv
│
├── traffic_lstm.keras
├── traffic_scaler.pkl
│
└── yolo11n.pt
```

---

# Data Files

## Historical Traffic Data

```text
SmartFlow_Traffic_TimeSeries.csv
```

This dataset is used for LSTM training and historical traffic-load analysis.

The data contains vehicle counts such as:

```text
cars
motorcycles
buses
trucks
```

---

## Live Traffic Data

```text
SmartFlow_Live_Traffic.csv
```

This file is generated during system operation and stores live observations.

Typical fields include:

```text
cycle_number
timestamp
phase
cars
motorcycles
buses
trucks
active_vehicles
new_vehicles
unique_vehicles
traffic_load
```

Live runtime data should generally not be committed to the repository.

---

# Camera Setup

SMARTFLOW uses a smartphone as an IP camera.

The smartphone and computer must be connected to the same local network.

The camera pipeline is:

```text
Smartphone
    ↓
IP Camera Application
    ↓
HTTP Video Stream
    ↓
OpenCV
    ↓
YOLO11n
```

For detailed physical assembly and network configuration, see:

**[CAMERA_SETUP.md](CAMERA_SETUP.md)**

---

# Configuration

Camera and system parameters are defined in:

```text
config.py
```

Important parameters include:

```python
CAMERA_IP = "YOUR_PHONE_IP"
CAMERA_URL = f"http://{CAMERA_IP}:8080/video"

YOLO_MODEL = "yolo11n.pt"

DETECTION_INTERVAL = 1.0

TIME_STEPS = 30

YELLOW_TIME = 5
RED_TIME = 10

MIN_GREEN_TIME = 10
MAX_GREEN_TIME = 60

PSO_PARTICLES = 20
PSO_ITERATIONS = 30
```

The actual smartphone IP address depends on the local network.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/Ayandeep777/SmartFlow-AI-Traffic-Control.git
```

Move into the project directory:

```bash
cd SmartFlow-AI-Traffic-Control
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# LSTM Training

The LSTM can be trained using:

```bash
python lstm_train.py
```

The training process generates:

```text
traffic_lstm.keras
traffic_scaler.pkl
```

The model produces two outputs:

```text
1. Future total vehicles
2. Future traffic load
```

---

# Running SMARTFLOW

After configuring the camera and ensuring the required model files are available:

```bash
streamlit run app.py
```

The Streamlit dashboard provides:

* Live camera view
* Vehicle detection
* Vehicle counts
* Traffic load
* Predicted vehicle count
* Predicted traffic load
* Traffic classification
* Signal state
* PSO green-time optimization
* Cycle information
* Live traffic history
* Runtime logs

---

# Complete Operational Flow

```text
LIVE CAMERA
     ↓
YOLO11n
     ↓
BoT-SORT
     ↓
VEHICLE DETECTION
     ↓
VEHICLE COUNTS
     ↓
TRAFFIC LOAD
     ↓
30-STEP LIVE WINDOW
     ↓
MULTI-OUTPUT LSTM
     ↓
┌────────────────────────┐
│ Future Vehicles        │
│ Future Traffic Load    │
└────────────┬───────────┘
             ↓
    TRAFFIC CLASSIFICATION
             ↓
       PSO OPTIMIZATION
             ↓
     OPTIMAL GREEN TIME
             ↓
     SIGNAL CONTROLLER
             ↓
       NEXT CYCLE
             ↓
    NEW OBSERVATIONS
             ↓
       REPEAT PROCESS
```

---

# Important Parameters

| Parameter          | Current Value |
| ------------------ | ------------: |
| LSTM time steps    |            30 |
| Detection interval |      1 second |
| Minimum green      |    10 seconds |
| Maximum green      |    60 seconds |
| Yellow             |     5 seconds |
| Red                |    10 seconds |
| PSO particles      |            20 |
| PSO iterations     |            30 |
| Motorcycle weight  |           0.5 |
| Car weight         |           1.0 |
| Bus weight         |           3.0 |
| Truck weight       |           3.5 |

---

# Limitations

The current prototype has several practical limitations:

* Vehicle detection depends on camera quality and viewing angle.
* Lighting and weather can affect detection performance.
* Smartphone network connectivity can affect video streaming.
* Traffic-load weights are model parameters and can be calibrated using real traffic data.
* The PSO service-rate parameter is an approximation and can be calibrated using observed traffic discharge rates.
* The current system controls a simulated/software signal controller rather than a physical traffic signal.
* LSTM forecasting performance depends on the quality and quantity of historical training data.

---

# Future Scope

Possible extensions include:

* Multi-intersection coordination.
* Reinforcement learning-based signal control.
* Weather-aware traffic prediction.
* Longer-term historical data collection.
* Automatic model retraining.
* Edge deployment.
* Dedicated CCTV/IP cameras.
* Real traffic signal hardware integration.
* Emergency vehicle prioritization.
* Pedestrian-aware signal control.
* Real-world calibration of traffic-load weights.
* Real-world calibration of traffic discharge/service rates.

---

# Team

**Group 16**

| Member                    | Roll Number |
| ------------------------- | ----------- |
| Ayandeep Maji             | MBA24057    |
| Deepak Verma              | MBA25075    |
| Sakshi Kumari             | MBA25233    |
| Suryanshu Shekhar Singh   | MBA25274    |
| Himani                    | MBA25110    |
| CHANDRASHEKHAR ATUL UBALE | MBA25068    |
| Mahale Aashish Shashikant | MBA25145    |

---

# Academic Information

**Indian Institute of Management Jammu (IIM Jammu)**

**Subject:** Advanced Business Analytics

**Guided by:** Dr. Sundar

---

# Project Pipeline

```text
Detection
    ↓
Traffic Measurement
    ↓
Prediction
    ↓
Classification
    ↓
Optimization
    ↓
Signal Control
    ↓
Adaptive Cycle
```

SMARTFLOW integrates these components into a single adaptive traffic-management pipeline.

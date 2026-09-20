# SmartFlow — Camera Assembly & Setup

This document explains how to assemble and configure the camera component of the **SMARTFLOW AI-Based Adaptive Traffic Signal Control System**.

The camera provides the live traffic stream that is processed by YOLO11n for real-time vehicle detection.

---

## 1. Camera System Overview

The SmartFlow camera pipeline is:

```text
                SMARTPHONE CAMERA
                       │
                       │ Wi-Fi
                       ▼
               IP CAMERA STREAM
                       │
                       │ HTTP
                       ▼
                  OpenCV
                       │
                       ▼
                   YOLO11n
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
        Cars       Motorcycles     Buses/Trucks
          │            │            │
          └────────────┼────────────┘
                       ▼
                Traffic Counts
                       │
                       ▼
                  Traffic Load
```

The smartphone acts as an **IP camera**, while the computer running SmartFlow receives the camera stream over the local network.

---

# 2. Hardware Required

### Required Components

| Component                | Purpose                                   |
| ------------------------ | ----------------------------------------- |
| Android smartphone       | Live traffic camera                       |
| Smartphone holder/tripod | Keeps the camera stable                   |
| Computer/laptop          | Runs YOLO11n, LSTM, PSO and Streamlit     |
| Wi-Fi router/hotspot     | Connects phone and computer               |
| Power bank/charger       | Keeps smartphone powered during operation |

### Recommended Setup

The smartphone should be mounted at an elevated and stable position with a clear view of the traffic lane/intersection.

```text
              TRAFFIC ROAD
────────────────────────────────────

        🚗     🏍️      🚌      🚚
        
              ↑
              │
        CAMERA VIEW
              │
              │
        ┌─────────────┐
        │ Smartphone  │
        │   Camera    │
        └─────────────┘
              │
           Tripod/
            Mount
```

---

# 3. Physical Camera Assembly

## Step 1 — Mount the Smartphone

Place the smartphone on a stable tripod or phone holder.

The camera should:

* Remain stationary.
* Have a clear view of incoming traffic.
* Avoid excessive vibration.
* Avoid direct obstruction by trees, poles, vehicles, etc.
* Capture the road from an elevated position where possible.

### Important

Do not continuously move or rotate the phone after starting the system.

YOLO detection works more consistently when the camera viewpoint remains stable.

---

# 4. Recommended Camera Position

The camera should be positioned so that vehicles are clearly visible.

### Preferred View

```text
                 CAMERA
                    │
                    │
                    ▼
        ┌─────────────────────┐
        │                     │
        │     ROAD AREA       │
        │                     │
        │   🚗  🏍️  🚗       │
        │      🚌             │
        │          🚚         │
        │                     │
        └─────────────────────┘
```

Try to ensure that:

* Vehicles are not extremely small.
* The road occupies most of the useful camera frame.
* Vehicles are visible from the front/side rather than completely hidden.
* Lighting is adequate.
* The camera is not pointed directly toward strong sunlight.

---

# 5. Connect the Smartphone and Computer

The smartphone and computer must be connected to the **same local network**.

For example:

```text
              Wi-Fi Network
                    │
          ┌─────────┴─────────┐
          │                   │
          ▼                   ▼
     Smartphone            Computer
     IP Camera           SmartFlow
```

The computer must be able to access the IP address assigned to the smartphone.

---

# 6. Start the Smartphone IP Camera

Use an Android IP-camera application capable of providing an HTTP video stream.

Start the camera server on the smartphone.

The application will display an IP address similar to:

```text
http://10.10.210.139:8080
```

Your SmartFlow system uses the video endpoint:

```text
http://10.10.210.139:8080/video
```

The exact IP address can change depending on the network.

---

# 7. Configure `config.py`

Open:

```text
config.py
```

Set the smartphone IP address:

```python
CAMERA_IP = "10.10.210.139"
CAMERA_URL = f"http://{CAMERA_IP}:8080/video"
```

If the phone receives a different IP address, update only:

```python
CAMERA_IP
```

For example:

```python
CAMERA_IP = "192.168.1.25"
```

The video URL will automatically become:

```text
http://192.168.1.25:8080/video
```

---

# 8. Test the Camera Before Running SmartFlow

Before starting the complete application, test the camera stream.

Open the camera stream URL in a browser on the computer:

```text
http://YOUR_PHONE_IP:8080/video
```

For example:

```text
http://10.10.210.139:8080/video
```

If the live video appears, the phone-camera connection is working.

---

# 9. SmartFlow Camera Connection

The SmartFlow application uses OpenCV to connect to the camera:

```python
cap = cv2.VideoCapture(CAMERA_URL)
```

The camera stream is then read frame by frame:

```text
IP Camera
    ↓
OpenCV VideoCapture
    ↓
Frame
    ↓
YOLO11n
    ↓
Vehicle Detection
```

The application does not need to save the video continuously.

The camera provides the live stream while SmartFlow processes selected observations.

---

# 10. YOLO Detection Frequency

The current configuration uses:

```python
DETECTION_INTERVAL = 1.0
```

This means SmartFlow attempts to create approximately **one new traffic observation every second**.

For each observation:

```text
Camera Frame
     ↓
YOLO11n
     ↓
Cars
Motorcycles
Buses
Trucks
     ↓
Total Vehicles
     ↓
Traffic Load
     ↓
Live CSV
     ↓
LSTM rolling window
```

For example:

```text
10:10:01 → Cars 5 | Bikes 8 | Bus 1 | Truck 0
10:10:02 → Cars 6 | Bikes 7 | Bus 1 | Truck 0
10:10:03 → Cars 6 | Bikes 9 | Bus 0 | Truck 1
...
```

---

# 11. Camera Does NOT Control the Traffic Signal Directly

The camera is only the **data acquisition component**.

The complete system is:

```text
             CAMERA
                ↓
             YOLO11n
                ↓
        Vehicle Detection
                ↓
          Traffic Load
                ↓
       Latest 30 Observations
                ↓
              LSTM
                ↓
      Predicted Traffic Load
                ↓
               PSO
                ↓
       Optimal Green Time
                ↓
       Signal Controller
```

Therefore:

**Camera → Detection → Prediction → Optimization → Signal Control**

---

# 12. Recommended Physical Setup

For a demonstration/project setup, use:

```text
                    ROAD
═══════════════════════════════════════

       🚗      🏍️       🚗
             🚌
                   🚚

                    ↑
                    │
              Camera View
                    │
              ┌───────────┐
              │ Smartphone│
              │   Camera  │
              └───────────┘
                    │
                 Tripod
                    │
                    ▼

             ┌─────────────┐
             │   TABLE /   │
             │   STAND     │
             └─────────────┘

                    Wi-Fi
                     │
                     ▼

             ┌─────────────┐
             │   LAPTOP    │
             │  SmartFlow  │
             └─────────────┘
```

---

# 13. Before Starting the System

Check the following:

### Smartphone

* [ ] Camera is mounted securely.
* [ ] Camera has a clear view of the road.
* [ ] Smartphone is sufficiently charged.
* [ ] IP-camera application is running.
* [ ] Camera server is started.

### Network

* [ ] Smartphone and laptop are connected to the same network.
* [ ] Smartphone IP address is known.
* [ ] Camera stream opens from the laptop.

### SmartFlow

* [ ] `CAMERA_IP` is correct in `config.py`.
* [ ] `CAMERA_URL` points to `/video`.
* [ ] `yolo11n.pt` is available.
* [ ] `traffic_lstm.keras` is available.
* [ ] `traffic_scaler.pkl` is available.
* [ ] Historical CSV is available.
* [ ] Live CSV is available.

---

# 14. Starting the Complete System

From the SmartFlow project directory:

```bash
streamlit run app.py
```

The dashboard should open in the browser.

The system will then:

```text
1. Connect to smartphone camera
          ↓
2. Receive live video
          ↓
3. Capture/process traffic observations
          ↓
4. Run YOLO11n
          ↓
5. Count vehicles
          ↓
6. Calculate traffic load
          ↓
7. Collect first 30 observations
          ↓
8. Run LSTM prediction
          ↓
9. Classify traffic
          ↓
10. Run PSO
          ↓
11. Generate signal timing
          ↓
12. Start Cycle 1
          ↓
13. Continue collecting new observations
          ↓
14. Use latest 30 observations
          ↓
15. Generate Cycle 2
          ↓
16. Continue adaptively
```

---

# 15. Troubleshooting

## Camera does not appear

Check:

```python
CAMERA_IP = "YOUR_PHONE_IP"
```

Then verify:

```text
http://YOUR_PHONE_IP:8080/video
```

opens from the laptop.

---

## YOLO says camera cannot be opened

Possible causes:

* Phone and laptop are on different networks.
* Phone IP address changed.
* IP-camera server is not running.
* Incorrect port.
* Incorrect video endpoint.
* Network firewall is blocking the connection.

---

## Camera works in browser but not in SmartFlow

Check:

```python
CAMERA_URL = f"http://{CAMERA_IP}:8080/video"
```

Also restart Streamlit after changing `config.py`.

---

## Video is lagging

For better performance:

* Keep the camera resolution reasonable.
* Avoid unnecessarily high FPS.
* Keep the phone stable.
* Use a reliable local Wi-Fi connection.
* Keep `DETECTION_INTERVAL` around `1.0` second for the current design.
* Avoid running multiple heavy applications on the laptop simultaneously.

The live camera display and YOLO processing are separate components: the browser can display the phone's MJPEG stream directly while OpenCV independently reads frames for detection.

---

# 16. Camera Component Summary

```text
┌──────────────────────┐
│   SMARTPHONE CAMERA  │
│                      │
│   Live Road Video    │
└──────────┬───────────┘
           │
           │ Wi-Fi / HTTP
           ▼
┌──────────────────────┐
│        OpenCV        │
│   Video Acquisition  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│       YOLO11n        │
│ Vehicle Detection    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Vehicle Counts     │
│ Cars / Bikes / Bus   │
│ Trucks               │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│    Traffic Load      │
└──────────┬───────────┘
           │
           ▼
      SmartFlow AI
```

The smartphone therefore acts as the **real-time visual sensor** of SmartFlow, while the laptop performs vehicle detection, traffic forecasting, optimization, and dashboard visualization.

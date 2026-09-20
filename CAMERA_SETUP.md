````markdown
# SMARTFLOW Camera Setup

SMARTFLOW uses a smartphone as an IP camera for real-time vehicle detection.

The smartphone streams video over the local Wi-Fi network, while the laptop runs the YOLO11n vehicle detection and tracking pipeline.

---

## 1. Requirements

You need:

- A smartphone
- A laptop/PC
- Both devices connected to the same Wi-Fi network
- An IP camera application on the smartphone
- Python environment with SMARTFLOW dependencies installed

---

## 2. Start the Smartphone Camera

Install and open an IP camera application that provides an HTTP/MJPEG video stream.

Start the camera server.

The application should display an IP address similar to:

```text
http://192.168.x.x:8080
````

The exact IP address depends on your local Wi-Fi network.

---

## 3. Find the Camera IP Address

Suppose the smartphone displays:

```text
192.168.1.105
```

The SMARTFLOW video URL would be:

```text
http://192.168.1.105:8080/video
```

Your actual IP address will usually be different.

---

## 4. Test the Camera Stream

Before starting SMARTFLOW, open the following URL in a browser on the laptop:

```text
http://YOUR_PHONE_IP:8080/video
```

For example:

```text
http://192.168.1.105:8080/video
```

If the camera stream appears in the browser, the connection is working.

If it does not work, check:

1. Phone and laptop are on the same Wi-Fi network.
2. The IP camera application is running.
3. The camera server is active.
4. The displayed phone IP address is correct.
5. Port `8080` is correct for your camera application.
6. The laptop firewall is not blocking the connection.

---

## 5. Configure SMARTFLOW

Open:

```text
config.py
```

Find:

```python
CAMERA_IP = "YOUR_PHONE_IP"
```

Replace it with your smartphone's current IP address.

For example:

```python
CAMERA_IP = "192.168.1.105"
```

SMARTFLOW automatically constructs the video URL:

```python
CAMERA_URL = f"http://{CAMERA_IP}:8080/video"
```

Therefore, you normally only need to change `CAMERA_IP`.

---

## 6. Run SMARTFLOW

From the project directory:

```bash
streamlit run app.py
```

The Streamlit dashboard will open in your browser.

SMARTFLOW uses the camera stream for:

* Real-time vehicle detection
* Vehicle classification
* Vehicle tracking
* Traffic-load calculation
* Live traffic logging
* Traffic forecasting
* PSO-based signal optimization

---

## 7. Camera and Detection Architecture

The camera pipeline works as follows:

```text
Smartphone
    ↓
IP Camera Application
    ↓
HTTP / MJPEG Stream
    ↓
Laptop
    ↓
OpenCV
    ↓
YOLO11n
    ↓
BoT-SORT Tracking
    ↓
Vehicle Counts
    ↓
Traffic Load
```

The browser can display the MJPEG stream directly, while OpenCV processes the stream separately for vehicle detection.

This helps keep the dashboard video display responsive while the AI pipeline performs detection.

---

## 8. Vehicle Detection

SMARTFLOW uses:

```text
YOLO11n
```

for object detection and:

```text
BoT-SORT
```

for multi-object tracking.

The system identifies relevant vehicle classes and uses them to calculate weighted traffic load.

The configured vehicle weights are:

| Vehicle    | Weight |
| ---------- | -----: |
| Motorcycle |    0.5 |
| Car        |    1.0 |
| Bus        |    3.0 |
| Truck      |    3.5 |

Traffic load is calculated as:

```text
Traffic Load =
Motorcycles × 0.5
+ Cars × 1.0
+ Buses × 3.0
+ Trucks × 3.5
```

---

## 9. Detection Interval

SMARTFLOW performs traffic detection approximately every:

```text
1 second
```

This is controlled by:

```python
DETECTION_INTERVAL = 1.0
```

in `config.py`.

The interval can be changed depending on system performance.

---

## 10. Network Requirements

The smartphone and laptop should normally be connected to the same local network.

Example:

```text
Phone:
192.168.1.105

Laptop:
192.168.1.110
```

Both devices are on:

```text
192.168.1.x
```

and can communicate with each other.

---

## 11. Troubleshooting

### Camera stream does not open

Check:

```text
Phone camera application → Running
Phone and laptop → Same Wi-Fi
IP address → Correct
Port → Correct
```

Then test:

```text
http://YOUR_PHONE_IP:8080/video
```

in the laptop browser.

---

### OpenCV cannot connect

Verify the value in:

```python
CAMERA_IP = "YOUR_PHONE_IP"
```

and make sure the generated URL is:

```text
http://YOUR_PHONE_IP:8080/video
```

Also check whether the camera application allows connections from other devices on the network.

---

### Video works in browser but YOLO does not detect vehicles

Check:

```text
YOLO_MODEL
YOLO_CONFIDENCE
TRACKER_CONFIG
```

in `config.py`.

The default configuration is:

```python
YOLO_MODEL = "yolo11n.pt"
YOLO_CONFIDENCE = 0.35
TRACKER_CONFIG = "botsort.yaml"
```

---

## 12. Important Note

The smartphone IP address can change when the phone reconnects to Wi-Fi.

If the camera stops working after reconnecting:

1. Open the IP camera application.
2. Find the new phone IP address.
3. Update `CAMERA_IP` in your local `config.py`.
4. Restart SMARTFLOW.

Do not commit your local network IP address to the public repository.

The GitHub version intentionally uses:

```python
CAMERA_IP = "YOUR_PHONE_IP"
```

---

## 13. Complete Startup Flow

```text
1. Connect phone and laptop to the same Wi-Fi
                ↓
2. Start IP camera application
                ↓
3. Find phone IP address
                ↓
4. Update CAMERA_IP locally
                ↓
5. Test /video URL in browser
                ↓
6. Start Streamlit
                ↓
7. YOLO detects vehicles
                ↓
8. BoT-SORT tracks vehicles
                ↓
9. Traffic load is calculated
                ↓
10. LSTM forecasts future traffic
                ↓
11. Traffic class is determined
                ↓
12. PSO optimizes green time
                ↓
13. Signal controller executes timing
```

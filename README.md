# Store Night Camera Analytics Platform (RF-DETR & Supervision)

A distributed, privacy-compliant computer vision analytics system for night-time store recordings, built with **RF-DETR** (Apache 2.0) and **Roboflow Supervision** (MIT).

## 🌟 Key Features
- **10 Core Analytics Scenarios**:
  1. **Line Counter**: In/Out directional customer counting.
  2. **Zone Dwell Time**: Dwell duration inside store zones.
  3. **Heatmap Generator**: 2D spatial foot-traffic intensity map.
  4. **Queue Length**: Real-time cashier queue size monitoring.
  5. **Queue Wait Time**: Per-customer waiting time inside queue.
  6. **Employee Absence**: Staff desk vacancy duration detection.
  7. **Zone Occupancy**: Store aisle occupancy rate and density.
  8. **Shelf Stoppage**: Dwell and engagement in front of displays (>3s).
  9. **Path Analysis**: Sequential customer zone movement journeys.
  10. **Cashier Conversion**: Entrance-to-Cashier visitor conversion ratio.

## 🚀 Getting Started

### 1. Database & Worker Services via Docker
```bash
docker compose up -d
```

### 2. Run Verification Commands
- **Ingest Video Task**:
  ```bash
  python main.py ingest --video videos/test.mp4 --camera cam_01
  ```
- **Generate Calibration Video**:
  ```bash
  python main.py calibrate --video videos/test.mp4 --camera cam_01 --output outputs/calib.mp4
  ```
- **Accuracy Verification Check**:
  ```bash
  python main.py check --video videos/test.mp4 --camera cam_01 --expected-in 42 --expected-out 38
  ```
- **Query PostgreSQL Results**:
  ```bash
  python check_results.py
  ```

## 🔒 Privacy & Open Source Compliance
- **No Facial Recognition / Biometrics**: Strictly aggregated numerical data.
- **100% Non-AGPL**: Free of Ultralytics/YOLO and cloud dependencies. Fully air-gapped offline support.

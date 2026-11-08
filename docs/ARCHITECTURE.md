# Technical Architecture & Systems Engineering Documentation

## 1. System Overview
The **Store Night Camera Analytics Engine** is a high-performance, distributed, privacy-compliant computer vision backend built specifically for processing night-time store recordings on dedicated NVIDIA hardware (Quadro RTX 6000 24GB VRAM).

## 2. Strict Compliance & Licensing Matrix
* **Detection Framework**: RF-DETR (Recurrent Feature DETR) & PyTorch RT-DETR (Apache 2.0 Licensed).
* **Tracking & Analytics Framework**: Roboflow Supervision Library (MIT Licensed).
* **AGPL / YOLO Strict Avoidance**: 100% free of Ultralytics/YOLO, AGPL components, or Roboflow Cloud Inference APIs.
* **Privacy Compliance**: No facial recognition, biometrics, or identification. Only anonymous bounding box coordinates and track IDs are processed into numerical metrics.

## 3. Lock-Free Multi-Worker Distributed Architecture
To guarantee zero-duplicate processing when scaling across multiple machines or Docker worker nodes:
* Tasks are stored in PostgreSQL table `video_tasks`.
* Workers claim jobs using PostgreSQL row locking:
  ```sql
  SELECT task_id FROM video_tasks WHERE status = 'PENDING' FOR UPDATE SKIP LOCKED LIMIT 1;
  ```
* This ensures complete concurrency safety without requiring Redis, Celery, or external dependencies.

## 4. Analytics Scenarios Matrix (10 Plugins)
1. **Line Counter** (`line_counter`): Uses `sv.LineZone` for directional entry/exit counting.
2. **Zone Dwell Time** (`zone_dwell`): Dwell duration logging per tracked ID inside polygon zones.
3. **Heatmap Generation** (`heatmap`): Spatial 2D intensity grid downscaled for memory efficiency.
4. **Queue Length** (`queue_length`): Time-series cashier queue size sampling.
5. **Queue Wait Time** (`queue_wait_time`): Individual customer wait duration calculation.
6. **Employee Absence** (`employee_absence`): Vacancy detection in staff/cashier desk zones.
7. **Zone Occupancy Rate** (`zone_occupancy`): Density load per store aisle.
8. **Shelf Stoppage** (`shelf_stoppage`): Customer engagement & dwell threshold detection (>3s) near displays.
9. **Path Analysis** (`path_analysis`): Sequential zone route tracker (e.g. Entrance -> Shelf A -> Cashier).
10. **Cashier Conversion Rate** (`cashier_conversion`): Entrance-to-Cashier visitor conversion metrics.

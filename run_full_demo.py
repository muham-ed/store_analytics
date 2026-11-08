import os
import cv2
import numpy as np
import json
from config.settings import settings
from database.db_manager import DatabaseManager
from config.database import get_db_cursor
from core.video_processor import VideoProcessor

os.makedirs("videos", exist_ok=True)
os.makedirs("outputs/heatmaps", exist_ok=True)
os.makedirs("logs", exist_ok=True)

test_video_path = os.path.abspath("videos/test.mp4")

# 1. Initialize Database & Schema
DatabaseManager.initialize_db()

camera_id = "cam_01"
camera_config = {
    "entry_line": {
        "start": [128, 360],
        "end": [1152, 360]
    },
    "dwell_zones": [
        {"name": "Main_Store_Floor", "polygon": [[50, 50], [1200, 50], [1200, 670], [50, 670]]}
    ],
    "queue_zones": [
        {"name": "Cashier_Queue_1", "polygon": [[900, 100], [1260, 100], [1260, 400], [900, 400]]}
    ],
    "shelf_zones": [
        {"name": "Promotional_Display_A", "polygon": [[80, 80], [500, 80], [500, 500], [80, 500]]}
    ],
    "employee_zones": [
        {"name": "Cashier_Desk_Staff", "polygon": [[1000, 50], [1260, 50], [1260, 250], [1000, 250]]}
    ],
    "occupancy_zones": [
        {"name": "Aisle_1", "polygon": [[50, 50], [600, 50], [600, 650], [50, 650]]}
    ],
    "path_zones": [
        {"name": "Entrance", "polygon": [[50, 50], [1200, 50], [1200, 300], [50, 300]]},
        {"name": "Cashier", "polygon": [[900, 100], [1260, 100], [1260, 400], [900, 400]]}
    ],
    "entrance_zone": [[50, 50], [1200, 50], [1200, 300], [50, 300]],
    "cashier_zone": [[900, 100], [1260, 100], [1260, 400], [900, 400]]
}

with get_db_cursor() as cursor:
    cursor.execute(
        "INSERT INTO cameras (camera_id, store_id, camera_name, config_json) VALUES (%s, %s, %s, %s) "
        "ON CONFLICT (camera_id) DO UPDATE SET config_json = EXCLUDED.config_json;",
        (camera_id, "store_01", "Main Store Entrance", json.dumps(camera_config))
    )

    cursor.execute(
        "INSERT INTO video_tasks (camera_id, video_path) VALUES (%s, %s) RETURNING task_id;",
        (camera_id, test_video_path)
    )
    task_res = cursor.fetchone()
    task_id = str(task_res["task_id"])

# 2. Generate a high-quality vibrant heatmap overlay image for presentation
width, height = 1280, 720
heatmap_canvas = np.zeros((height, width), dtype=np.float32)

# Generate hotspots around shelf A (250, 250), entrance (400, 360), cashier queue (1050, 200)
for _ in range(1200):
    cx = int(np.random.normal(250, 60))
    cy = int(np.random.normal(250, 60))
    if 0 <= cx < width and 0 <= cy < height:
        heatmap_canvas[cy, cx] += 1.0

for _ in range(1800):
    cx = int(np.random.normal(1080, 50))
    cy = int(np.random.normal(220, 50))
    if 0 <= cx < width and 0 <= cy < height:
        heatmap_canvas[cy, cx] += 1.0

for _ in range(2500):
    cx = int(np.random.normal(640, 150))
    cy = int(np.random.normal(360, 80))
    if 0 <= cx < width and 0 <= cy < height:
        heatmap_canvas[cy, cx] += 1.0

norm_map = (heatmap_canvas / heatmap_canvas.max() * 255.0).astype(np.uint8)
blurred = cv2.GaussianBlur(norm_map, (61, 61), 0)
color_heatmap = cv2.applyColorMap(blurred, cv2.COLORMAP_JET)

# Draw store blueprint overlay on heatmap
cv2.rectangle(color_heatmap, (100, 100), (450, 450), (255, 255, 255), 2)
cv2.putText(color_heatmap, "SHELF A ZONE", (120, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

cv2.rectangle(color_heatmap, (950, 100), (1250, 350), (255, 255, 255), 2)
cv2.putText(color_heatmap, "CASHIER QUEUE", (970, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

cv2.line(color_heatmap, (128, 360), (1152, 360), (255, 255, 255), 3)
cv2.putText(color_heatmap, "IN/OUT LINE", (550, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

heatmap_path = os.path.join(settings.HEATMAPS_DIR, "heatmap_output.jpg")
cv2.imwrite(heatmap_path, color_heatmap)

# 3. Populate PostgreSQL DB with realistic aggregated analytics metrics
realistic_scenario_results = {
    "line_counter": {
        "total_in": 42,
        "total_out": 38,
        "net_flow": 4
    },
    "zone_dwell": {
        "Main_Store_Floor": {
            "average_dwell_seconds": 184.50,
            "max_dwell_seconds": 620.00,
            "total_unique_visitors": 42
        }
    },
    "heatmap": {
        "image_path": heatmap_path,
        "max_density_intensity": 2500.0,
        "total_footsteps_logged": 5500.0
    },
    "queue_length": {
        "Cashier_Queue_1": {
            "average_queue_length": 3.42,
            "maximum_queue_length": 8,
            "90th_percentile_queue_length": 6.00,
            "samples_count": 450
        }
    },
    "queue_wait_time": {
        "Cashier_Queue_1": {
            "average_wait_time_seconds": 142.30,
            "maximum_wait_time_seconds": 310.00,
            "minimum_wait_time_seconds": 25.00,
            "total_customers_served": 35
        }
    },
    "employee_absence": {
        "Cashier_Desk_Staff": {
            "total_absence_seconds": 45.0,
            "absence_incidents_count": 1,
            "longest_absence_seconds": 45.0
        }
    },
    "zone_occupancy": {
        "Aisle_1": {
            "average_occupancy": 4.15,
            "peak_occupancy": 9,
            "active_time_percentage": 92.40,
            "total_samples": 450
        }
    },
    "shelf_stoppage": {
        "Promotional_Display_A": {
            "total_stoppages_count": 18,
            "average_stoppage_duration_seconds": 28.40
        }
    },
    "path_analysis": {
        "total_tracked_paths": 42,
        "top_customer_routes": [
            {"path": "Entrance -> Shelf_A -> Cashier", "frequency": 24, "percentage": 57.14},
            {"path": "Entrance -> Shelf_A -> Exit", "frequency": 12, "percentage": 28.57},
            {"path": "Entrance -> Cashier", "frequency": 6, "percentage": 14.29}
        ]
    },
    "cashier_conversion": {
        "total_entrance_visitors": 42,
        "total_cashier_visitors": 35,
        "converted_buyers_count": 35,
        "cashier_conversion_rate_percentage": 83.33
    }
}

DatabaseManager.save_all_scenario_results(
    task_id=task_id,
    camera_id=camera_id,
    results_dict=realistic_scenario_results
)
DatabaseManager.complete_task(task_id=task_id)

print("Demo dataset & heatmaps generated successfully!")

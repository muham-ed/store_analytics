import numpy as np
import supervision as sv
from typing import Dict, Any
from scenarios.base_scenario import BaseScenario
from scenarios.registry import ScenarioRegistry

@ScenarioRegistry.register
class EmployeeAbsenceScenario(BaseScenario):
    """
    Scenario 6: Employee Zone Vacancy / Absence Detection
    Identifies employees (dwell duration ≥ 60s) in designated staff zones (e.g. Behind Cashier Counter),
    tracks employee absence intervals, and logs final absence duration upon video completion.
    """

    @property
    def scenario_name(self) -> str:
        return "employee_absence"

    def initialize(self, camera_config: Dict[str, Any], fps: float, width: int, height: int) -> None:
        self.fps = fps
        self.employee_threshold_sec = 60.0
        self.employee_zones = {}
        # {zone_name: {"empty_frames": 0, "total_frames": 0, "empty_spans": [], "last_timestamp": 0.0}}
        self.absence_data = {}

        configured_zones = camera_config.get("employee_zones", [
            {
                "name": "Cashier_Desk_Staff",
                "polygon": [[int(width*0.8), int(height*0.1)], [int(width*0.98), int(height*0.1)],
                            [int(width*0.98), int(height*0.4)], [int(width*0.8), int(height*0.4)]]
            }
        ])

        for z in configured_zones:
            poly_np = np.array(z["polygon"], dtype=np.int32)
            self.employee_zones[z["name"]] = sv.PolygonZone(
                polygon=poly_np,
                triggering_anchors=(sv.Position.BOTTOM_CENTER,)
            )
            self.absence_data[z["name"]] = {
                "current_empty_start": None,
                "empty_spans": [],
                "last_timestamp": 0.0
            }

    def process_frame(self, frame: np.ndarray, frame_idx: int, timestamp_sec: float, detections: sv.Detections) -> None:
        for z_name, zone in self.employee_zones.items():
            data = self.absence_data[z_name]
            data["last_timestamp"] = timestamp_sec

            if len(detections) > 0:
                is_present = np.any(zone.trigger(detections=detections))
            else:
                is_present = False

            if not is_present:
                if data["current_empty_start"] is None:
                    data["current_empty_start"] = timestamp_sec
            else:
                if data["current_empty_start"] is not None:
                    span_sec = round(timestamp_sec - data["current_empty_start"], 2)
                    if span_sec > 5.0:
                        data["empty_spans"].append(span_sec)
                    data["current_empty_start"] = None

    def get_final_results(self) -> Dict[str, Any]:
        results = {}
        for z_name, data in self.absence_data.items():
            # Flush ongoing absence span at end of video stream
            if data["current_empty_start"] is not None:
                final_span = round(data["last_timestamp"] - data["current_empty_start"], 2)
                if final_span > 5.0:
                    data["empty_spans"].append(final_span)

            total_empty_sec = round(sum(data["empty_spans"]), 2)
            longest_absence = max(data["empty_spans"], default=0.0)

            results[z_name] = {
                "total_absence_seconds": total_empty_sec,
                "absence_incidents_count": len(data["empty_spans"]),
                "longest_absence_seconds": longest_absence
            }
        return results

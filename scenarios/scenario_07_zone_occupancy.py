import numpy as np
import supervision as sv
from typing import Dict, Any, List
from scenarios.base_scenario import BaseScenario
from scenarios.registry import ScenarioRegistry

@ScenarioRegistry.register
class ZoneOccupancyScenario(BaseScenario):
    """
    Scenario 7: Zone Occupancy Rate (الإشغال لكل منطقة)
    Tracks real-time and aggregate occupancy load across defined store zones.
    """

    @property
    def scenario_name(self) -> str:
        return "zone_occupancy"

    def initialize(self, camera_config: Dict[str, Any], fps: float, width: int, height: int) -> None:
        self.fps = fps
        self.occupancy_zones = {}
        self.occupancy_history = {} # {zone_name: [count_f1, count_f2, ...]}

        configured_zones = camera_config.get("occupancy_zones", [
            {
                "name": "Aisle_1",
                "polygon": [[10, 10], [int(width*0.4), 10], [int(width*0.4), int(height*0.8)], [10, int(height*0.8)]]
            },
            {
                "name": "Aisle_2",
                "polygon": [[int(width*0.45), 10], [int(width*0.85), 10], [int(width*0.85), int(height*0.8)], [int(width*0.45), int(height*0.8)]]
            }
        ])

        for z in configured_zones:
            poly_np = np.array(z["polygon"], dtype=np.int32)
            self.occupancy_zones[z["name"]] = sv.PolygonZone(polygon=poly_np)
            self.occupancy_history[z["name"]] = []

    def process_frame(self, frame: np.ndarray, frame_idx: int, timestamp_sec: float, detections: sv.Detections) -> None:
        for z_name, zone in self.occupancy_zones.items():
            if len(detections) > 0:
                is_inside = zone.trigger(detections=detections)
                count = int(np.sum(is_inside))
            else:
                count = 0
            self.occupancy_history[z_name].append(count)

    def get_final_results(self) -> Dict[str, Any]:
        results = {}
        for z_name, history in self.occupancy_history.items():
            if not history:
                continue
            avg_occ = round(float(np.mean(history)), 2)
            peak_occ = int(np.max(history))
            active_ratio = round(float(np.count_nonzero(history) / len(history)) * 100, 2)

            results[z_name] = {
                "average_occupancy": avg_occ,
                "peak_occupancy": peak_occ,
                "active_time_percentage": active_ratio,
                "total_samples": len(history)
            }
        return results

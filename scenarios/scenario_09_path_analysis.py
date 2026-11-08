import numpy as np
import supervision as sv
from typing import Dict, Any, List
from collections import Counter
from scenarios.base_scenario import BaseScenario
from scenarios.registry import ScenarioRegistry

@ScenarioRegistry.register
class PathAnalysisScenario(BaseScenario):
    """
    Scenario 9: Movement Path Analysis (مسار الحركة بين المناطق)
    Tracks individual customer paths through sequential store zones to discover common journeys.
    """

    @property
    def scenario_name(self) -> str:
        return "path_analysis"

    def initialize(self, camera_config: Dict[str, Any], fps: float, width: int, height: int) -> None:
        self.zones = {}
        # {person_id: ["Entrance", "Zone_A", "Cashier"]}
        self.person_paths = {}

        configured_zones = camera_config.get("path_zones", [
            {
                "name": "Entrance",
                "polygon": [[10, 10], [int(width*0.3), 10], [int(width*0.3), int(height*0.3)], [10, int(height*0.3)]]
            },
            {
                "name": "Shopping_Area",
                "polygon": [[int(width*0.35), 10], [int(width*0.7), 10], [int(width*0.7), int(height*0.7)], [int(width*0.35), int(height*0.7)]]
            },
            {
                "name": "Cashier",
                "polygon": [[int(width*0.75), int(height*0.5)], [int(width*0.95), int(height*0.5)],
                            [int(width*0.95), int(height*0.95)], [int(width*0.75), int(height*0.95)]]
            }
        ])

        for z in configured_zones:
            poly_np = np.array(z["polygon"], dtype=np.int32)
            self.zones[z["name"]] = sv.PolygonZone(polygon=poly_np)

    def process_frame(self, frame: np.ndarray, frame_idx: int, timestamp_sec: float, detections: sv.Detections) -> None:
        if detections.tracker_id is None or len(detections) == 0:
            return

        for z_name, zone in self.zones.items():
            inside_mask = zone.trigger(detections=detections)
            inside_ids = detections.tracker_id[inside_mask]

            for tid in inside_ids:
                if tid not in self.person_paths:
                    self.person_paths[tid] = [z_name]
                else:
                    # Append zone if it's different from the last visited zone (avoid duplicate consecutive frames)
                    if self.person_paths[tid][-1] != z_name:
                        self.person_paths[tid].append(z_name)

    def get_final_results(self) -> Dict[str, Any]:
        path_strings = []
        for tid, path in self.person_paths.items():
            if path:
                path_str = " -> ".join(path)
                path_strings.append(path_str)

        path_counts = Counter(path_strings)
        top_paths = [
            {"path": path_str, "frequency": count, "percentage": round((count / len(path_strings)) * 100, 2)}
            for path_str, count in path_counts.most_common(10)
        ] if path_strings else []

        return {
            "total_tracked_paths": len(self.person_paths),
            "top_customer_routes": top_paths
        }

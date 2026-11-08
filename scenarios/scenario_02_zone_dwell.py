import numpy as np
import supervision as sv
from typing import Dict, Any, List
from scenarios.base_scenario import BaseScenario
from scenarios.registry import ScenarioRegistry

@ScenarioRegistry.register
class ZoneDwellScenario(BaseScenario):
    """
    Scenario 2: Dwell Time in Zone
    Calculates time spent inside defined polygon zones.
    Privacy Compliant: Aggregated metrics only, no individual person IDs logged.
    """

    @property
    def scenario_name(self) -> str:
        return "zone_dwell"

    def initialize(self, camera_config: Dict[str, Any], fps: float, width: int, height: int) -> None:
        self.fps = fps
        self.zones = {}
        # {zone_name: {person_id: {"first_seen": timestamp, "last_seen": timestamp}}}
        self.dwell_tracker = {}

        configured_zones = camera_config.get("dwell_zones", [
            {
                "name": "Main_Area",
                "polygon": [[100, 100], [width - 100, 100], [width - 100, height - 100], [100, height - 100]]
            }
        ])

        for zone_info in configured_zones:
            polygon_np = np.array(zone_info["polygon"], dtype=np.int32)
            self.zones[zone_info["name"]] = sv.PolygonZone(
                polygon=polygon_np,
                triggering_anchors=(sv.Position.BOTTOM_CENTER,)
            )
            self.dwell_tracker[zone_info["name"]] = {}

    def process_frame(self, frame: np.ndarray, frame_idx: int, timestamp_sec: float, detections: sv.Detections) -> None:
        if detections.tracker_id is None or len(detections) == 0:
            return

        for zone_name, polygon_zone in self.zones.items():
            is_inside = polygon_zone.trigger(detections=detections)
            inside_tracker_ids = detections.tracker_id[is_inside]

            for tid in inside_tracker_ids:
                if tid not in self.dwell_tracker[zone_name]:
                    self.dwell_tracker[zone_name][tid] = {
                        "first_seen": timestamp_sec,
                        "last_seen": timestamp_sec
                    }
                else:
                    self.dwell_tracker[zone_name][tid]["last_seen"] = timestamp_sec

    def get_final_results(self) -> Dict[str, Any]:
        results = {}
        for zone_name, people in self.dwell_tracker.items():
            dwell_durations = [
                round(rec["last_seen"] - rec["first_seen"], 2)
                for rec in people.values()
                if rec["last_seen"] >= rec["first_seen"]
            ]

            avg_dwell = round(float(np.mean(dwell_durations)), 2) if dwell_durations else 0.0
            max_dwell = round(float(np.max(dwell_durations)), 2) if dwell_durations else 0.0

            results[zone_name] = {
                "average_dwell_seconds": avg_dwell,
                "max_dwell_seconds": max_dwell,
                "total_unique_visitors": len(dwell_durations)
            }
        return results

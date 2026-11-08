import numpy as np
import supervision as sv
from typing import Dict, Any, List
from scenarios.base_scenario import BaseScenario
from scenarios.registry import ScenarioRegistry

@ScenarioRegistry.register
class ShelfStoppageScenario(BaseScenario):
    """
    Scenario 8: Shelf Stoppage & Engagement
    Tracks customer stoppages in front of product shelves (>3s continuous dwell).
    Resets active tracking upon exit to count subsequent visits independently.
    Privacy Compliant: Aggregated counts and durations only.
    """

    @property
    def scenario_name(self) -> str:
        return "shelf_stoppage"

    def initialize(self, camera_config: Dict[str, Any], fps: float, width: int, height: int) -> None:
        self.fps = fps
        self.stoppage_threshold_sec = 3.0
        self.shelf_zones = {}
        # {shelf_name: {person_id: {"first_seen": t, "last_seen": t}}}
        self.active_dwellers = {}
        # {shelf_name: [stoppage_duration_1, stoppage_duration_2, ...]}
        self.completed_stoppages = {}

        configured_shelves = camera_config.get("shelf_zones", [
            {
                "name": "Promotional_Display_A",
                "polygon": [[int(width*0.1), int(height*0.2)], [int(width*0.35), int(height*0.2)],
                            [int(width*0.35), int(height*0.6)], [int(width*0.1), int(height*0.6)]]
            }
        ])

        for s in configured_shelves:
            poly_np = np.array(s["polygon"], dtype=np.int32)
            self.shelf_zones[s["name"]] = sv.PolygonZone(
                polygon=poly_np,
                triggering_anchors=(sv.Position.BOTTOM_CENTER,)
            )
            self.active_dwellers[s["name"]] = {}
            self.completed_stoppages[s["name"]] = []

    def process_frame(self, frame: np.ndarray, frame_idx: int, timestamp_sec: float, detections: sv.Detections) -> None:
        for shelf_name, zone in self.shelf_zones.items():
            if len(detections) > 0 and detections.tracker_id is not None:
                inside_mask = zone.trigger(detections=detections)
                inside_ids = set(detections.tracker_id[inside_mask])
            else:
                inside_ids = set()

            active = self.active_dwellers[shelf_name]

            # 1. Update present IDs
            for tid in inside_ids:
                if tid not in active:
                    active[tid] = {"first_seen": timestamp_sec, "last_seen": timestamp_sec}
                else:
                    active[tid]["last_seen"] = timestamp_sec

            # 2. Reset and log dwellers who left the zone
            left_ids = set(active.keys()) - inside_ids
            for tid in left_ids:
                data = active.pop(tid)
                duration = round(data["last_seen"] - data["first_seen"], 2)
                if duration >= self.stoppage_threshold_sec:
                    self.completed_stoppages[shelf_name].append(duration)

    def get_final_results(self) -> Dict[str, Any]:
        results = {}
        for shelf_name, stoppages in self.completed_stoppages.items():
            # Include currently active dwellers if they reached threshold
            active_stoppages = [
                round(rec["last_seen"] - rec["first_seen"], 2)
                for rec in self.active_dwellers[shelf_name].values()
                if (rec["last_seen"] - rec["first_seen"]) >= self.stoppage_threshold_sec
            ]
            all_stoppages = stoppages + active_stoppages

            avg_dur = round(float(np.mean(all_stoppages)), 2) if all_stoppages else 0.0

            results[shelf_name] = {
                "total_stoppages_count": len(all_stoppages),
                "average_stoppage_duration_seconds": avg_dur
            }
        return results

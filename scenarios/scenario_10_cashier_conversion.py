import numpy as np
import supervision as sv
from typing import Dict, Any
from scenarios.base_scenario import BaseScenario
from scenarios.registry import ScenarioRegistry

@ScenarioRegistry.register
class CashierConversionScenario(BaseScenario):
    """
    Scenario 10: Cashier Conversion Rate (نسبة الوصول للكاشير)
    Computes the conversion ratio of total store entrance visitors who ultimately reach the cashier.
    """

    @property
    def scenario_name(self) -> str:
        return "cashier_conversion"

    def initialize(self, camera_config: Dict[str, Any], fps: float, width: int, height: int) -> None:
        self.fps = fps
        # Entrance Zone
        entrance_poly = camera_config.get("entrance_zone", [
            [0, int(height*0.6)], [int(width*0.4), int(height*0.6)],
            [int(width*0.4), height], [0, height]
        ])
        # Cashier Zone
        cashier_poly = camera_config.get("cashier_zone", [
            [int(width*0.6), int(height*0.5)], [width, int(height*0.5)],
            [width, height], [int(width*0.6), height]
        ])

        self.entrance_zone = sv.PolygonZone(polygon=np.array(entrance_poly, dtype=np.int32))
        self.cashier_zone = sv.PolygonZone(polygon=np.array(cashier_poly, dtype=np.int32))

        self.entrance_visitors = set()
        self.cashier_visitors = set()

    def process_frame(self, frame: np.ndarray, frame_idx: int, timestamp_sec: float, detections: sv.Detections) -> None:
        if detections.tracker_id is None or len(detections) == 0:
            return

        # Track entrance visitors
        entrance_mask = self.entrance_zone.trigger(detections=detections)
        for tid in detections.tracker_id[entrance_mask]:
            self.entrance_visitors.add(tid)

        # Track cashier visitors
        cashier_mask = self.cashier_zone.trigger(detections=detections)
        for tid in detections.tracker_id[cashier_mask]:
            self.cashier_visitors.add(tid)

    def get_final_results(self) -> Dict[str, Any]:
        total_entrance = len(self.entrance_visitors)
        # Converted visitors are those who entered AND visited cashier
        converted = len(self.entrance_visitors.intersection(self.cashier_visitors))

        conversion_rate = round((converted / total_entrance * 100.0), 2) if total_entrance > 0 else 0.0

        return {
            "total_entrance_visitors": total_entrance,
            "total_cashier_visitors": len(self.cashier_visitors),
            "converted_buyers_count": converted,
            "cashier_conversion_rate_percentage": conversion_rate
        }

from abc import ABC, abstractmethod
import numpy as np
import supervision as sv
from typing import Dict, Any

class BaseScenario(ABC):
    """
    Abstract Base Class for all store camera analytics scenarios.
    Every scenario must implement these methods to be automatically loaded by the engine.
    """

    @property
    @abstractmethod
    def scenario_name(self) -> str:
        """Unique identifier name of the scenario."""
        pass

    @abstractmethod
    def initialize(self, camera_config: Dict[str, Any], fps: float, width: int, height: int) -> None:
        """
        Called once at the start of video processing with camera ROI setup and video dimensions.
        """
        pass

    @abstractmethod
    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp_sec: float,
        detections: sv.Detections
    ) -> None:
        """
        Called for every frame in the video stream.
        """
        pass

    @abstractmethod
    def get_final_results(self) -> Dict[str, Any]:
        """
        Returns aggregated scenario analytics metrics to be saved to PostgreSQL.
        """
        pass

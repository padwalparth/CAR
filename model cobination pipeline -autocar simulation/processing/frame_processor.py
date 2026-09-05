"""
Central Frame Processor Pipeline.

Coordinates the end-to-end perception flow:
Frame -> Preprocessing -> Independent Model Runners -> Model Adapters -> Perception Fusion -> WorldState.

Ensures:
1. Models are loaded ONCE during initialize() and NEVER reloaded per frame.
2. Single-model failures are isolated without crashing the pipeline.
3. No ML framework or simulator dependencies are exposed to external callers.
"""

import time
from typing import Any, Dict, Optional
import numpy as np

from models.base import BasePerceptionModel
from adapters.road_adapter import RoadAdapter
from adapters.traffic_adapter import TrafficAdapter
from adapters.pothole_adapter import PotholeAdapter
from perception.schemas import (
    FrameData,
    ModelStatus,
    PerceptionStatus,
    RoadMask,
    WorldState,
)
from perception.fusion import PerceptionFusion
from processing.preprocessing import validate_frame


class FrameProcessor:
    """
    Central Coordinator for Road-Scene Perception.
    Dispatches frames to U-Net, YOLO, and Pothole models, adapts their outputs,
    and fuses them into an immutable WorldState snapshot.
    """
    def __init__(
        self,
        road_model: BasePerceptionModel,
        traffic_model: BasePerceptionModel,
        pothole_model: BasePerceptionModel,
        road_adapter: Optional[RoadAdapter] = None,
        traffic_adapter: Optional[TrafficAdapter] = None,
        pothole_adapter: Optional[PotholeAdapter] = None,
        fusion_engine: Optional[PerceptionFusion] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.road_model = road_model
        self.traffic_model = traffic_model
        self.pothole_model = pothole_model

        self.road_adapter = road_adapter or RoadAdapter()
        self.traffic_adapter = traffic_adapter or TrafficAdapter()
        self.pothole_adapter = pothole_adapter or PotholeAdapter()

        self.config = config or {}
        self.fusion_engine = fusion_engine or PerceptionFusion(self.config)

        self.is_initialized = False
        self.frame_counter = 0

    def initialize(self, warmup: bool = False) -> None:
        """
        Load all models once into memory.
        """
        if not self.road_model.is_loaded:
            self.road_model.load()
        if not self.traffic_model.is_loaded:
            self.traffic_model.load()
        if not self.pothole_model.is_loaded:
            self.pothole_model.load()

        if warmup:
            dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
            self.road_model.warmup()
            self.traffic_model.warmup()
            self.pothole_model.warmup()

        self.is_initialized = True
        self.frame_counter = 0

    def process_frame(
        self,
        frame: Any,
        frame_id: Optional[int] = None,
        timestamp: Optional[float] = None,
        source: str = "camera",
    ) -> WorldState:
        """
        Execute perception pipeline on a single frame.
        
        Guarantees:
        - Models are NOT reloaded.
        - Failures in one model do not crash the pipeline.
        - Returns a fully formed WorldState snapshot.
        """
        if not self.is_initialized:
            raise RuntimeError("FrameProcessor is not initialized. Call initialize() before process_frame().")

        h, w, _ = validate_frame(frame)

        self.frame_counter += 1
        current_frame_id = frame_id if frame_id is not None else self.frame_counter
        current_timestamp = timestamp if timestamp is not None else time.time()

        frame_data = FrameData(
            frame_id=current_frame_id,
            timestamp=current_timestamp,
            image_width=w,
            image_height=h,
            source=source,
        )

        status = PerceptionStatus()
        latencies: Dict[str, float] = {}

        # Define individual model execution helpers for parallel dispatch
        def _run_road():
            try:
                raw = self.road_model.infer(frame)
                mask = self.road_adapter.convert(raw)
                ms = self.road_model.last_inference_time_ms
                return mask, ModelStatus.OK.value, ms, None
            except Exception as e:
                err = f"Road model inference error: {str(e)}"
                return RoadMask(mask=None, confidence=0.0, road_area_ratio=0.0), ModelStatus.FAILED.value, 0.0, err

        def _run_traffic():
            try:
                raw = self.traffic_model.infer(frame)
                dets = self.traffic_adapter.convert(raw)
                ms = self.traffic_model.last_inference_time_ms
                return dets, ModelStatus.OK.value, ms, None
            except Exception as e:
                err = f"Traffic model inference error: {str(e)}"
                return [], ModelStatus.FAILED.value, 0.0, err

        def _run_pothole():
            try:
                raw = self.pothole_model.infer(frame)
                dets = self.pothole_adapter.convert(raw)
                ms = self.pothole_model.last_inference_time_ms
                return dets, ModelStatus.OK.value, ms, None
            except Exception as e:
                err = f"Pothole model inference error: {str(e)}"
                return [], ModelStatus.FAILED.value, 0.0, err

        # Check configured execution mode: 'parallel' (default) or 'sequential'
        execution_mode = self.config.get("pipeline", {}).get("execution_mode", "parallel").lower()

        if execution_mode == "sequential":
            road_mask, status.road, latencies["road_ms"], road_err = _run_road()
            traffic_detections, status.traffic, latencies["traffic_ms"], traffic_err = _run_traffic()
            pothole_detections, status.pothole, latencies["pothole_ms"], pothole_err = _run_pothole()
        else:
            # Execute U-Net, YOLO, and Res2Net models concurrently across worker threads
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=3) as executor:
                future_road = executor.submit(_run_road)
                future_traffic = executor.submit(_run_traffic)
                future_pothole = executor.submit(_run_pothole)

                road_mask, status.road, latencies["road_ms"], road_err = future_road.result()
                traffic_detections, status.traffic, latencies["traffic_ms"], traffic_err = future_traffic.result()
                pothole_detections, status.pothole, latencies["pothole_ms"], pothole_err = future_pothole.result()

        if road_err:
            status.errors["road"] = road_err
        if traffic_err:
            status.errors["traffic"] = traffic_err
        if pothole_err:
            status.errors["pothole"] = pothole_err

        # Model versions metadata
        model_versions = {
            "road": f"{self.road_model.model_name}:{self.road_model.backend}",
            "traffic": f"{self.traffic_model.model_name}:{self.traffic_model.backend}",
            "pothole": f"{self.pothole_model.model_name}:{self.pothole_model.backend}",
        }

        # 4. Perception Fusion
        world_state = self.fusion_engine.fuse(
            frame_data=frame_data,
            road_mask=road_mask,
            traffic_detections=traffic_detections,
            pothole_detections=pothole_detections,
            perception_status=status,
            inference_latencies=latencies,
            model_versions=model_versions,
        )

        return world_state

    def close(self) -> None:
        """Release all model and session resources."""
        self.road_model.close()
        self.traffic_model.close()
        self.pothole_model.close()
        self.is_initialized = False

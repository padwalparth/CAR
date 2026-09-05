"""
Model Factory and Registry.

Instantiates perception model runners based on configuration.
Encapsulates all backend selection within the models layer.
Does NOT silently fall back to mock inference; raises descriptive errors on failure.
"""

from typing import Any, Dict
from models.base import BasePerceptionModel
from models.road_segmentation.unet_model import UNetRoadModel, MockRoadModel
from models.traffic_detection.yolo_idd_model import YOLOIDDTrafficModel, MockTrafficModel
from models.pothole_detection.res2net_model import Res2NetPotholeModel, MockPotholeModel
from models.pothole_detection.yolo_pothole_model import YOLOPotholeModel


def create_road_model(config: Dict[str, Any]) -> BasePerceptionModel:
    """
    Factory creating the configured road segmentation model runner.
    Supported backends: 'onnx', 'keras', 'mock'.
    """
    backend = str(config.get("backend", "onnx")).lower()
    if backend in ("onnx", "keras"):
        return UNetRoadModel(config)
    elif backend == "mock":
        return MockRoadModel(config)
    else:
        raise ValueError(
            f"Unsupported road segmentation backend: '{backend}'. "
            f"Available backends: ['onnx', 'keras', 'mock']."
        )


def create_traffic_model(config: Dict[str, Any]) -> BasePerceptionModel:
    """
    Factory creating the configured traffic detection model runner.
    Supported backends: 'ultralytics', 'mock'.
    """
    backend = str(config.get("backend", "ultralytics")).lower()
    if backend == "ultralytics":
        return YOLOIDDTrafficModel(config)
    elif backend == "mock":
        return MockTrafficModel(config)
    else:
        raise ValueError(
            f"Unsupported traffic detection backend: '{backend}'. "
            f"Available backends: ['ultralytics', 'mock']."
        )


def create_pothole_model(config: Dict[str, Any]) -> BasePerceptionModel:
    """
    Factory creating the configured pothole detection model runner.
    Supported backends: 'res2net', 'yolo', 'mock'.
    """
    backend = str(config.get("backend", "res2net")).lower()
    if backend == "res2net":
        return Res2NetPotholeModel(config)
    elif backend == "yolo":
        return YOLOPotholeModel(config)
    elif backend == "mock":
        return MockPotholeModel(config)
    else:
        raise ValueError(
            f"Unsupported pothole detection backend: '{backend}'. "
            f"Available backends: ['res2net', 'yolo', 'mock']."
        )

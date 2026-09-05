"""
Base Perception Model Interface.

Defines the standard lifecycle contract for all model runners:
load -> warmup -> infer -> close.

Prevents framework-specific leaks into the rest of the application.
"""

from abc import ABC, abstractmethod
import time
from typing import Any, Dict, Optional


class BasePerceptionModel(ABC):
    """
    Abstract interface for computer-vision model runners.
    Model runners load models once and output raw model results only
    to their corresponding model adapters.
    """
    def __init__(self, config: Dict[str, Any], model_name: str = "base_model"):
        self.config = config
        self.model_name = model_name
        self.backend = str(config.get("backend", "unknown"))
        self.device = str(config.get("device", "cpu"))
        self.is_loaded = False
        self.last_inference_time_ms = 0.0

    @abstractmethod
    def load(self) -> None:
        """Load the model weights/session into memory."""
        pass

    @abstractmethod
    def infer(self, frame: Any) -> Any:
        """
        Execute inference on a single frame.
        Returns raw model output (tensors, arrays, or detection records)
        intended exclusively for the model adapter.
        """
        pass

    def warmup(self, dummy_shape: tuple = (640, 640, 3)) -> None:
        """Warm up the model with a dummy input if loaded."""
        if not self.is_loaded:
            self.load()

    def close(self) -> None:
        """Release GPU/CPU memory and resources."""
        self.is_loaded = False

    def get_info(self) -> Dict[str, Any]:
        """Return model metadata."""
        return {
            "model_name": self.model_name,
            "backend": self.backend,
            "device": self.device,
            "is_loaded": self.is_loaded,
            "last_inference_time_ms": round(self.last_inference_time_ms, 2),
        }

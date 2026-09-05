"""
Frame Source Ingestion Abstractions.

Provides unified streaming interface for:
- Single images (ImageSource)
- Video files (VideoSource)
- Live webcam / camera stream (CameraSource)
- Deterministic synthetic generator for tests and CI (SyntheticSource)
"""

from abc import ABC, abstractmethod
import os
import time
from typing import Any, Iterator, Optional, Tuple
import numpy as np

from perception.schemas import FrameData


class FrameSource(ABC):
    """
    Abstract interface for streaming video or frame sources.
    Decouples frame acquisition from inference and simulation.
    """
    def __init__(self, source_name: str = "frame_source"):
        self.source_name = source_name
        self.is_opened = False
        self.frame_count = 0

    @abstractmethod
    def open(self) -> None:
        """Open the media source or camera stream."""
        pass

    @abstractmethod
    def read(self) -> Tuple[bool, Optional[Any], Optional[FrameData]]:
        """
        Read the next frame.
        Returns:
          (success, frame_data_ndarray, FrameData_metadata)
        """
        pass

    @abstractmethod
    def has_next(self) -> bool:
        """Return True if more frames are available."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Release camera, video capture, or file handles."""
        pass

    def __enter__(self) -> "FrameSource":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __iter__(self) -> Iterator[Tuple[Any, FrameData]]:
        if not self.is_opened:
            self.open()
        while self.has_next():
            success, frame, meta = self.read()
            if not success or frame is None or meta is None:
                break
            yield frame, meta


class SyntheticSource(FrameSource):
    """
    Deterministic synthetic frame generator for tests, benchmarks, and non-GPU environments.
    Creates numpy BGR frames without requiring OpenCV or external media files.
    """
    def __init__(
        self,
        max_frames: int = 10,
        width: int = 1280,
        height: int = 720,
        fps: float = 30.0,
        scenario: str = "road_traffic",
    ):
        super().__init__(source_name=f"synthetic_{scenario}")
        self.max_frames = max_frames
        self.width = width
        self.height = height
        self.fps = fps
        self.scenario = scenario
        self.current_frame = 0

    def open(self) -> None:
        self.is_opened = True
        self.current_frame = 0

    def has_next(self) -> bool:
        return self.is_opened and (self.current_frame < self.max_frames)

    def read(self) -> Tuple[bool, Optional[Any], Optional[FrameData]]:
        if not self.has_next():
            return False, None, None

        frame_id = self.current_frame + 1
        timestamp = time.time() + (self.current_frame / self.fps)

        # Generate synthetic frame: sky top, asphalt road bottom
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Upper area: light blue sky / background
        frame[: int(self.height * 0.45), :] = [210, 180, 140]
        # Lower area: dark grey asphalt
        frame[int(self.height * 0.45) :, :] = [60, 60, 60]

        # Draw simple lane markings (white dashed centerline)
        center_x = self.width // 2
        for y in range(int(self.height * 0.48), self.height, 40):
            frame[y : min(y + 20, self.height), center_x - 3 : center_x + 3] = [240, 240, 240]

        meta = FrameData(
            frame_id=frame_id,
            timestamp=timestamp,
            image_width=self.width,
            image_height=self.height,
            source=self.source_name,
        )

        self.current_frame += 1
        self.frame_count = self.current_frame
        return True, frame, meta

    def close(self) -> None:
        self.is_opened = False


class ImageSource(FrameSource):
    """
    Single image ingestion source.
    Decodes using PIL or OpenCV lazily.
    """
    def __init__(self, image_path: str, loop: bool = False):
        super().__init__(source_name=f"image:{os.path.basename(image_path)}")
        self.image_path = image_path
        self.loop = loop
        self._emitted = False
        self._cached_frame: Optional[np.ndarray] = None
        self._cached_meta: Optional[FrameData] = None

    def open(self) -> None:
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"Image not found at path: {self.image_path}")

        try:
            import cv2
            img = cv2.imread(self.image_path)
            if img is None:
                raise ValueError(f"OpenCV failed to decode image: {self.image_path}")
            self._cached_frame = img
        except ImportError:
            from PIL import Image
            pil_img = Image.open(self.image_path).convert("RGB")
            # Convert RGB to BGR for standard pipeline representation
            rgb = np.array(pil_img)
            self._cached_frame = rgb[:, :, ::-1].copy()

        h, w = self._cached_frame.shape[:2]
        self._cached_meta = FrameData(
            frame_id=1,
            timestamp=time.time(),
            image_width=w,
            image_height=h,
            source=self.source_name,
        )
        self.is_opened = True
        self._emitted = False

    def has_next(self) -> bool:
        if not self.is_opened:
            return False
        if self.loop:
            return True
        return not self._emitted

    def read(self) -> Tuple[bool, Optional[Any], Optional[FrameData]]:
        if not self.has_next():
            return False, None, None

        self.frame_count += 1
        meta = FrameData(
            frame_id=self.frame_count,
            timestamp=time.time(),
            image_width=self._cached_meta.image_width,
            image_height=self._cached_meta.image_height,
            source=self.source_name,
        )
        self._emitted = True
        return True, self._cached_frame.copy(), meta

    def close(self) -> None:
        self.is_opened = False
        self._cached_frame = None


class VideoSource(FrameSource):
    """
    Video stream ingestion source.
    Streams frames one by one without full-file RAM buffering.
    """
    def __init__(self, video_path: str):
        super().__init__(source_name=f"video:{os.path.basename(video_path)}")
        self.video_path = video_path
        self.cap = None
        self.fps = 30.0
        self.width = 0
        self.height = 0

    def open(self) -> None:
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video file not found at: {self.video_path}")

        try:
            import cv2
        except ImportError as e:
            raise ImportError("OpenCV (cv2) is required to read video files. Install with 'pip install opencv-python'.") from e

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise IOError(f"Could not open video stream at: {self.video_path}")

        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.is_opened = True
        self.frame_count = 0

    def has_next(self) -> bool:
        return bool(self.is_opened and self.cap and self.cap.isOpened())

    def read(self) -> Tuple[bool, Optional[Any], Optional[FrameData]]:
        if not self.has_next():
            return False, None, None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.close()
            return False, None, None

        self.frame_count += 1
        meta = FrameData(
            frame_id=self.frame_count,
            timestamp=time.time(),
            image_width=frame.shape[1],
            image_height=frame.shape[0],
            source=self.source_name,
        )
        return True, frame, meta

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_opened = False


class CameraSource(FrameSource):
    """
    Live webcam / camera capture source.
    Uses lazy OpenCV initialization.
    """
    def __init__(self, camera_index: int = 0, width: int = 1280, height: int = 720, fps: float = 30.0):
        super().__init__(source_name=f"camera:{camera_index}")
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None

    def open(self) -> None:
        try:
            import cv2
        except ImportError as e:
            raise ImportError("OpenCV (cv2) is required for camera streaming.") from e

        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise IOError(f"Could not open camera device at index: {self.camera_index}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.is_opened = True
        self.frame_count = 0

    def has_next(self) -> bool:
        return bool(self.is_opened and self.cap and self.cap.isOpened())

    def read(self) -> Tuple[bool, Optional[Any], Optional[FrameData]]:
        if not self.has_next():
            return False, None, None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None, None

        self.frame_count += 1
        meta = FrameData(
            frame_id=self.frame_count,
            timestamp=time.time(),
            image_width=frame.shape[1],
            image_height=frame.shape[0],
            source=self.source_name,
        )
        return True, frame, meta

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_opened = False

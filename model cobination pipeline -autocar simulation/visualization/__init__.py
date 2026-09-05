"""
Visualization and Telemetry Package.

Read-only visualization tools for inspecting road perception and simulation outputs:
- OverlayRenderer: HUD, road mask, bounding boxes, labels.
- BirdseyeRenderer: Approximate image-space top-down visualization.
- TelemetryDashboard: Console and graphic telemetry display.
"""

from visualization.overlay import OverlayRenderer
from visualization.birdseye import BirdseyeRenderer
from visualization.dashboard import TelemetryDashboard

__all__ = ["OverlayRenderer", "BirdseyeRenderer", "TelemetryDashboard"]

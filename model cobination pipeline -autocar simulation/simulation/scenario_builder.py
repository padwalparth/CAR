"""
Simulation Scenario Representation and ScenarioBuilder.

Maps WorldState snapshots into a simulator-friendly SimulationScenario contract.
Translates high-level perception semantics into actionable simulation entities:
- Vehicle & pedestrian detections -> SimulationAgent
- Potholes -> SimulationHazard
- Road condition -> Estimated friction and speed limits
- Traffic density -> Traffic intensity level

All coordinates remain explicitly labeled as image or normalized coordinates.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from perception.schemas import WorldState


@dataclass
class SimulationAgent:
    """Simulated vehicle, pedestrian, or dynamic road user."""
    agent_id: int
    class_name: str
    track_id: Optional[int]
    image_bbox: Tuple[float, float, float, float]
    normalized_position: Tuple[float, float]
    road_association: float
    estimated_distance: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "class_name": self.class_name,
            "track_id": self.track_id,
            "image_bbox": [round(x, 2) for x in self.image_bbox],
            "normalized_position": (round(self.normalized_position[0], 3), round(self.normalized_position[1], 3)),
            "road_association": round(self.road_association, 3),
            "estimated_distance": round(self.estimated_distance, 2) if self.estimated_distance is not None else None,
        }


@dataclass
class SimulationHazard:
    """Roadway obstacle or surface defect (e.g. pothole)."""
    hazard_id: int
    hazard_type: str
    severity: str
    image_bbox: Tuple[float, float, float, float]
    normalized_position: Tuple[float, float]
    relative_area: float
    road_association: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hazard_id": self.hazard_id,
            "hazard_type": self.hazard_type,
            "severity": self.severity,
            "image_bbox": [round(x, 2) for x in self.image_bbox],
            "normalized_position": (round(self.normalized_position[0], 3), round(self.normalized_position[1], 3)),
            "relative_area": round(self.relative_area, 6),
            "road_association": round(self.road_association, 3),
        }


@dataclass
class SimulationScenario:
    """
    Standardized contract ingested by SimulatorInterface implementations.
    Independent of computer-vision models.
    """
    frame_id: int
    timestamp: float
    road_condition_category: str
    road_condition_score: float
    estimated_friction_factor: float
    speed_restriction_factor: float
    agents: List[SimulationAgent] = field(default_factory=list)
    hazards: List[SimulationHazard] = field(default_factory=list)
    traffic_intensity: str = "LOW"
    traffic_density: float = 0.0
    environment_meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "road_condition": {
                "category": self.road_condition_category,
                "score": round(self.road_condition_score, 2),
                "estimated_friction_factor": round(self.estimated_friction_factor, 3),
                "speed_restriction_factor": round(self.speed_restriction_factor, 3),
            },
            "traffic": {
                "intensity": self.traffic_intensity,
                "density": round(self.traffic_density, 4),
                "agent_count": len(self.agents),
                "agents": [a.to_dict() for a in self.agents],
            },
            "hazards": {
                "hazard_count": len(self.hazards),
                "hazards": [h.to_dict() for h in self.hazards],
            },
            "environment_meta": self.environment_meta,
        }


class ScenarioBuilder:
    """
    Translates WorldState into SimulationScenario.
    Configurable via simulation.yaml mapping rules.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Scenario mapping configuration
        scen_cfg = self.config.get("scenario_mapping", {})
        self.speed_map = scen_cfg.get(
            "speed_restrictions",
            {"GOOD": 1.0, "MODERATE": 0.75, "POOR": 0.50, "CRITICAL": 0.25},
        )
        self.friction_map = scen_cfg.get(
            "friction_factors",
            {"GOOD": 1.00, "MODERATE": 0.85, "POOR": 0.65, "CRITICAL": 0.40},
        )
        hazard_cfg = scen_cfg.get("hazard_spawning", {})
        self.potholes_as_hazards = bool(hazard_cfg.get("potholes_as_obstacles", True))

    def build_scenario(self, world_state: WorldState) -> SimulationScenario:
        """
        Convert WorldState snapshot to SimulationScenario.
        """
        w = max(1.0, float(world_state.image_width))
        h = max(1.0, float(world_state.image_height))

        # Road condition parameters
        cat_name = world_state.road_condition.category.value
        score = world_state.road_condition.score
        speed_factor = float(self.speed_map.get(cat_name, world_state.road_condition.speed_restriction_factor))
        friction_factor = float(self.friction_map.get(cat_name, world_state.road_condition.estimated_friction_factor))

        # Map dynamic agents (vehicles & pedestrians)
        agents: List[SimulationAgent] = []
        all_detections = world_state.traffic.vehicles + world_state.traffic.pedestrians

        for det in all_detections:
            norm_box = det.bbox.to_normalized(int(w), int(h))
            agent = SimulationAgent(
                agent_id=det.object_id,
                class_name=det.class_name,
                track_id=det.track_id,
                image_bbox=det.bbox.as_tuple(),
                normalized_position=norm_box.bottom_center,
                road_association=det.road_association,
                estimated_distance=det.estimated_distance,
            )
            agents.append(agent)

        # Map road hazards (potholes)
        hazards: List[SimulationHazard] = []
        if self.potholes_as_hazards and world_state.potholes.potholes:
            for p in world_state.potholes.potholes:
                norm_box = p.bbox.to_normalized(int(w), int(h))
                hazard = SimulationHazard(
                    hazard_id=p.pothole_id,
                    hazard_type="pothole",
                    severity=p.estimated_severity.value,
                    image_bbox=p.bbox.as_tuple(),
                    normalized_position=norm_box.center,
                    relative_area=p.relative_area,
                    road_association=p.road_association,
                )
                hazards.append(hazard)

        scenario = SimulationScenario(
            frame_id=world_state.frame_id,
            timestamp=world_state.timestamp,
            road_condition_category=cat_name,
            road_condition_score=score,
            estimated_friction_factor=friction_factor,
            speed_restriction_factor=speed_factor,
            agents=agents,
            hazards=hazards,
            traffic_intensity=world_state.traffic.congestion_level.value,
            traffic_density=world_state.traffic.traffic_density,
            environment_meta={
                "image_width": world_state.image_width,
                "image_height": world_state.image_height,
                "total_perception_objects": world_state.traffic.total_objects,
                "perception_healthy": world_state.perception_status.is_all_ok(),
            },
        )
        return scenario

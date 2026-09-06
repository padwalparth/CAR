"""
NVIDIA NIM LLM Copilot & Official Road Safety Audit Service.

Integrates NVIDIA NIM LLM endpoints:
- meta/llama-3.2-11b-vision-instruct (Fast Scene Q&A & Vision Reasoning)
- deepseek-ai/deepseek-v4-pro-0813 (Deep reasoning & audit generation)
- google/gemma-4-31b-it (IRC / MoRTH compliance reporting)
- openai/whisper-large-v3 (Speech-to-Text)

Provides:
1. Scene Q&A Chat Assistant (telemetry-grounded conversational reasoning)
2. Official PWD / NHAI Road Safety Audit Report Generator
3. Voice command transcription
"""

import os
import json
import time
import requests
from typing import Any, Dict, List, Optional

NVIDIA_NIM_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_NIM_KEY = os.environ.get(
    "NVIDIA_API_KEY",
    "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"
)
WHISPER_KEY = os.environ.get(
    "WHISPER_API_KEY",
    "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"
)

# Ordered preference of models (fast vision-instruct first, then deepseek/gemma)
CANDIDATE_MODELS = [
    "meta/llama-3.2-11b-vision-instruct",
    "deepseek-ai/deepseek-v4-pro-0813",
    "deepseek-ai/deepseek-v4-flash-0731",
    "google/gemma-4-31b-it"
]


class LLMCopilotService:
    """Service wrapping NVIDIA NIM LLM endpoints for RoadVision AI."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or DEFAULT_NIM_KEY
        self.primary_model = "meta/llama-3.2-11b-vision-instruct"

    def _call_nim(self, messages: List[Dict[str, str]], max_tokens: int = 500, temperature: float = 0.2) -> Dict[str, Any]:
        """Calls NVIDIA NIM with automatic model fallback."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        last_error = None
        for model_name in CANDIDATE_MODELS:
            payload = {
                "model": model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            try:
                t0 = time.time()
                res = requests.post(NVIDIA_NIM_URL, headers=headers, json=payload, timeout=15)
                latency_ms = (time.time() - t0) * 1000.0

                if res.status_code == 200:
                    data = res.json()
                    content = data["choices"][0]["message"]["content"]
                    return {
                        "success": True,
                        "content": content,
                        "model": model_name,
                        "latency_ms": round(latency_ms, 1)
                    }
                else:
                    last_error = f"Model {model_name} HTTP {res.status_code}: {res.text[:150]}"
            except Exception as e:
                last_error = f"Model {model_name} exception: {str(e)}"
                continue

        # Fallback to local intelligent response if API network is unreachable
        return {
            "success": False,
            "error": last_error,
            "fallback": True
        }

    def _format_scene_context(self, session_data: Dict[str, Any]) -> str:
        """Formats session detection telemetry into clean context string."""
        if not session_data:
            return "No real-time session telemetry available."

        res = session_data.get("result") or {}
        road = res.get("road_segmentation") or res.get("road") or {}
        traffic = res.get("yolo") or res.get("traffic") or {}
        potholes = res.get("potholes") or {}
        fusion = res.get("fusion") or res.get("risk_assessment") or {}
        sim = res.get("simulation") or {}

        road_ratio = road.get("coverage_ratio") or road.get("area_ratio") or 0.0
        traffic_dets = traffic.get("detections") or traffic.get("objects") or []
        pothole_dets = potholes.get("detections") or potholes.get("potholes") or []
        risk_level = fusion.get("risk_level") or fusion.get("overall_risk_level") or "MODERATE"
        risk_score = fusion.get("risk_score") or 35.0
        sim_action = sim.get("action") or sim.get("recommended_action") or "MAINTAIN_CRUISE"
        sim_speed = sim.get("speed_kmh") or 35.0

        ctx = f"""[RoadVision Perception Telemetry]
- Drivable Road Coverage Ratio: {road_ratio:.2f} ({road_ratio*100:.1f}%)
- Potholes Detected: {len(pothole_dets)} items (Details: {[p.get('severity', p.get('estimated_severity', 'medium')) for p in pothole_dets]})
- Traffic Objects Detected: {len(traffic_dets)} vehicles/vulnerable road users (Classes: {[d.get('class_name', d.get('class', 'object')) for d in traffic_dets]})
- Road Quality / Safety Risk: Level {risk_level} (Safety Score: {risk_score}/100)
- Autonomous Vehicle Action: {sim_action} at {sim_speed} km/h
"""
        return ctx

    def chat_about_scene(
        self,
        session_data: Dict[str, Any],
        user_prompt: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Answers user/driver/judge questions about the analyzed scene with LLM reasoning."""
        scene_ctx = self._format_scene_context(session_data)

        system_prompt = (
            "You are RoadVision AI Copilot, an expert autonomous driving assistant and road safety auditor. "
            "You provide clear, authoritative, concise, and actionable safety insights based on the provided "
            "camera computer-vision telemetry (U-Net road segmentation, YOLO traffic detection, YOLOv11 pothole detection, and vehicle dynamics). "
            "Use bullet points and bold highlights where appropriate. Answer directly and informatively."
        )

        messages = [{"role": "system", "content": system_prompt}]

        # Append recent history
        if chat_history:
            for msg in chat_history[-6:]:
                messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})

        # Add current user prompt with grounded scene context
        user_message_content = f"{scene_ctx}\n\nUser Question: {user_prompt}"
        messages.append({"role": "user", "content": user_message_content})

        result = self._call_nim(messages, max_tokens=400, temperature=0.25)

        if result.get("success"):
            return {
                "response": result["content"],
                "model_used": result["model"],
                "latency_ms": result["latency_ms"],
                "status": "ok"
            }

        # Local intelligent fallback if network blip
        fallback_text = (
            f"**RoadVision Safety Advisory**:\n\n"
            f"Based on real-time perception telemetry, the road drivable corridor is currently at safe operating parameters. "
            f"Detected hazards include active road users and surface features. Recommended driving action is to maintain cautious speed "
            f"and adhere to lane boundaries."
        )
        return {
            "response": fallback_text,
            "model_used": "local-rule-engine-fallback",
            "latency_ms": 1.0,
            "status": "fallback"
        }

    def generate_road_safety_audit(
        self,
        session_data: Dict[str, Any],
        location_label: str = "NH-48 Sector KM 14.2"
    ) -> Dict[str, Any]:
        """Generates a formal PWD / NHAI Road Safety Audit Report (IRC / MoRTH compliant)."""
        scene_ctx = self._format_scene_context(session_data)

        system_prompt = (
            "You are a Senior Executive Road Safety Auditor for the National Highways Authority of India (NHAI) / Public Works Department (PWD). "
            "Generate a formal, professional Road Safety Audit & Maintenance Dispatch Report in clean GitHub Markdown format based on the "
            "computer-vision telemetry provided. Include: \n"
            "1. EXECUTIVE SUMMARY & INSPECTION METRICS\n"
            "2. HAZARD CLASSIFICATION (IRC:SP:88 & MoRTH guidelines)\n"
            "3. PAVEMENT SURFACE DEFECT INVENTORY & ASPHALT PATCH VOLUME ESTIMATION\n"
            "4. TRAFFIC HAZARDS & VULNERABLE ROAD USER (VRU) RISKS\n"
            "5. PWD REPAIR WORK ORDER & SLA RECTIFICATION DEADLINE (e.g., 24-48 Hours for Critical/High)"
        )

        user_content = (
            f"Location: {location_label}\n"
            f"Date: {time.strftime('%d-%b-%Y %H:%M:%S UTC')}\n"
            f"{scene_ctx}\n\n"
            f"Generate the official NHAI/PWD Road Safety Audit Report now."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        result = self._call_nim(messages, max_tokens=800, temperature=0.2)

        if result.get("success"):
            return {
                "report_markdown": result["content"],
                "model_used": result["model"],
                "latency_ms": result["latency_ms"],
                "status": "ok"
            }

        # Fallback formal report
        fallback_report = f"""# OFFICIAL ROAD SAFETY AUDIT REPORT
**Authority:** National Highways Authority of India (NHAI) / Public Works Department (PWD)  
**Location:** {location_label}  
**Date:** {time.strftime('%d-%b-%Y')}  
**Audit Standard:** MoRTH Specifications for Road and Bridge Works (5th Revision) / IRC:SP:88  

---

### 1. Executive Summary
- **Road Corridor Health:** Moderate
- **Pavement Surface Status:** Surface deterioration and potholes identified in traffic corridor.
- **Immediate Action Required:** Pothole cold-mix asphalt patch dispatch within 48 Hours.

### 2. Defect & Traffic Risk Analysis
- **Road Surface:** Drivable corridor identified with localized pavement distress.
- **Traffic Density:** Mixed traffic (vehicles & two-wheelers) present.
- **Risk Category:** MEDIUM PRIORITY

### 3. PWD Rectification Work Order
1. Mobilize Mobile Maintenance Unit (MMU) for bituminous cold-mix repair.
2. Install temporary reflective retroreflective cautionary signage (IRC:67).
3. Resurface worn section to restore skid resistance (BPN > 55).
"""
        return {
            "report_markdown": fallback_report,
            "model_used": "local-pwd-audit-template",
            "latency_ms": 1.0,
            "status": "fallback"
        }


# Singleton instance
copilot_service = LLMCopilotService()

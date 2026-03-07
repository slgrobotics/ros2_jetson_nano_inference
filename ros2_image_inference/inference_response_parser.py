from dataclasses import dataclass
from typing import List, Optional, Tuple
import json


@dataclass(frozen=True)
class Detection:
    class_id: int
    label: str
    confidence: float
    track_id: Optional[int]
    bbox_xyxy: Tuple[float, float, float, float]
    bbox_xywh: Tuple[float, float, float, float]


@dataclass(frozen=True)
class InferenceResult:
    ok: bool
    frame_id: int
    timestamp_ns: int
    server_received_ns: int
    server_infer_start_ns: int
    server_infer_end_ns: int
    queue_delay_ms: float
    infer_ms: float
    model_name: str
    model_path: str
    ultralytics_version: str
    imgsz: int
    detections: List[Detection]


def parse_inference_response(json_str: str) -> InferenceResult:
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid inference response JSON: {e}") from e

    try:
        detections = [
            Detection(
                class_id=int(d["class_id"]),
                label=str(d["label"]),
                confidence=float(d["confidence"]),
                track_id=None if d["track_id"] is None else int(d["track_id"]),
                bbox_xyxy=tuple(float(x) for x in d["bbox_xyxy"]),
                bbox_xywh=tuple(float(x) for x in d["bbox_xywh"]),
            )
            for d in data.get("detections", [])
        ]

        return InferenceResult(
            ok=bool(data["ok"]),
            frame_id=int(data["frame_id"]),
            timestamp_ns=int(data["timestamp_ns"]),
            server_received_ns=int(data["server_received_ns"]),
            server_infer_start_ns=int(data["server_infer_start_ns"]),
            server_infer_end_ns=int(data["server_infer_end_ns"]),
            queue_delay_ms=float(data["queue_delay_ms"]),
            infer_ms=float(data["infer_ms"]),
            model_name=str(data["model_name"]),
            model_path=str(data["model_path"]),
            ultralytics_version=str(data["ultralytics_version"]),
            imgsz=int(data["imgsz"]),
            detections=detections,
        )
    except KeyError as e:
        raise ValueError(f"Missing required field in inference response: {e}") from e
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid inference response field type: {e}") from e
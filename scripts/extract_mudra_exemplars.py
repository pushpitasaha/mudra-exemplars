#!/usr/bin/env python3
import argparse, json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import cv2, numpy as np
from scipy.spatial.transform import Rotation as R

import mediapipe as mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

HAND_CONNECTIONS = list(mp_hands.HAND_CONNECTIONS)
LANDMARK_NAMES = [
    "WRIST","THUMB_CMC","THUMB_MCP","THUMB_IP","THUMB_TIP",
    "INDEX_MCP","INDEX_PIP","INDEX_DIP","INDEX_TIP",
    "MIDDLE_MCP","MIDDLE_PIP","MIDDLE_DIP","MIDDLE_TIP",
    "RING_MCP","RING_PIP","RING_DIP","RING_TIP",
    "PINKY_MCP","PINKY_PIP","PINKY_DIP","PINKY_TIP"
]

@dataclass
class Transform:
    translation: List[float]
    rotation_matrix: List[List[float]]
    scale: float

def ensure_dirs(base: Path, label: str) -> Dict[str, Path]:
    label_dir = base / label
    subdirs = {
        "root": label_dir,
        "frames": label_dir / "frames",
        "obj_raw": label_dir / "obj_raw",
        "obj_norm": label_dir / "obj_norm",
        "obj_world": label_dir / "obj_world",
        "json": label_dir / "json",
    }
    for p in subdirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return subdirs

def landmarks_to_pixel_space(landmarks, width: int, height: int) -> np.ndarray:
    pts = []
    for lm in landmarks:
        x = lm.x * width
        y = lm.y * height
        z = lm.z * width  # pixel-ish depth scale
        pts.append([x, y, z])
    return np.asarray(pts, dtype=np.float32)

def compute_normalization_transform(pts_px: np.ndarray) -> Tuple[Transform, np.ndarray]:
    wrist = pts_px[0]; mid_mcp = pts_px[9]; idx_mcp = pts_px[5]
    translated = pts_px - wrist

    v_z = mid_mcp - wrist
    norm_vz = np.linalg.norm(v_z)
    if norm_vz < 1e-6:
        Rmat = np.eye(3, dtype=np.float32); scale = 1.0
        return Transform((-wrist).tolist(), Rmat.tolist(), float(scale)), translated

    z_axis = v_z / norm_vz
    v_x_raw = idx_mcp - wrist
    v_x_proj = v_x_raw - np.dot(v_x_raw, z_axis) * z_axis
    norm_vx = np.linalg.norm(v_x_proj)
    if norm_vx < 1e-6:
        v_x_proj = np.array([1.0, 0.0, 0.0], dtype=np.float32) - np.dot(np.array([1.0,0.0,0.0],dtype=np.float32), z_axis) * z_axis
        norm_vx = np.linalg.norm(v_x_proj)
    x_axis = v_x_proj / norm_vx
    y_axis = np.cross(z_axis, x_axis)
    Rmat = np.stack([x_axis, y_axis, z_axis], axis=1)
    scale = norm_vz
    normalized = (Rmat.T @ (translated.T / scale)).T
    return Transform((-wrist).tolist(), Rmat.tolist(), float(scale)), normalized

def write_obj(path: Path, verts: np.ndarray, edges=None) -> None:
    with open(path, "w") as f:
        f.write("# OBJ exported by Mudra Exemplar Extractor\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        if edges:
            for (i, j) in edges:
                f.write(f"l {i+1} {j+1}\n")

def write_json(path: Path, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

def draw_and_save_png(image_bgr, landmarks, handedness_str, out_path: Path) -> None:
    image = image_bgr.copy()
    mp_drawing.draw_landmarks(
        image, landmarks, mp_hands.HAND_CONNECTIONS,
        mp_styles.get_default_hand_landmarks_style(),
        mp_styles.get_default_hand_connections_style(),
    )
    cv2.putText(image, handedness_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), image)

def mediapipe_hand_detector():
    return mp_hands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=1,
                          min_detection_confidence=0.5, min_tracking_confidence=0.5)

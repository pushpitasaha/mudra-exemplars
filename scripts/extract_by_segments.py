import csv, json
from pathlib import Path
import cv2, numpy as np

# ---- utils & constants ----
from extract_mudra_exemplars import (
    mediapipe_hand_detector, landmarks_to_pixel_space,
    draw_and_save_png, write_obj, ensure_dirs,
    HAND_CONNECTIONS, LANDMARK_NAMES
)

def load_segments(csv_path, min_duration=1, padding=0):
    segs = []
    with open(csv_path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            s = int(row["start_frame"]); e = int(row["end_frame"])
            s += padding; e -= padding
            if e < s: continue
            if (e - s + 1) < min_duration: continue
            segs.append({"label": row["label"], "start_frame": s, "end_frame": e})
    return sorted(segs, key=lambda s: s["start_frame"])

def compute_normalization_transform(pts_px: np.ndarray, hand_label: str, canonicalize_right: bool):
    """Wrist @ origin; +Z along wrist->middle MCP; +X toward index MCP (projected).
       If hand_label=='Left' and canonicalize_right, flip X so normalized space looks like Right.
       Returns (transform_dict, normalized_pts, Rmat_used, scale_used, wrist_used)
    """
    wrist = pts_px[0]; mid_mcp = pts_px[9]; idx_mcp = pts_px[5]
    translated = pts_px - wrist

    v_z = mid_mcp - wrist
    norm_vz = np.linalg.norm(v_z)
    if norm_vz < 1e-6:
        Rmat = np.eye(3, dtype=np.float32); scale = 1.0
        normalized = translated.copy()
        return ({
            "translation": (-wrist).tolist(),
            "rotation_matrix": Rmat.tolist(),
            "scale": float(scale)
        }, normalized, Rmat, scale, wrist)

    z_axis = v_z / norm_vz
    v_x_raw = idx_mcp - wrist
    v_x_proj = v_x_raw - np.dot(v_x_raw, z_axis) * z_axis
    norm_vx = np.linalg.norm(v_x_proj)
    if norm_vx < 1e-6:
        v_x_proj = np.array([1.0,0.0,0.0], dtype=np.float32) - np.dot(np.array([1.0,0.0,0.0],dtype=np.float32), z_axis) * z_axis
        norm_vx = np.linalg.norm(v_x_proj)
    x_axis = v_x_proj / max(norm_vx, 1e-6)

    # If MediaPipe says this is LEFT and we want right-canonical, flip X axis
    if canonicalize_right and (hand_label.lower() == "left"):
        x_axis = -x_axis

    y_axis = np.cross(z_axis, x_axis)
    # Orthonormalize (just in case)
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-6)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-6)
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-6)

    Rmat = np.stack([x_axis, y_axis, z_axis], axis=1)   # columns are axes in camera space
    scale = norm_vz
    normalized = (Rmat.T @ (translated.T / scale)).T

    return ({
        "translation": (-wrist).tolist(),
        "rotation_matrix": Rmat.tolist(),
        "scale": float(scale)
    }, normalized, Rmat, scale, wrist)

def main(video, segments_csv, outdir, min_conf=0.6, stride=1, save_lines=True,
         min_duration=1, segment_padding=0, hand_mode="either", canonicalize_right=True, verbose=True):
    segs = load_segments(segments_csv, min_duration=min_duration, padding=segment_padding)
    if not segs:
        raise SystemExit("No segments to extract (check min_duration/padding).")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    label_dirs, manifests = {}, {}
    edges = [(a,b) for a,b in HAND_CONNECTIONS] if save_lines else None

    def get_dirs(label):
        if label not in label_dirs:
            d = ensure_dirs(Path(outdir), label)
            label_dirs[label] = d
            manifests[label] = {"video": str(video), "label": label, "fps": fps,
                                "frame_width": width, "frame_height": height, "records": []}
        return label_dirs[label]

    # debug counters
    stats = {s["label"]: {"attempt":0,"saved":0,"no_hand":0,"low_conf":0,"wrong_hand":0} for s in segs}

    frame_idx, seg_ptr, saved_total = 0, 0, 0
    with mediapipe_hand_detector() as hands:
        while True:
            ok, frame = cap.read()
            if not ok: break
            # advance segment ptr if we've passed it
            while seg_ptr < len(segs) and frame_idx > segs[seg_ptr]["end_frame"]:
                seg_ptr += 1

            active_label = None
            if seg_ptr < len(segs):
                s = segs[seg_ptr]
                if s["start_frame"] <= frame_idx <= s["end_frame"]:
                    active_label = s["label"]

            if active_label is None or (frame_idx % stride) != 0:
                frame_idx += 1
                continue

            stats[active_label]["attempt"] += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            if not results.multi_hand_landmarks:
                stats[active_label]["no_hand"] += 1
                frame_idx += 1
                continue

            # choose hand by mode
            chosen_idx, chosen_label, chosen_score = None, None, 0.0
            if results.multi_handedness:
                for i, hand_h in enumerate(results.multi_handedness):
                    lbl = hand_h.classification[0].label  # "Right" or "Left"
                    scr = float(hand_h.classification[0].score)
                    if hand_mode == "right":
                        if lbl.lower() == "right":
                            chosen_idx, chosen_label, chosen_score = i, lbl, scr
                            break
                    else:  # either → pick the highest score
                        if (chosen_idx is None) or (scr > chosen_score):
                            chosen_idx, chosen_label, chosen_score = i, lbl, scr

            if chosen_idx is None:
                stats[active_label]["no_hand"] += 1
                frame_idx += 1; continue

            if chosen_score < min_conf:
                stats[active_label]["low_conf"] += 1
                frame_idx += 1; continue

            if hand_mode == "right" and chosen_label.lower() != "right":
                stats[active_label]["wrong_hand"] += 1
                frame_idx += 1; continue

            # landmarks
            hand_landmarks = results.multi_hand_landmarks[chosen_idx]
            pts_px = landmarks_to_pixel_space(hand_landmarks.landmark, width, height)

            # world landmarks (meters)
            pts_world = None
            if results.multi_hand_world_landmarks:
                world = results.multi_hand_world_landmarks[chosen_idx]
                pts_world = np.asarray([[lm.x, lm.y, lm.z] for lm in world.landmark], dtype=np.float32)

            # normalization (canonicalize left→right if requested)
            transform, pts_norm, _, _, _ = compute_normalization_transform(
                pts_px, hand_label=chosen_label, canonicalize_right=canonicalize_right
            )

            # write outputs
            dirs = get_dirs(active_label)
            stem = f"frame_{frame_idx:06d}"
            png_path = dirs["frames"] / f"{stem}.png"
            obj_raw_path = dirs["obj_raw"] / f"{stem}.obj"
            obj_norm_path = dirs["obj_norm"] / f"{stem}_norm.obj"
            obj_world_path = dirs["obj_world"] / f"{stem}_world.obj"
            json_path = dirs["json"] / f"{stem}.json"

            draw_and_save_png(frame, hand_landmarks, f"{chosen_label.upper()} ({chosen_score:.2f})", png_path)
            write_obj(obj_raw_path, pts_px, edges)
            write_obj(obj_norm_path, pts_norm, edges)
            if pts_world is not None:
                write_obj(obj_world_path, pts_world, edges=None)

            record = {
                "frame_idx": frame_idx,
                "time_ms": float((frame_idx / fps) * 1000.0),
                "image_path": str(png_path),
                "obj_raw_path": str(obj_raw_path),
                "obj_norm_path": str(obj_norm_path),
                "obj_world_path": str(obj_world_path) if pts_world is not None else None,
                "json_path": str(json_path),
                "mudra_label": active_label,
                "hand_label": chosen_label,
                "handedness_score": chosen_score,
                "landmarks": {
                    "names": LANDMARK_NAMES,
                    "raw_pixel_xyzw": pts_px.tolist(),
                    "normalized_obj_space": pts_norm.tolist(),
                    "world_landmarks_m": pts_world.tolist() if pts_world is not None else None
                },
                "transform": transform
            }
            with open(json_path, "w") as f: json.dump(record, f, indent=2)

            manifests = manifests if 'manifests' in locals() else {}
            manifests.setdefault(active_label, {"video": str(video), "label": active_label, "fps": fps,
                                                "frame_width": width, "frame_height": height, "records": []})
            manifests[active_label]["records"].append({
                k: record[k] for k in ["frame_idx","time_ms","image_path","obj_raw_path","obj_norm_path","json_path","mudra_label","handedness_score"]
            })

            stats[active_label]["saved"] += 1
            saved_total += 1
            frame_idx += 1

    cap.release()

    # write manifests
    for label, man in manifests.items():
        with open(Path(outdir)/label/"manifest.json","w") as f: json.dump(man, f, indent=2)

    # summary
    print(f"[OK] saved {saved_total} frames into {outdir}")
    if verbose:
        print("Per-label summary (attempt/saved/no_hand/low_conf/wrong_hand):")
        for lbl, d in stats.items():
            print(f"  {lbl:14s}  {d['attempt']:5d}/{d['saved']:5d}  {d['no_hand']:5d}  {d['low_conf']:5d}  {d['wrong_hand']:5d}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--segments_csv", required=True)
    ap.add_argument("--outdir", default="data/exemplars")
    ap.add_argument("--min_conf", type=float, default=0.6)   # lower default
    ap.add_argument("--stride", type=int, default=1, help="sample every Nth frame")
    ap.add_argument("--no_lines", action="store_true")
    ap.add_argument("--min_duration", type=int, default=1)
    ap.add_argument("--segment_padding", type=int, default=0)
    ap.add_argument("--hand", choices=["right","either"], default="either", help="accept only Right, or whichever hand is detected")
    ap.add_argument("--no_canonicalize_right", action="store_true", help="do NOT flip Left to Right in normalized OBJ")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    main(args.video, args.segments_csv, args.outdir, args.min_conf, args.stride,
         save_lines=not args.no_lines, min_duration=args.min_duration, segment_padding=args.segment_padding,
         hand_mode=args.hand, canonicalize_right=not args.no_canonicalize_right, verbose=not args.quiet)

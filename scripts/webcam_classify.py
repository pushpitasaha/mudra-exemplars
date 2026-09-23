# scripts/webcam_classify.py
# Webcam → MediaPipe → normalize → 63D → nearest-mean (L1) → score via per-class (mu_d, sigma_d)
import time
from collections import deque
import numpy as np, cv2, mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

# ---------- normalization (same as extractor; Left -> Right canonical) ----------
def normalize_pts_px(pts_px: np.ndarray, hand_label: str, canonicalize_right: bool = True):
    wrist, mid_mcp, idx_mcp = pts_px[0], pts_px[9], pts_px[5]
    translated = pts_px - wrist
    v_z = mid_mcp - wrist; nz = np.linalg.norm(v_z)
    if nz < 1e-6:
        R = np.eye(3, dtype=np.float32); s = 1.0
        return (R, s), translated
    z = v_z / nz
    vx = idx_mcp - wrist
    vxp = vx - (vx @ z) * z
    nx = np.linalg.norm(vxp)
    if nx < 1e-6:
        vxp = np.array([1,0,0], np.float32) - (np.array([1,0,0],np.float32) @ z) * z
        nx = np.linalg.norm(vxp)
    x = vxp / nx
    if canonicalize_right and hand_label.lower() == "left":
        x = -x
    y = np.cross(z, x)
    x/=np.linalg.norm(x)+1e-6; y/=np.linalg.norm(y)+1e-6; z/=np.linalg.norm(z)+1e-6
    R = np.stack([x, y, z], axis=1); s = nz
    pts_norm = (R.T @ (translated.T / s)).T
    return (R, s), pts_norm

def landmarks_to_px(landmarks, w, h):
    return np.asarray([[lm.x*w, lm.y*h, lm.z*w] for lm in landmarks], dtype=np.float32)

# ---------- distance ----------
def l1(a, b):  # sum of absolute diffs
    return float(np.abs(a - b).sum())

# ---------- exemplar library ----------
class ExemplarLibrary:
    """Loads class means (63D) and distance distribution stats; exposes predict + scoring."""
    def __init__(self, npz_path: str):
        d = np.load(npz_path, allow_pickle=True)
        self.labels = d["label_names"].tolist()     # [C]
        self.means  = d["means"].astype(np.float32) # [C,63]
        self.mu_d   = d["mu_d"].astype(np.float32)  # [C]
        self.sigma  = d["sigma_d"].astype(np.float32)  # [C]
        self.global_sigma = float(d["global_sigma"])
        assert self.means.shape[0] == len(self.labels) == self.mu_d.shape[0] == self.sigma.shape[0]
        print(f"[OK] loaded {npz_path}: {len(self.labels)} classes")

    # classification: nearest mean by L1
    def predict_l1(self, x: np.ndarray):
        dists = np.abs(self.means - x[None, :]).sum(axis=1)      # [C]
        i = int(np.argmin(dists))
        top3_idx = np.argsort(dists)[:min(3, len(self.labels))]
        top3 = [(self.labels[j], float(dists[j])) for j in top3_idx]
        return i, self.labels[i], float(dists[i]), top3

    # scoring: map distance to z, qualitative band, 0..100 score
    def score(self, class_idx: int, raw_dist: float):
        mu  = float(self.mu_d[class_idx])
        sig = float(max(self.sigma[class_idx], self.global_sigma, 1e-6))
        # z >= 0 means how many std devs above the typical class distance
        z = max(0.0, (raw_dist - mu) / sig)

        if z <= 1.0:
            band, color, tips = "Good", (40,255,200), "Great alignment"
        elif z <= 2.0:
            band, color, tips = "Okay", (0,255,255), "Close; fine-tune fingers/wrist"
        else:
            band, color, tips = "Poor", (0,0,255), "Adjust pose-reposition fingers/wrist"

        # 0..100 — linear to 0 at 3σ (simple, monotonic, forgiving)
        score_pct = int(np.clip(100.0 * (1.0 - z/3.0), 0.0, 100.0))
        return z, band, color, score_pct, tips

# ---------- camera helper ----------
BACKENDS = {"msmf": cv2.CAP_MSMF, "dshow": cv2.CAP_DSHOW, "any": cv2.CAP_ANY}
def open_camera(index=0, backend="auto"):
    order = ["msmf","dshow","any"] if backend=="auto" else [backend]
    for name in order:
        cap = cv2.VideoCapture(index, BACKENDS[name])
        if cap.isOpened():
            print(f"[OK] camera opened via {name} on index {index}")
            return cap
    print("[ERR] could not open camera"); return None

# ---------- main ----------
def main(model="models/exemplars.npz", min_conf=0.6, smooth=7, camera=0, backend="auto"):
    lib = ExemplarLibrary(model)
    dq_pred, dq_pct = deque(maxlen=max(1,smooth)), deque(maxlen=max(1,smooth))

    cap = open_camera(camera, backend)
    if cap is None: return

    t0, n, fps = time.time(), 0, 0.0
    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=1,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while True:
            ok, frame = cap.read()
            if not ok: break
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            pred_name = "No hand"
            score_pct = 0
            top3 = []
            band = ""
            tips = ""
            color = (200,200,200)
            z = 0.0
            raw_l1 = 0.0
            
            if res.multi_hand_landmarks and res.multi_handedness:
                # highest-confidence hand
                bi, bs, bl = None, -1.0, ""
                for i, hnd in enumerate(res.multi_handedness):
                    s = float(hnd.classification[0].score)
                    if s > bs: bi, bs, bl = i, s, hnd.classification[0].label
                if bi is not None and bs >= min_conf:
                    pts_px = landmarks_to_px(res.multi_hand_landmarks[bi].landmark, w, h)
                    (_, _), pts_norm = normalize_pts_px(pts_px, bl, canonicalize_right=True)
                    x = pts_norm.flatten().astype(np.float32)

                    cls_idx, pred_name, raw_l1, top3 = lib.predict_l1(x)      # classify
                    z, band, color, score_pct, tips = lib.score(cls_idx, raw_l1)  # score

                    # draw hand
                    mp_draw.draw_landmarks(frame, res.multi_hand_landmarks[bi],
                                           mp_hands.HAND_CONNECTIONS,
                                           mp_style.get_default_hand_landmarks_style(),
                                           mp_style.get_default_hand_connections_style())

            dq_pred.append(pred_name); dq_pct.append(score_pct)
            pred_sm = max(set(dq_pred), key=dq_pred.count)
            pct_sm = int(round(sum(dq_pct)/len(dq_pct))) if dq_pct else 0

            n += 1
            if time.time() - t0 > 0.5:
                fps = n / (time.time()-t0); t0, n = time.time(), 0

            # HUD
            cv2.rectangle(frame, (0,0), (w, 124), (30,30,30), -1)
            cv2.putText(frame, f"Pred: {pred_sm}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40,255,200), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Score: {pct_sm}/100  |  Band: {band}  |  z={z:.2f}  |  L1={raw_l1:.3f}",
                        (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(
                frame,
                f"Feedback: {tips}",
                (12, 86),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA
            )
            cv2.putText(frame, f"FPS: {fps:.1f}", (w-140, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2, cv2.LINE_AA)

            y = 130
            for lab, dval in top3:
                cv2.putText(frame, f"{lab}: L1={dval:.3f}", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2, cv2.LINE_AA)
                y += 26

            cv2.imshow("Mudra Webcam Classifier (q=quit)", frame)
            if (cv2.waitKey(1) & 0xFF) in (ord('q'), ord('Q'), 27): break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/exemplars.npz")
    ap.add_argument("--min_conf", type=float, default=0.6)
    ap.add_argument("--smooth", type=int, default=7)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--backend", choices=["auto","msmf","dshow","any"], default="auto")
    args = ap.parse_args()
    main(args.model, args.min_conf, args.smooth, args.camera, args.backend)

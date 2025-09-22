import cv2, json, csv, yaml
import numpy as np
from pathlib import Path

# ---------- helpers ----------

def load_mudras(cfg_path):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    mudras = cfg["mudras"]
    if not mudras:
        raise SystemExit("config/mudras.yaml has empty 'mudras' list.")
    return mudras

def clamp(n, lo, hi):
    return max(lo, min(hi, n))

def resize_and_letterbox(img, max_w: int, max_h: int):
    """Resize img to fit (max_w,max_h) keeping aspect, center on black canvas."""
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    canvas = np.zeros((max_h, max_w, 3), dtype=np.uint8)
    x = (max_w - new_w) // 2
    y = (max_h - new_h) // 2
    canvas[y:y+new_h, x:x+new_w] = resized
    return canvas

def put_text_outline(img, text, org, scale=1.0, color=(255,255,255), thick=2):
    # ASCII-safe, with black outline for readability
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thick+2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def right_aligned_x(w, margin_px, text, scale, thick):
    size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x = w - margin_px - size[0]
    return max(0, x)

def draw_hud(frame, frame_idx, fps, mudras, sel_idx, active_start, active_label, last_segments, ui_scale=1.0):
    # colors
    COLOR_TEXT = (255,255,255)
    COLOR_SUB  = (220,220,220)
    COLOR_SEL  = (40,255,200)
    COLOR_PREV = (180,200,255)
    COLOR_NEXT = (180,255,180)
    COLOR_SEQ  = (255,255,0)
    COLOR_SEQ_DIM = (200,200,160)
    COLOR_REC_BG = (0,0,80)
    COLOR_REC_DOT= (0,0,255)
    COLOR_FOOT  = (200,200,200)

    h, w = frame.shape[:2]
    overlay = frame.copy()

    # ---------- top bar ----------
    bar_h = int(72 * ui_scale)
    cv2.rectangle(overlay, (0,0), (w, bar_h), (30,30,30), -1)
    put_text_outline(overlay, f"Frame: {frame_idx}",
                     (int(12*ui_scale), int(28*ui_scale)),
                     0.9*ui_scale, COLOR_TEXT, max(2, int(2*ui_scale)))
    put_text_outline(overlay, f"FPS: {fps:.2f}",
                     (int(12*ui_scale), int(52*ui_scale)),
                     0.8*ui_scale, COLOR_SUB, max(2, int(2*ui_scale)))

    # selected banner (center)
    selected = mudras[sel_idx]
    banner = f"SELECTED: {selected}"
    (tw, _), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 1.2*ui_scale, max(2, int(2*ui_scale)))
    cx = max(int(10*ui_scale), (w - tw)//2)
    put_text_outline(overlay, banner, (cx, int(50*ui_scale)),
                     1.2*ui_scale, COLOR_SEL, max(2, int(2*ui_scale)))

    # prev / next (right-aligned)
    prev_name = mudras[sel_idx-1] if sel_idx-1 >= 0 else "(none)"
    next_name = mudras[sel_idx+1] if sel_idx+1 < len(mudras) else "(none)"
    s_prev = 0.8*ui_scale; t_prev = max(2, int(2*ui_scale))
    s_next = 0.8*ui_scale; t_next = max(2, int(2*ui_scale))
    margin = int(12*ui_scale)
    txt_prev = f"Prev: {prev_name}"
    txt_next = f"Next: {next_name}"
    x_prev = right_aligned_x(w, margin, txt_prev, s_prev, t_prev)
    x_next = right_aligned_x(w, margin, txt_next, s_next, t_next)
    put_text_outline(overlay, txt_prev, (x_prev, int(28*ui_scale)),
                     s_prev, COLOR_PREV, t_prev)
    put_text_outline(overlay, txt_next, (x_next, int(52*ui_scale)),
                     s_next, COLOR_NEXT, t_next)

    # ---------- recording strip ----------
    if active_start is not None and active_label is not None:
        rec_h = int(40 * ui_scale)
        y0 = bar_h + int(6*ui_scale)
        cv2.rectangle(overlay, (0, y0), (w, y0 + rec_h), COLOR_REC_BG, -1)
        cv2.circle(overlay, (int(18*ui_scale), y0 + rec_h//2), int(8*ui_scale), COLOR_REC_DOT, -1)
        dur_frames = max(0, frame_idx - active_start)
        dur_ms = int((dur_frames / max(fps, 1e-6)) * 1000)
        rec_text = f"REC {active_label}  start={active_start}  len={dur_frames}f ~{dur_ms}ms"
        put_text_outline(overlay, rec_text,
                         (int(36*ui_scale), y0 + rec_h//2 + int(8*ui_scale)),
                         0.9*ui_scale, COLOR_TEXT, max(2, int(2*ui_scale)))
        msg = f"Saving as: {active_label}"
        x_msg = right_aligned_x(w, int(12*ui_scale), msg, 0.8*ui_scale, max(2, int(2*ui_scale)))
        put_text_outline(overlay, msg,
                         (x_msg, y0 + rec_h//2 + int(8*ui_scale)),
                         0.8*ui_scale, (0,255,255), max(2, int(2*ui_scale)))

    # ---------- left sequence list ----------
    x0, y0 = int(10*ui_scale), bar_h + int(56*ui_scale)
    put_text_outline(overlay, "Sequence:", (x0, y0),
                     0.9*ui_scale, COLOR_SEQ, max(2, int(2*ui_scale)))
    y = y0 + int(12*ui_scale)
    max_show = min(len(mudras), 18)
    start_idx = max(0, min(sel_idx - 6, max(0, len(mudras) - max_show)))
    end_idx = min(start_idx + max_show, len(mudras))
    for i in range(start_idx, end_idx):
        y += int(26*ui_scale)
        marker = ">" if i == sel_idx else "  "     # ASCII marker
        color = COLOR_SEQ if i == sel_idx else COLOR_SEQ_DIM
        put_text_outline(overlay, f"{marker} {i+1:02d}. {mudras[i]}",
                         (x0+int(8*ui_scale), y), 0.85*ui_scale, color, max(2, int(2*ui_scale)))

    # ---------- right saved-segments tail ----------
    x1 = w - int(440*ui_scale)
    y1 = bar_h + int(56*ui_scale)
    put_text_outline(overlay, "Saved segments (tail):", (x1, y1),
                     0.9*ui_scale, (180,180,255), max(2, int(2*ui_scale)))
    y = y1 + int(12*ui_scale)
    for s in last_segments[-7:]:
        y += int(24*ui_scale)
        put_text_outline(overlay, f"{s['label']} [{s['start_frame']}..{s['end_frame']}]",
                         (x1+int(8*ui_scale), y), 0.8*ui_scale, (180,180,255), max(2, int(2*ui_scale)))

    # ---------- footer ----------
    footer = "SPACE pause | , and . = prev/next | [ and ] = jump +/- 5 | HOME/END | S start | E end | b/f step (paused) | R cancel | U undo | Q finish"
    cv2.rectangle(overlay, (0, h - int(32*ui_scale)), (w, h), (25,25,25), -1)
    put_text_outline(overlay, footer, (int(10*ui_scale), h - int(8*ui_scale)),
                     0.7*ui_scale, (200,200,200), max(2, int(2*ui_scale)))

    return overlay

# ---------- main ----------

def main(args):
    mudras = load_mudras(args.config)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    seg_csv = out_dir / "segments.csv"
    seg_json = out_dir / "segments.json"

    abs_path = str(Path(args.video).resolve())
    cap = cv2.VideoCapture(abs_path)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(abs_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {abs_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = 0
    paused = False

    sel_idx = 0
    active_start = None
    active_label = None
    segments = []

    cv2.namedWindow("label_video_segments", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("label_video_segments", args.fit_w, args.fit_h)

    HOME_KEY = 2359296
    END_KEY  = 2293760

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

        if 'frame' in locals() and frame is not None:
            # keep HUD readable after downscaling for display
            h, w = frame.shape[:2]
            display_scale = min(args.fit_w / w, args.fit_h / h, 1.0)
            ui_scale = args.hud_scale * (1.0 / max(display_scale, 1e-6))
            overlay = draw_hud(frame, frame_idx, fps, mudras, sel_idx,
                               active_start, active_label, segments, ui_scale=ui_scale)
            disp = resize_and_letterbox(overlay, args.fit_w, args.fit_h)
            cv2.imshow("label_video_segments", disp)

        key = cv2.waitKey(1 if not paused else 30) & 0xFFFFFFFF

        if key == ord(' '):                      # pause/play
            paused = not paused

        elif key == ord(','):                    # prev mudra
            sel_idx = (sel_idx - 1) % len(mudras)

        elif key == ord('.'):                    # next mudra
            sel_idx = (sel_idx + 1) % len(mudras)

        elif key == ord('['):                    # jump -5
            sel_idx = max(0, sel_idx - 5)

        elif key == ord(']'):                    # jump +5
            sel_idx = min(len(mudras)-1, sel_idx + 5)

        elif key == HOME_KEY:                    # first
            sel_idx = 0

        elif key == END_KEY:                     # last
            sel_idx = len(mudras)-1

        elif key in (ord('s'), ord('S')):        # START
            if active_start is None:
                active_start = frame_idx
                active_label = mudras[sel_idx]
                print(f"START {active_label} @ frame {active_start}")
            else:
                print("[WARN] Already recording. Press E to end or R to cancel.")

        elif key in (ord('e'), ord('E')):        # END
            if active_start is None:
                print("[WARN] No active segment. Press S to start.")
            else:
                end_idx = max(frame_idx - 1, active_start)
                segments.append({"label": active_label, "start_frame": active_start, "end_frame": end_idx})
                print(f"END   {active_label} @ frame {end_idx} (saved)")
                active_start, active_label = None, None

        elif key == ord('b'):                    # step back ONE FRAME
            if paused:
                pos = max(frame_idx - 1, 0)
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ok, frame = cap.read()
                if ok:
                    frame_idx = pos
            else:
                print("[HINT] Pause first (SPACE), then 'b' steps back one frame.")

        elif key == ord('f'):                    # step forward ONE FRAME
            if paused:
                ok, frame = cap.read()
                if ok:
                    frame_idx += 1
            else:
                print("[HINT] Pause first (SPACE), then 'f' steps forward one frame.")

        elif key in (ord('r'), ord('R')):        # cancel current (in-progress) segment
            if active_start is not None:
                print(f"[CANCEL] active segment from {active_start} discarded")
            active_start, active_label = None, None

        elif key in (ord('u'), ord('U')):        # undo last saved segment
            if segments:
                removed = segments.pop()
                print(f"[UNDO] removed: {removed['label']} [{removed['start_frame']}..{removed['end_frame']}]")
            else:
                print("[WARN] No segments to undo.")

        elif key in (ord('q'), ord('Q')):        # finish
            break

    cap.release()
    cv2.destroyAllWindows()

    # save labels
    rows = []
    for s in segments:
        start_ms = int((s["start_frame"]/fps)*1000.0)
        end_ms = int((s["end_frame"]/fps)*1000.0)
        rows.append([s["label"], s["start_frame"], s["end_frame"], start_ms, end_ms])

    with open(seg_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label","start_frame","end_frame","start_ms","end_ms"])
        w.writerows(rows)

    with open(seg_json, "w") as f:
        json.dump({
            "video": abs_path,
            "fps": fps,
            "segments": segments,
            "mode": "label_video_segments (start-end HUD)",
            "order": mudras
        }, f, indent=2)

    print(f"[OK] wrote: {seg_csv}\n[OK] wrote: {seg_json}")

# ---------- entry ----------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="label_video_segments: start/end labeling with big HUD (ASCII only).")
    ap.add_argument("--video", required=True, help="Path to input video")
    ap.add_argument("--config", default="config/mudras.yaml")
    ap.add_argument("--out_dir", default="labels/run1")
    ap.add_argument("--fit_w", type=int, default=1280, help="display window width")
    ap.add_argument("--fit_h", type=int, default=720,  help="display window height")
    ap.add_argument("--hud_scale", type=float, default=1.8, help="HUD magnification")
    args = ap.parse_args()
    main(args)

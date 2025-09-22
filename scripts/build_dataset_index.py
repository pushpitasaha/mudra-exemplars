import json, csv
from pathlib import Path

def collect_manifests(root="data/exemplars"):
    root = Path(root)
    for label_dir in root.iterdir():
        if not label_dir.is_dir(): continue
        man = label_dir / "manifest.json"
        if man.exists():
            yield label_dir.name, man

def main(root="data/exemplars", out_csv="data/index.csv"):
    rows = []
    for label, man_path in collect_manifests(root):
        with open(man_path, "r") as f: man = json.load(f)
        for r in man.get("records", []):
            rows.append({
                "mudra_label": label,
                "video": man.get("video",""),
                "fps": man.get("fps", 0),
                **{k: r[k] for k in ["frame_idx","time_ms","image_path","obj_raw_path","obj_norm_path","json_path","handedness_score"]}
            })
    out = Path(out_csv); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        fields = ["mudra_label","video","fps","frame_idx","time_ms","image_path","obj_raw_path","obj_norm_path","json_path","handedness_score"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"[OK] wrote {out} with {len(rows)} rows")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/exemplars")
    ap.add_argument("--out_csv", default="data/index.csv")
    args = ap.parse_args()
    main(args.root, args.out_csv)

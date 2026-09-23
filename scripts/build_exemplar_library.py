# scripts/build_exemplar_library.py
# Builds per-class mean (63D) + distance distribution stats (mu_d, sigma_d) for scoring.
import re, csv
import numpy as np
from pathlib import Path

OBJ_V_RE = re.compile(r"^v\s+([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)")

def read_obj_vertices(path: Path):
    verts = []
    with open(path, "r") as f:
        for line in f:
            m = OBJ_V_RE.match(line.strip())
            if m: verts.append([float(m[1]), float(m[2]), float(m[3])])
    verts = np.asarray(verts, dtype=np.float32)
    return verts if verts.shape == (21, 3) else None  # expect 21 landmarks

def l1(a, b):  # sum of absolute diffs
    return float(np.abs(a - b).sum())

def main(root="data/exemplars", out_npz="models/exemplars.npz", summary_csv="models/summary.csv", max_per_label=0):
    root = Path(root)
    Path(out_npz).parent.mkdir(parents=True, exist_ok=True)

    X, y, labels, files = [], [], [], []
    for d in sorted(root.iterdir()):
        if not d.is_dir(): continue
        lab = d.name
        objs = sorted((d / "obj_norm").glob("*_norm.obj"))
        if not objs: continue
        if lab not in labels: labels.append(lab)
        li = labels.index(lab)
        cnt = 0
        for p in objs:
            v = read_obj_vertices(p)
            if v is None: continue
            X.append(v.flatten())  # 63D
            y.append(li)
            files.append(str(p))
            cnt += 1
            if max_per_label and cnt >= max_per_label:
                break

    if not X:
        raise SystemExit("No obj_norm files found.")
    X = np.vstack(X).astype(np.float32)  # [N,63]
    y = np.asarray(y, np.int32)
    labels = np.array(labels)

    means, mu_d, sigma_d = [], [], []
    for i in range(len(labels)):
        Xi = X[y == i]
        mu_i = Xi.mean(axis=0)                   # class mean pose (63D)
        means.append(mu_i)
        dists = np.abs(Xi - mu_i[None, :]).sum(axis=1)  # L1 distance to class mean
        md = float(dists.mean())                 # average distance (distribution center)
        sd = float(dists.std(ddof=1)) if len(dists) > 1 else 0.0  # std dev
        # robust fallbacks so sigma never collapses to 0
        if sd <= 1e-6:
            sd = max(0.1 * md, 0.05)            # small but non-zero spread
        mu_d.append(md)
        sigma_d.append(sd)

    means   = np.vstack(means).astype(np.float32)       # [C,63]
    mu_d    = np.asarray(mu_d, dtype=np.float32)        # [C]
    sigma_d = np.asarray(sigma_d, dtype=np.float32)     # [C]
    global_sigma = float(np.median(sigma_d)) if sigma_d.size else 0.1

    np.savez(out_npz,
             label_names=labels,
             means=means,          # [C,63] class centroids
             mu_d=mu_d,            # [C]   mean(L1 distance to centroid)
             sigma_d=sigma_d,      # [C]   std(L1 distance)
             global_sigma=global_sigma,
             X=X, y=y, files=np.array(files))   # kept for debugging
    print(f"[OK] wrote {out_npz}: {X.shape[0]} samples, {len(labels)} labels; median sigma={global_sigma:.3f}")

    # Tiny summary for sanity
    Path(summary_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["label","num_samples","mean_L1","std_L1"])
        for i, name in enumerate(labels):
            w.writerow([name, int((y==i).sum()), float(mu_d[i]), float(sigma_d[i])])

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/exemplars")
    ap.add_argument("--out_npz", default="models/exemplars.npz")
    ap.add_argument("--summary_csv", default="models/summary.csv")
    ap.add_argument("--max_per_label", type=int, default=0)
    args = ap.parse_args()
    main(args.root, args.out_npz, args.summary_csv, args.max_per_label)

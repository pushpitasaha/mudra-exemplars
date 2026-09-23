import json, numpy as np, os
os.makedirs("web/model", exist_ok=True)
d = np.load("models/exemplars.npz", allow_pickle=True)
out = {
  "labels": d["label_names"].tolist(),
  "means":  d["means"].tolist(),      # [C, 63]
  "mu_d":   d["mu_d"].tolist(),       # [C]
  "sigma_d":d["sigma_d"].tolist(),    # [C]
  "version": "v1"
}
with open("web/model/model.json","w") as f:
    json.dump(out, f)
print("[OK] wrote web/model/model.json")

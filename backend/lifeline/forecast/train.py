"""Train the CVAE and the baselines, then score all three on held-out 2026 hours.

Run:  python -m lifeline.forecast.train
Writes data/artifacts/{cvae.pt, cvae_stats.json, poisson_glm.pkl, forecast_eval.json}
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import torch

from lifeline.config import ARTIFACTS, HORIZON, TEST_START
from lifeline.forecast.baselines import PoissonGLM, SeasonalNaive
from lifeline.forecast.cvae import train_cvae
from lifeline.forecast.dataset import load_series, splits
from lifeline.forecast.metrics import crps, energy_score, interval_hit, pit

N_EVAL = 1500
N_SAMPLES = 200


def main(eval_only: bool = False) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    s = load_series()
    sp = splits(s)
    (Xtr, Ctr, _), (Xva, Cva, _), (Xte, Cte, Ite) = sp["train"], sp["val"], sp["test"]
    print(f"train {len(Xtr):,}  val {len(Xva):,}  test {len(Xte):,}  zones {Xtr.shape[2]}  cond {Ctr.shape[1]}")

    if eval_only:
        from lifeline.forecast.cvae import CVAE
        model = CVAE(Xtr.shape[1] * Xtr.shape[2], Ctr.shape[1])
        model.load_state_dict(torch.load(ARTIFACTS / "cvae.pt", weights_only=True))
        model.eval()
    else:
        print("training CVAE ...")
        model = train_cvae(Xtr, Ctr, Xva, Cva)
        torch.save(model.state_dict(), ARTIFACTS / "cvae.pt")

    mu_tr = model.posterior_mean(Xtr.reshape(len(Xtr), -1), Ctr)
    cov = np.cov(mu_tr.T)
    prec = np.linalg.inv(cov + 1e-6 * np.eye(cov.shape[0]))
    d = mu_tr - mu_tr.mean(0)
    maha = np.sqrt(np.einsum("ij,jk,ik->i", d, prec, d))
    c_names = (["hour_sin", "hour_cos", "weekend", "holiday"] + [f"dow{i}" for i in range(7)]
               + ["precip_mean", "precip_max", "temp"] + [f"lag_{z}" for z in s.zones] + ["lag_total"])
    stats = {"x_dim": int(Xtr.shape[1] * Xtr.shape[2]), "c_dim": int(Ctr.shape[1]), "zones": s.zones,
             "latent_mean": mu_tr.mean(0).tolist(), "latent_cov": cov.tolist(),
             "maha_p99": float(np.percentile(maha, 99)),
             "c_min": Ctr.min(0).tolist(), "c_max": Ctr.max(0).tolist(), "c_names": c_names,
             "train_latent_sample": mu_tr[np.random.default_rng(0).choice(len(mu_tr), 600, replace=False)].tolist()}
    (ARTIFACTS / "cvae_stats.json").write_text(json.dumps(stats))

    if eval_only and (ARTIFACTS / "poisson_glm.pkl").exists():
        with open(ARTIFACTS / "poisson_glm.pkl", "rb") as f:
            glm = pickle.load(f)
    else:
        print("fitting Poisson GLM ...")
        glm = PoissonGLM().fit(Xtr[::3], Ctr[::3])
        with open(ARTIFACTS / "poisson_glm.pkl", "wb") as f:
            pickle.dump(glm, f)
    naive = SeasonalNaive(s.Y)

    print("evaluating on held-out 2026 hours ...")
    rng = np.random.default_rng(42)
    pick = rng.choice(len(Xte), min(N_EVAL, len(Xte)), replace=False)
    pos_of = {ts: i for i, ts in enumerate(s.index)}
    kinds = ("cvae", "cvae_no_latent", "poisson", "naive")
    res = {k: {"crps": [], "es": [], "crps_total": [], "hit90": [], "pit": []} for k in kinds}
    for i in pick:
        y = Xte[i].reshape(-1)
        p = pos_of[Ite[i]]
        samples = {
            "cvae": model.sample(Cte[i], N_SAMPLES, rng)[0],
            # ablation: z fixed at the prior mean -> cells are independent negative binomials
            "cvae_no_latent": model.sample(Cte[i], N_SAMPLES, rng, z_scale=0.0)[0],
            "poisson": glm.sample(Cte[i], N_SAMPLES, rng),
            "naive": naive.sample_at(p, HORIZON, N_SAMPLES, rng),
        }
        for k, smp in samples.items():
            tot = smp.sum(1)
            res[k]["crps"].append(crps(smp, y))
            res[k]["es"].append(energy_score(smp, y))
            res[k]["crps_total"].append(crps(tot[:, None], np.array([y.sum()])))
            res[k]["hit90"].append(interval_hit(tot, y.sum()))
            res[k]["pit"].append(pit(tot, y.sum(), rng))
    out = {}
    for k, r in res.items():
        out[k] = {"crps_cell": float(np.mean(r["crps"])), "energy_score": float(np.mean(r["es"])),
                  "crps_total": float(np.mean(r["crps_total"])), "coverage90_total": float(np.mean(r["hit90"])),
                  "pit_hist": np.histogram(r["pit"], bins=10, range=(0, 1))[0].tolist(),
                  # paired bootstrap CI of energy score difference vs CVAE
                  }
    for k in ("cvae_no_latent", "poisson", "naive"):
        for metric in ("es", "crps_total"):
            diff = np.array(res[k][metric]) - np.array(res["cvae"][metric])
            boots = [rng.choice(diff, len(diff)).mean() for _ in range(2000)]
            out[k][f"{metric}_minus_cvae_ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    out["meta"] = {"n_eval_hours": int(len(pick)), "samples": N_SAMPLES, "test_from": TEST_START}
    (ARTIFACTS / "forecast_eval.json").write_text(json.dumps(out, indent=1))
    for k in kinds:
        o = out[k]
        print(f"  {k:8s} CRPS/cell {o['crps_cell']:.4f}  energy {o['energy_score']:.3f}  "
              f"CRPS total {o['crps_total']:.3f}  90% cover {o['coverage90_total']:.2f}")


if __name__ == "__main__":
    import sys
    main(eval_only="--eval-only" in sys.argv)

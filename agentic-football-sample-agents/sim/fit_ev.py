"""Fit the EV / Markov model with a proper train / test split."""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "lib"))
LIB = HERE.parent / "lib"

OUR_TUNING = {"ev_select": True, "def_search": True, "gk_ev": True,
              "ev_support": True, "ev_support_roles": "FWD1,FWD2"}
_CACHE = {}


def _team(name, our_model):
    from teams import AGG, TEAMS, repo_team
    key = (name, our_model)
    if key not in _CACHE:
        if name == "OURS":
            _CACHE[key] = repo_team(AGG, True, {"global": dict(OUR_TUNING, ev_model=our_model)})
        else:
            _CACHE[key] = TEAMS[name]()
    return _CACHE[key]


def _play(job):
    from engine import Match
    a, b, seed, our_model = job
    log = {"shot": [], "pass": [], "keep": [], "xt": []}
    Match(_team(a, our_model), _team(b, our_model), duration=240, seed=seed, log=log).run()
    for k in ("shot", "pass"):
        log[k] = [r for r in log[k] if r["y"] is not None]
    return log


def collect(teams, n, seed0, our_model, jobs):
    pairs = list(itertools.combinations(teams, 2))
    work = [(*pairs[i % len(pairs)], seed0 + i, our_model) for i in range(n)]
    data = {"shot": [], "pass": [], "keep": [], "xt": []}
    with ProcessPoolExecutor(jobs) as ex:
        for log in ex.map(_play, work, chunksize=2):
            for k in data:
                data[k].extend(log[k])
    return data


def _feats(row, fv, kind):
    """v3 = v2 for shot/keep, and only [1, lane, margin] for passes."""
    if fv == 3:
        x = row["v2"]
        return [x[0], x[2], x[4]] if kind == "pass" else x
    return row[f"v{fv}"]


def logistic(X, y, l2=1e-3, iters=4000, lr=0.3):
    """Logistic regression on standardized features; weights returned raw."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    mu, sd = X[:, 1:].mean(0), X[:, 1:].std(0) + 1e-9
    Z = np.c_[np.ones(len(X)), (X[:, 1:] - mu) / sd]
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))
        w -= lr * (Z.T @ (p - y) / len(y) + l2 * np.r_[0, w[1:]])
    return np.r_[w[0] - np.sum(w[1:] * mu / sd), w[1:] / sd]


def logloss(w, X, y):
    X, y = np.asarray(X, float), np.asarray(y, float)
    p = np.clip(1 / (1 + np.exp(-np.clip(X @ w, -30, 30))), 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def xt(rows, n_cells, iters=60, prior=25.0):
    tot, shots, goals = np.zeros(n_cells), np.zeros(n_cells), np.zeros(n_cells)
    T = np.zeros((n_cells, n_cells))
    for c, sh, g, nxt in rows:
        tot[c] += 1
        if sh:
            shots[c] += 1
            goals[c] += g
        elif nxt is not None:
            T[c, nxt] += 1
    s = shots / (tot + prior)
    gr = (goals + 0.5) / (shots + 5)
    m = T.sum(1) / (tot + prior)
    Tn = T / np.maximum(T.sum(1, keepdims=True), 1)
    V = np.zeros(n_cells)
    for _ in range(iters):
        V = s * gr + m * (Tn @ V)
    return V


def xt_logloss(V, rows):
    """How well V ranks 'this possession tick ends in a goal' on held-out data
    (coarse check that zone values generalize)."""
    y = np.array([g for _, sh, g, _ in rows], float)
    p = np.clip(np.array([V[c] for c, *_ in rows]), 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def main():
    import ev_policy as evp
    from teams import TEST_RIVALS, TEST_SEEDS, TRAIN_RIVALS, TRAIN_SEEDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-matches", type=int, default=600)
    ap.add_argument("--test-matches", type=int, default=240)
    ap.add_argument("--our-model", default="ev_model.json",
                    help="model our team plays with while generating data (self-play)")
    ap.add_argument("--fv", type=int, choices=[1, 2, 3], default=3)
    ap.add_argument("--out", default="ev_model.json")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    args = ap.parse_args()

    import gzip
    cache = HERE / "data" / f"data-{args.our_model.replace('.json', '')}-{args.train_matches}-{args.test_matches}.json.gz"
    if cache.exists():
        train, test = json.loads(gzip.decompress(cache.read_bytes()))
    else:
        train = collect(["OURS"] + TRAIN_RIVALS, args.train_matches, TRAIN_SEEDS + 50_000,
                        args.our_model, args.jobs)
        test = collect(["OURS"] + TEST_RIVALS, args.test_matches, TEST_SEEDS + 50_000,
                       args.our_model, args.jobs)
        cache.parent.mkdir(exist_ok=True)
        cache.write_bytes(gzip.compress(json.dumps([train, test]).encode()))

    model, report = {"fv": args.fv}, {}
    for k in ("shot", "pass", "keep"):
        report[k] = {"n_train": len(train[k]), "n_test": len(test[k])}
        ytr = np.array([r["y"] for r in train[k]], float)
        yte = np.array([r["y"] for r in test[k]], float)
        base = ytr.mean()
        report[k]["baseline_test"] = round(float(-np.mean(yte * np.log(base) +
                                                          (1 - yte) * np.log(1 - base))), 4)
        for fv in (1, 2, 3):
            Xtr = [_feats(r, fv, k) for r in train[k]]
            Xte = [_feats(r, fv, k) for r in test[k]]
            w = logistic(Xtr, ytr, l2=0.1 if (fv == 3 and k == "pass") else 1e-3)
            report[k][f"v{fv}_train"] = round(logloss(w, Xtr, ytr), 4)
            report[k][f"v{fv}_test"] = round(logloss(w, Xte, yte), 4)
            if fv == args.fv:
                model[k] = [round(float(v), 5) for v in w]
    n_cells = evp.GRID_X * evp.GRID_Y
    V = xt(train["xt"], n_cells)
    report["xt"] = {"train": round(xt_logloss(V, train["xt"]), 4),
                    "test": round(xt_logloss(V, test["xt"]), 4)}
    model["xt"] = [round(float(v), 4) for v in V]
    model["meta"] = {"source": "simulator", "our_model": args.our_model, "fit": report,
                     "train_matches": args.train_matches, "test_matches": args.test_matches}
    (LIB / args.out).write_text(json.dumps(model, indent=1), encoding="utf-8")

    print("Log-loss (menos es mejor). 'baseline' = predecir siempre la media de train.\n")
    print("| Modelo | n train | n test | baseline test | v1 train | v1 test | v2 train | v2 test | v3 train | v3 test |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for k, nm in (("shot", "tiro → gol"), ("pass", "pase → llega"), ("keep", "conservar")):
        r = report[k]
        print(f"| {nm} | {r['n_train']} | {r['n_test']} | {r['baseline_test']} | {r['v1_train']} | "
              f"{r['v1_test']} | {r['v2_train']} | {r['v2_test']} | {r['v3_train']} | {r['v3_test']} |")
    print(f"\nValores de zona V (Markov): train {report['xt']['train']} / test {report['xt']['test']}")
    print("\nV(zona) % (ataque hacia la derecha):")
    grid = np.array(V).reshape(evp.GRID_X, evp.GRID_Y).T[::-1]
    for row in grid:
        print("  " + " ".join(f"{100*v:4.1f}" for v in row))
    print(f"\nescrito lib/{args.out} (features v{args.fv})")


if __name__ == "__main__":
    main()

"""Simulated matches of one deployed strategy, fed like the REAL engine."""
from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIB = HERE.parent / "lib"
SX, SY = 55.0 / 7.0, 30.0 / 3.5


def _compress(gs):
    g = copy.deepcopy(gs)

    def sc(v):
        if isinstance(v, dict):
            if isinstance(v.get("x"), (int, float)):
                v["x"] /= SX
            if isinstance(v.get("y"), (int, float)):
                v["y"] /= SY
    b = g.get("ball") or {}
    sc(b.get("position"))
    sc(b.get("velocity"))
    for p in g.get("players") or []:
        sc(p.get("position"))
        sc(p.get("velocity"))
    return g


_POL = {}
VARIANTS = {
    "mem31 sin portador": ("mem31", {"m31_carrier": False}),
    "mem31 sin defensa": ("mem31", {"m31_defence": False}),
    "mem31 sin balón suelto": ("mem31", {"m31_loose": False}),
    "mem31 sin asistencia": ("mem31", {"m31_assist_margin": 9.0}),
}


def _setup(strategy):
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(LIB))
    import tuning
    fname, extra = VARIANTS.get(strategy, (strategy, {}))
    tuning._DATA = tuning._load(str(LIB / f"tuning.{fname}.json"))
    tuning._DATA.setdefault("global", {}).update(extra)
    from teams import AGG, POS, TEAMS, repo_team
    import mem3
    base = repo_team(AGG, True, None, fast_ev=True)

    def ours(gs, team_id, pid):
        cgs = _compress(gs)
        label = POS[pid][2]
        c, tag = mem3.decide(cgs, team_id, pid, label)
        if c is not None:
            c = dict(c)
            c["_latency"] = 0.05
            return c
        c = base(cgs, team_id, pid)
        if c:
            mem3.shape([c], cgs, team_id, pid, label)
        return c
    _POL["ours"], _POL["teams"] = ours, TEAMS


def _play(job):
    from engine import Match
    rival, seed, duration = job
    key = "rival:" + rival
    if key not in _POL:
        _POL[key] = _POL["teams"][rival]()
    ours, opp = _POL["ours"], _POL[key]
    home = seed % 2 == 0
    m = Match(ours if home else opp, opp if home else ours, duration=duration, seed=seed)
    st = m.run()
    a, b = (st[0], st[1]) if home else (st[1], st[0])
    return {"rival": rival, "gf": a.goals, "ga": b.goals, "shots": a.shots,
            "opp_shots": b.shots, "pos": a.possession_s / duration}


def run_one(strategy, matches, duration, rivals, jobs):
    _setup(strategy)
    from teams import TEST_RIVALS, TRAIN_RIVALS
    rivals = rivals or (TRAIN_RIVALS + TEST_RIVALS)
    work = [(r, 700_000 + s, duration) for r in rivals for s in range(matches)]
    ctx = mp.get_context("fork")
    with ctx.Pool(jobs or os.cpu_count()) as pool:
        res = pool.map(_play, work, chunksize=2)
    out = {"n": 0, "pts": 0, "gf": 0, "ga": 0, "shots": 0, "opp_shots": 0, "pos": 0.0,
           "per_rival": {}}
    for r in res:
        pts = 3 if r["gf"] > r["ga"] else 1 if r["gf"] == r["ga"] else 0
        pr = out["per_rival"].setdefault(r["rival"], {"n": 0, "pts": 0, "gf": 0, "ga": 0})
        for d in (out, pr):
            d["n"] += 1
            d["pts"] += pts
            d["gf"] += r["gf"]
            d["ga"] += r["ga"]
        out["shots"] += r["shots"]
        out["opp_shots"] += r["opp_shots"]
        out["pos"] += r["pos"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", nargs="*", default=["mem3", "mem31"])
    ap.add_argument("--matches", type=int, default=20, help="per rival")
    ap.add_argument("--duration", type=float, default=240.0)
    ap.add_argument("--rivals", nargs="*")
    ap.add_argument("--jobs", type=int)
    ap.add_argument("--one", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.one:
        print("RESULT " + json.dumps(run_one(a.one, a.matches, a.duration, a.rivals, a.jobs)))
        return
    results = {}
    for s in a.strategies:
        cmd = [sys.executable, __file__, "--one", s, "--matches", str(a.matches),
               "--duration", str(a.duration)]
        if a.rivals:
            cmd += ["--rivals", *a.rivals]
        if a.jobs:
            cmd += ["--jobs", str(a.jobs)]
        out = subprocess.run(cmd, capture_output=True, text=True)
        line = [x for x in out.stdout.splitlines() if x.startswith("RESULT ")]
        if not line:
            print(out.stdout[-2000:], out.stderr[-3000:])
            continue
        results[s] = json.loads(line[-1][7:])
    if not results:
        return
    rivals = list(next(iter(results.values()))["per_rival"])
    print("| Estrategia | Pts/PJ | Goles | " + " | ".join(rivals) + " | Tiros (rival) | Posesión |")
    print("|---|---|---|" + "---|" * len(rivals) + "---|---|")
    for s, o in results.items():
        n = max(1, o["n"])
        cells = [f"{o['per_rival'][r]['pts'] / max(1, o['per_rival'][r]['n']):.2f}" for r in rivals]
        print(f"| {s} | **{o['pts'] / n:.2f}** | {o['gf']}-{o['ga']} | " + " | ".join(cells) +
              f" | {o['shots'] / n:.1f} ({o['opp_shots'] / n:.1f}) | {100 * o['pos'] / n:.0f} % |")


if __name__ == "__main__":
    main()

"""Round-robin tournament between teams in the local simulator."""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

_POLICIES = {}


def _policy(name):
    from teams import TEAMS
    if name not in _POLICIES:
        _POLICIES[name] = TEAMS[name]()
    return _POLICIES[name]


def play(job):
    from engine import Match
    a, b, seed, duration = job
    home, away = (a, b) if seed % 2 == 0 else (b, a)
    m = Match(_policy(home), _policy(away), duration=duration, seed=seed)
    st = m.run()
    out = {}
    for name, s in ((home, st[0]), (away, st[1])):
        out[name] = s.__dict__.copy()
    out["_pair"] = (a, b)
    return out


def main():
    from teams import TEAMS
    ap = argparse.ArgumentParser()
    ap.add_argument("--teams", nargs="*", default=list(TEAMS))
    ap.add_argument("--matches", type=int, default=20, help="per pair (even)")
    ap.add_argument("--duration", type=float, default=240.0, help="seconds per match")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--gauntlet", action="store_true",
                    help="only our variants vs every rival (no rival-vs-rival)")
    args = ap.parse_args()

    if args.gauntlet:
        from teams import OURS, RIVALS
        pairs = list(itertools.product(OURS, RIVALS))
        args.teams = OURS + RIVALS
    else:
        pairs = list(itertools.combinations(args.teams, 2))
    jobs = [(a, b, args.seed_offset + s, args.duration)
            for a, b in pairs for s in range(args.matches)]
    t0 = time.time()
    table = {n: dict(P=0, W=0, D=0, L=0, GF=0, GA=0, Pts=0, shots=0, on_target=0,
                     passes=0, passes_ok=0, steals=0, possession_s=0.0) for n in args.teams}
    h2h = {}
    with ProcessPoolExecutor(args.jobs) as ex:
        for res in ex.map(play, jobs, chunksize=4):
            a, b = res.pop("_pair")
            ga, gb = res[a]["goals"], res[b]["goals"]
            for n, gf, ga_ in ((a, ga, gb), (b, gb, ga)):
                row = table[n]
                row["P"] += 1
                row["GF"] += gf
                row["GA"] += ga_
                for k in ("shots", "on_target", "passes", "passes_ok", "steals", "possession_s"):
                    row[k] += res[n][k]
                if gf > ga_:
                    row["W"] += 1
                    row["Pts"] += 3
                elif gf == ga_:
                    row["D"] += 1
                    row["Pts"] += 1
                else:
                    row["L"] += 1
            k = (a, b)
            w = h2h.setdefault(k, [0, 0, 0, 0, 0])
            w[0 if ga > gb else 1 if ga == gb else 2] += 1
            w[3] += ga
            w[4] += gb
    elapsed = time.time() - t0

    ranked = sorted(table.items(), key=lambda kv: (-kv[1]["Pts"], -(kv[1]["GF"] - kv[1]["GA"])))
    lines = [f"# Torneo local ({len(jobs)} partidos, {args.matches} por cruce, "
             f"{args.duration:.0f} s por partido, {elapsed:.0f} s de cálculo)", "",
             "| # | Equipo | PJ | G | E | P | GF | GC | Pts | Pts/PJ | Tiros/PJ | % a puerta | % pases | Posesión |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, (n, r) in enumerate(ranked, 1):
        total_pos = r["possession_s"]
        lines.append(
            f"| {i} | {n} | {r['P']} | {r['W']} | {r['D']} | {r['L']} | {r['GF']} | {r['GA']} | "
            f"{r['Pts']} | {r['Pts']/r['P']:.2f} | {r['shots']/r['P']:.1f} | "
            f"{100*r['on_target']/max(1, r['shots']):.0f} | "
            f"{100*r['passes_ok']/max(1, r['passes']):.0f} | "
            f"{100*total_pos/(r['P']*args.duration):.0f} % |")
    lines += ["", "## Cara a cara (G-E-P y goles, desde el equipo de la fila)", ""]
    for (a, b), (w, dr, l, gf, ga) in sorted(h2h.items()):
        lines.append(f"- {a} vs {b}: {w}-{dr}-{l}  ({gf}-{ga})")
    if args.gauntlet:
        from teams import OURS, RIVALS
        short = {r: r.split(" (")[0] for r in RIVALS}
        lines += ["", "## Puntos por partido de cada variante contra cada rival", "",
                  "| Variante | " + " | ".join(short[r] for r in RIVALS) + " | Media |",
                  "|---|" + "---|" * (len(RIVALS) + 1)]
        for o in OURS:
            cells, tot = [], 0.0
            for r in RIVALS:
                w, dr, l, gf, ga = h2h[(o, r)]
                ppg = (3 * w + dr) / max(1, w + dr + l)
                tot += ppg
                cells.append(f"{ppg:.2f} ({gf}-{ga})")
            lines.append(f"| {o} | " + " | ".join(cells) + f" | **{tot/len(RIVALS):.2f}** |")
    report = "\n".join(lines)
    print(report)

    outdir = Path(__file__).parent / "results"
    outdir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (outdir / f"torneo-{stamp}.md").write_text(report + "\n", encoding="utf-8")
    (outdir / f"torneo-{stamp}.json").write_text(
        json.dumps({"table": table, "h2h": {f"{a} | {b}": v for (a, b), v in h2h.items()},
                    "args": vars(args)}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

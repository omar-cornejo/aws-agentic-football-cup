"""Compare strategies from REAL matches: results, play, LLM, memory, AgentCore."""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import difflib
import json
import re
import statistics as stats
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / "reports" / "cache"
PREFIXES = ["agg_", "mem31_", "mem32_", "mem33_", "mem3_", "mem2_", "mem_", "alg_", "llm_"]
NAMES = {"agg_": "agg (original)", "mem_": "mem v1", "mem2_": "mem2 (memoria v2 + Markov)",
         "mem3_": "mem3 (mem2 + órdenes/tiro/forma)",
         "mem31_": "mem3.1 (carril de tiro + Markov/CSP + bloqueo)",
         "mem32_": "mem3.2 (mem3 + órdenes 3.1 + tiro al centro, 35 m)",
         "mem33_": "mem3.3 (mem3.2 + DEF ≤ −20 m, memoria propia)",
         "alg_": "alg (Markov+CSP v3)", "llm_": "llm (megaprompts + memoria)"}
CODE = {"ev", "dsearch", "gk-ev", "gk-angle", "ev-support", "shoot-zone", "mid-pass",
        "def-up", "def-hold", "mid-centre", "aim-center", "coach-shoot-pass", "coach-shoot-carry",
        "m31-shot-close", "m31-shot-lane-clear", "m31-shot-deep-clear", "m31-assist",
        "m31-pass", "m31-carry", "m31-clear", "m31-shield", "m31-ev", "m31-press",
        "m31-double", "m31-block", "m31-loose"} | {f"coach-{o}" for o in ("shoot", "attack", "mark", "defend", "pass", "press")}
ROOT = "/aws/bedrock-agentcore/runtimes/"


def aws(*args):
    out = subprocess.run(["aws", *args, "--region", "us-east-1", "--output", "json"],
                         check=True, capture_output=True, text=True).stdout
    return json.loads(out) if out.strip() else None


def strategy_of(group: str) -> str | None:
    name = group[len(ROOT):]
    for p in PREFIXES:
        if name.startswith(p):
            return p
    return None


def fetch(start_ms: int) -> list[dict]:
    groups = aws("logs", "describe-log-groups", "--log-group-name-prefix", ROOT,
                 "--query", "logGroups[].logGroupName") or []
    rows = []
    for g in groups:
        strat = strategy_of(g)
        if strat is None:
            continue
        tok = None
        while True:
            a = ["logs", "filter-log-events", "--log-group-name", g, "--filter-pattern",
                 '"DECISION"', "--start-time", str(start_ms)]
            if tok:
                a += ["--next-token", tok]
            r = aws(*a)
            for e in r.get("events", []):
                try:
                    o = json.loads(e["message"])
                    d = json.loads(re.search(r"DECISION (\{.*\})", o["message"]).group(1))
                except Exception:
                    continue
                d["_ts"], d["_g"], d["_strat"] = e["timestamp"], g, strat
                d["_sid"] = re.sub(r"-pos\d$", "", o.get("sessionId", ""))
                rows.append(d)
            tok = r.get("nextToken")
            if not tok:
                break
    return rows


def runtime_metrics(start: dt.datetime, end: dt.datetime) -> dict:
    """Per strategy: AgentCore runtime metrics summed over its 5 runtimes."""
    out = collections.defaultdict(lambda: collections.defaultdict(float))
    lst = aws("cloudwatch", "list-metrics", "--namespace", "AWS/Bedrock-AgentCore") or {}
    wanted = {"Invocations": "Sum", "Errors": "Sum", "Throttles": "Sum",
              "SystemErrors": "Sum", "UserErrors": "Sum", "Latency": "Average",
              "CPUUsed-vCPUHours": "Sum", "MemoryUsed-GBHours": "Sum"}
    seen = set()
    for m in lst.get("Metrics", []):
        name = m["MetricName"]
        dims = {d["Name"]: d["Value"] for d in m["Dimensions"]}
        if name not in wanted or "Name" not in dims or "ComputeType" in dims:
            continue
        rt = dims["Name"].split("::")[0]
        strat = next((p for p in PREFIXES if rt.startswith(p)), None)
        key = (name, json.dumps(m["Dimensions"], sort_keys=True))
        if strat is None or key in seen:
            continue
        seen.add(key)
        r = aws("cloudwatch", "get-metric-statistics", "--namespace", "AWS/Bedrock-AgentCore",
                "--metric-name", name, "--dimensions", json.dumps(m["Dimensions"]),
                "--start-time", start.isoformat(), "--end-time", end.isoformat(),
                "--period", str(max(60, int((end - start).total_seconds()) // 60 * 60)),
                "--statistics", wanted[name])
        pts = [p[wanted[name]] for p in (r or {}).get("Datapoints", [])]
        if not pts:
            continue
        if wanted[name] == "Average":
            out[strat][name + "_n"] += 1
            out[strat][name] += stats.mean(pts)
        else:
            out[strat][name] += sum(pts)
    for s in out.values():
        if s.get("Latency_n"):
            s["Latency"] /= s.pop("Latency_n")
    return out


def our_team(R) -> int:
    tm = [r["tm"] for r in R if isinstance(r.get("tm"), int)]
    if tm:
        return collections.Counter(tm).most_common(1)[0][0]
    gk = [r["mx"] for r in R if r["pos"] == "GK" and isinstance(r.get("mx"), (int, float))]
    return 0 if (stats.median(gk) if gk else -1) < 0 else 1


def normalise(R):
    """Matches logged in engine units (x +-7): positions and dg to metres."""
    xs = [abs(r[k]) for r in R for k in ("mx",) if isinstance(r.get(k), (int, float))]
    if not xs or max(xs) > 8.5:
        return R
    sx, sy = 55.0 / 7.0, 30.0 / 3.5
    out = []
    for r in R:
        r = dict(r)
        if isinstance(r.get("bx"), (int, float)) and isinstance(r.get("by"), (int, float)):
            r["bx"], r["by"] = r["bx"] * sx, r["by"] * sy
        if isinstance(r.get("mx"), (int, float)) and isinstance(r.get("my"), (int, float)):
            r["mx"], r["my"] = r["mx"] * sx, r["my"] * sy
            gx = 55.0 if r.get("tm", 0) == 0 else -55.0
            r["dg"] = round(((gx - r["mx"]) ** 2 + r["my"] ** 2) ** 0.5)
        out.append(r)
    return out


def match_stats(R) -> dict:
    R = sorted(R, key=lambda r: r["_ts"])
    team = our_team(R)
    R = [dict(r, tm=r.get("tm", team)) for r in R]
    R = normalise(R)
    last = next((r for r in reversed(R) if isinstance(r.get("hs"), int)), {})
    gf = last.get("hs", 0) if team == 0 else last.get("as", 0)
    ga = last.get("as", 0) if team == 0 else last.get("hs", 0)
    ticks = collections.defaultdict(list)
    for r in R:
        ticks[round(r["_ts"] / 2000)].append(r)
    own = sum(1 for rr in ticks.values() if any(x.get("hb") for x in rr))
    opp = sum(1 for rr in ticks.values() if any(x.get("ob") for x in rr))
    src = collections.Counter(r["source"] for r in R)
    n = len(R)
    lat = sorted(r["latency_ms"] for r in R if r["source"] == "llm"
                 and isinstance(r.get("latency_ms"), (int, float)))
    shots = [r for r in R if r.get("cmd") == "SHOOT" and r.get("hb")
             and (r.get("pm") or "OPEN_PLAY") == "OPEN_PLAY"]
    dg = [r["dg"] for r in shots if isinstance(r.get("dg"), (int, float))]
    passes = sum(1 for r in R if r.get("cmd") in ("PASS", "GK_DISTRIBUTE") and r.get("hb"))
    per = collections.defaultdict(dict)
    for r in R:
        for k in ("mw", "mf", "ml", "mr", "md", "nb"):
            if k in r:
                per[r["pos"]][k] = r[k]
    mem = {"mw": sum(v.get("mw", 0) for v in per.values()),
           "mf": sum(v.get("mf", 0) for v in per.values()),
           "md": sum(v.get("md", 0) for v in per.values()),
           "ml": max((v.get("ml", 0) for v in per.values()), default=0),
           "mr": [v["mr"] for v in per.values() if "mr" in v],
           "nb": {p: v["nb"] for p, v in per.items() if "nb" in v}}
    beh = behaviour(R, team)
    return {
        "beh": beh,
        "sid": R[0]["_sid"], "start": R[0]["_ts"], "team": team, "gf": gf, "ga": ga,
        "res": "W" if gf > ga else "D" if gf == ga else "L",
        "ticks": len(ticks), "rows": n,
        "pos_pct": 100 * own / max(1, own + opp),
        "shots": len(shots), "shot_dist": stats.median(dg) if dg else None,
        "long_shots": sum(1 for d in dg if d > 45), "passes": passes,
        "llm_pct": 100 * src.get("llm", 0) / max(1, n),
        "code_pct": 100 * sum(v for k, v in src.items() if k in CODE) / max(1, n),
        "fb_pct": 100 * sum(v for k, v in src.items()
                            if "fallback" in k or k in ("last-resort", "llm-illegal")) / max(1, n),
        "timeouts": src.get("timeout-fallback", 0) + src.get("busy-fallback", 0),
        "lat_p50": stats.median(lat) if lat else None,
        "lat_p95": lat[max(0, int(len(lat) * 0.95) - 1)] if lat else None,
        "tin": sum(r.get("tin", 0) for r in R), "tout": sum(r.get("tout", 0) for r in R),
        "overrides": sum(1 for r in R if r.get("ov") and r["source"] == "llm"),
        "stamina": stats.mean([r["st"] for r in R if isinstance(r.get("st"), (int, float))] or [0]),
        "mem": mem,
    }


ORDERS = ["shoot", "goal", "attack", "mark", "defend", "pass", "press", "tackle"]
SHOOT_ORDERS = {"shoot", "goal", "attack"}


def order_word(chat: str) -> str | None:
    """Newest sideline message as one known word ('makr' -> 'mark'), else None."""
    try:
        m = str(json.loads(chat)[-1].get("message", "")).strip().lower()
    except Exception:
        return None
    hit = difflib.get_close_matches(m, ORDERS, 1, 0.6)
    return hit[0] if hit else (m or None)


def behaviour(R, team) -> dict:
    """What the players did, per row, as lists of 0/1 or values (aggregated later)."""
    sgn = 1 if team == 0 else -1
    b = collections.defaultdict(list)
    by_pos = collections.defaultdict(list)
    for r in R:
        by_pos[r["pos"]].append(r)
    for pos, rows in by_pos.items():
        order = None
        for a, r in zip([None] + rows, rows):
            if r.get("chat"):
                order = order_word(r["chat"])
                b["order_seen"].append(1)
                b["order_ok"].append(1 if r.get("co") else 0)
            if r.get("pm"):
                continue
            if a is not None and not a.get("pm"):
                b["gap"].append(r["t"] - a["t"])
            if pos == "GK":
                continue
            cmd, mx, my = r.get("cmd"), r.get("mx"), r.get("my")
            if r.get("hb") == 1:
                shot = 1 if cmd == "SHOOT" else 0
                d = r.get("dg")
                if isinstance(d, (int, float)):
                    b["sh_box" if d <= 16.5 else "sh_mid" if d <= 30 else "sh_far"].append(shot)
                b["sh_ord" if order in SHOOT_ORDERS else "sh_free"].append(shot)
            elif r.get("ob") == 1:
                b["mk_ord" if order == "mark" else "mk_free"].append(1 if cmd == "MARK" else 0)
                if isinstance(mx, (int, float)) and isinstance(r.get("bx"), (int, float)):
                    if ((mx - r["bx"]) ** 2 + (my - r["by"]) ** 2) ** 0.5 <= 4:
                        b["tackle_near"].append(1 if cmd == "SLIDE_TACKLE" else 0)
            if isinstance(mx, (int, float)) and abs(mx) <= 60:
                if pos == "DEF":
                    b["def_x"].append(sgn * mx)
                if pos == "MID":
                    b["mid_y"].append(abs(my))
            if r["source"] == "llm" and isinstance(r.get("latency_ms"), (int, float)):
                b["lat_" + pos].append(r["latency_ms"])
    return dict(b)


def matches(rows):
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["_strat"], r["_sid"])].append(r)
    out = collections.defaultdict(list)
    for (strat, sid), R in by.items():
        if len({r["pos"] for r in R}) >= 4 and len(R) >= 100:
            out[strat].append(match_stats(R))
    for v in out.values():
        v.sort(key=lambda m: m["start"])
    return out


def f(v, nd=0, suf=""):
    if v is None:
        return "-"
    return f"{v:.{nd}f}{suf}"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return stats.mean(xs) if xs else None


def report(M, RT, title) -> str:
    L = [f"# {title}", ""]
    L += ["## Resumen por estrategia", "",
          "| Estrategia | PJ | G-E-P | Pts/PJ | GF/PJ | GC/PJ | Posesión | Tiros/PJ | Dist. tiro (med.) | "
          "Tiros >45 m | Pases/PJ | LLM % | Código % | Fallback % | LLM p50 / p95 ms | Tokens in/out por PJ | Stamina media |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in PREFIXES:
        ms = M.get(p) or []
        if not ms:
            continue
        w = sum(m["res"] == "W" for m in ms)
        d = sum(m["res"] == "D" for m in ms)
        l_ = sum(m["res"] == "L" for m in ms)
        n = len(ms)
        L.append(
            f"| {NAMES[p]} | {n} | {w}-{d}-{l_} | **{(3 * w + d) / n:.2f}** | "
            f"{mean([m['gf'] for m in ms]):.2f} | {mean([m['ga'] for m in ms]):.2f} | "
            f"{f(mean([m['pos_pct'] for m in ms]), 0, ' %')} | {f(mean([m['shots'] for m in ms]), 1)} | "
            f"{f(mean([m['shot_dist'] for m in ms]), 0, ' m')} | {f(mean([m['long_shots'] for m in ms]), 1)} | "
            f"{f(mean([m['passes'] for m in ms]), 1)} | {f(mean([m['llm_pct'] for m in ms]), 0, ' %')} | "
            f"{f(mean([m['code_pct'] for m in ms]), 0, ' %')} | {f(mean([m['fb_pct'] for m in ms]), 1, ' %')} | "
            f"{f(mean([m['lat_p50'] for m in ms]))} / {f(mean([m['lat_p95'] for m in ms]))} | "
            f"{f(mean([m['tin'] for m in ms]))} / {f(mean([m['tout'] for m in ms]))} | "
            f"{f(mean([m['stamina'] for m in ms]))} |")
    L += ["", "Pts/PJ = 3 por victoria, 1 por empate. Posesión = ticks con balón nuestro / (nuestro + suyo). "
          "Fallback = el LLM llegó tarde, no respondió JSON o mandó una orden ilegal.", ""]

    L += ["## Comportamiento (qué hace cada modelo en el campo)", "",
          "| Estrategia | Hueco entre decisiones p50 / p90 s | Huecos ≥5 s | LLM p50 ms por puesto | "
          "Tira con balón: área / 16-30 m / >30 m | Tira con orden shoot·goal·attack / sin orden | "
          "MARK con orden mark / sin orden | Segada con rival ≤4 m | DEF x media | MID |y| media | Órdenes reconocidas |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]

    def pc(xs):
        return f"{100 * sum(xs) / len(xs):.0f} % ({len(xs)})" if xs else "-"

    def pq(xs, q):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(len(xs) * q))] if xs else None

    for p in PREFIXES:
        ms = M.get(p) or []
        if not ms:
            continue
        B = collections.defaultdict(list)
        for m in ms:
            for k, v in m["beh"].items():
                B[k] += v
        lat = " ".join(f"{k[4:]} {stats.median(v):.0f}" for k, v in sorted(B.items())
                       if k.startswith("lat_") and len(v) >= 5)
        L.append(f"| {NAMES[p]} | {f(pq(B['gap'], .5))} / {f(pq(B['gap'], .9))} | "
                 f"{f(100 * sum(g >= 5 for g in B['gap']) / max(1, len(B['gap'])), 0, ' %')} | {lat or '-'} | "
                 f"{pc(B['sh_box'])} / {pc(B['sh_mid'])} / {pc(B['sh_far'])} | "
                 f"{pc(B['sh_ord'])} / {pc(B['sh_free'])} | {pc(B['mk_ord'])} / {pc(B['mk_free'])} | "
                 f"{pc(B['tackle_near'])} | {f(mean(B['def_x']), 0, ' m')} | {f(mean(B['mid_y']), 0, ' m')} | "
                 f"{pc(B['order_ok'])} |")
    L += ["", "Hueco = segundos de partido entre dos decisiones del mismo jugador (sin contar reinicios). "
          "x en nuestra dirección de ataque (0 = medio campo, -55 = nuestra portería). "
          "Órdenes reconocidas = el mensaje del banquillo llegó al jugador como orden propia (co). "
          "Las órdenes con faltas de ortografía se agrupan (makr → mark).", ""]

    L += ["## Memoria (AgentCore Memory, memoria v2)", "",
          "| Estrategia | Eventos escritos/PJ | Fallos | Descartados | Partidos previos leídos (máx.) | Lectura lecciones ms |",
          "|---|---|---|---|---|---|"]
    for p in PREFIXES:
        ms = M.get(p) or []
        if not ms or not any(m["mem"]["mw"] or m["mem"]["ml"] for m in ms):
            continue
        L.append(f"| {NAMES[p]} | {f(mean([m['mem']['mw'] for m in ms]))} | "
                 f"{sum(m['mem']['mf'] for m in ms)} | {sum(m['mem']['md'] for m in ms)} | "
                 f"{max(m['mem']['ml'] for m in ms)} | "
                 f"{f(mean([x for m in ms for x in m['mem']['mr']]))} |")
    L.append("")

    if RT:
        L += ["## AgentCore runtime (CloudWatch, ventana completa)", "",
              "| Estrategia | Invocaciones | Latencia media ms | Errores | Throttles | vCPU-h | GB-h |",
              "|---|---|---|---|---|---|---|"]
        for p in PREFIXES:
            s = RT.get(p)
            if not s:
                continue
            L.append(f"| {NAMES[p]} | {f(s.get('Invocations'))} | {f(s.get('Latency'))} | "
                     f"{f(s.get('Errors', 0) + s.get('SystemErrors', 0) + s.get('UserErrors', 0))} | "
                     f"{f(s.get('Throttles'))} | {f(s.get('CPUUsed-vCPUHours'), 3)} | "
                     f"{f(s.get('MemoryUsed-GBHours'), 3)} |")
        L.append("")

    L += ["## Partido a partido", "",
          "| Estrategia | Hora | Sesión | Lado | Resultado | Posesión | Tiros (>45 m) | LLM % | Fallback % | p50 ms | Tokens in | Cuaderno (p=pases s=tiros c=regates) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in PREFIXES:
        for m in M.get(p) or []:
            t = time.strftime("%d/%m %H:%M", time.localtime(m["start"] / 1000))
            nb = " ".join(f"{k}:{v}" for k, v in sorted(m["mem"]["nb"].items()))
            L.append(f"| {NAMES[p]} | {t} | {m['sid'][:8]} | {'casa' if m['team'] == 0 else 'fuera'} | "
                     f"{m['gf']}–{m['ga']} ({m['res']}) | {m['pos_pct']:.0f} % | {m['shots']} ({m['long_shots']}) | "
                     f"{m['llm_pct']:.0f} % | {m['fb_pct']:.1f} % | {f(m['lat_p50'])} | {m['tin']} | {nb or '-'} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=24)
    ap.add_argument("--since", help='"YYYY-MM-DD HH:MM" local time (overrides --hours)')
    ap.add_argument("--cached", action="store_true")
    ap.add_argument("--no-metrics", action="store_true", help="skip AgentCore CloudWatch metrics")
    a = ap.parse_args()
    end = dt.datetime.now().astimezone()
    start = (dt.datetime.strptime(a.since, "%Y-%m-%d %H:%M").astimezone() if a.since
             else end - dt.timedelta(hours=a.hours))
    if start >= end:
        ap.error(f"--since {a.since} está en el futuro (ahora son las {end:%H:%M}); usa --hours")
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / "strategies.json"
    if a.cached and cache.exists():
        rows = json.loads(cache.read_text())
    else:
        rows = fetch(int(start.timestamp() * 1000))
        cache.write_text(json.dumps(rows))
    M = matches(rows)
    RT = {} if a.no_metrics else runtime_metrics(start.astimezone(dt.timezone.utc),
                                                 end.astimezone(dt.timezone.utc))
    title = f"Comparativa de estrategias — {start:%d/%m %H:%M} a {end:%d/%m %H:%M}"
    rep = report(M, RT, title)
    out = HERE / "reports" / f"strategies-{end:%Y%m%d-%H%M}.md"
    out.write_text(rep, encoding="utf-8")
    print(rep)
    print(f"-> {out}")


if __name__ == "__main__":
    main()

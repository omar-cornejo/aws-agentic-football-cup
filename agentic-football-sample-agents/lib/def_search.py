"""Defensive assignment by search (IDS + backtracking, heuristic ordering)."""
from __future__ import annotations

import math
import time

import ev_policy as evp

REACH_M = 11.0
PRESS_WIN = 0.35
TOP_K = 4
BUDGET_S = 0.030
MAX_EVALS = 400
SAFETY_S = 0.250

import tuning as _tun
CSP = {"on": bool(_tun.get("dsearch_csp", False)), "cover_m": 45.0,
       "def_press_m": 38.0, "double_m": 30.0, "double_win": 0.5}


def _d(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _toward(p, t, reach=None):
    reach = REACH_M if reach is None else reach
    dist = _d(p, t)
    if dist <= reach:
        return t
    f = reach / dist
    return (p[0] + (t[0] - p[0]) * f, p[1] + (t[1] - p[1]) * f)


def _task_target(task, carrier, opps, own_goal):
    kind, j = task
    if kind == "PRESS":
        return carrier
    if kind == "PRESS2":
        dist = _d(carrier, own_goal) or 1.0
        return (carrier[0] + (own_goal[0] - carrier[0]) / dist * 3.0,
                carrier[1] + (own_goal[1] - carrier[1]) / dist * 3.0)
    if kind == "MARK":
        o = opps[j]
        dist = _d(o, own_goal) or 1.0
        return (o[0] + (own_goal[0] - o[0]) / dist * 2.0, o[1] + (own_goal[1] - o[1]) / dist * 2.0)
    t = min(0.4, 18.0 / max(_d(carrier, own_goal), 1.0))
    return (own_goal[0] + (carrier[0] - own_goal[0]) * t, own_goal[1] + (carrier[1] - own_goal[1]) * t)


class _Problem:
    def __init__(self, game_state, team_id, model):
        from state import _player_idx, _is_my_team, get_goal_positions, resolve_holder
        self.m = model
        self.gs = game_state
        players = game_state.get("players") or []
        holder = resolve_holder(game_state.get("ball") or {}, players)
        self.ok = (holder is not None and not _is_my_team(holder, team_id) and model is not None)
        if not self.ok:
            return

        def pos(p):
            q = p.get("position") or {}
            return (float(q.get("x", 0) or 0), float(q.get("y", 0) or 0))

        self.team = team_id
        own_x, _ = get_goal_positions(team_id)
        self.own_goal = (own_x, 0.0)
        self.carrier_idx = _player_idx(holder)
        self.carrier = pos(holder)
        self.mine = {_player_idx(p): pos(p) for p in players
                     if _is_my_team(p, team_id) and _player_idx(p) != 0}
        self.opps = {_player_idx(p): pos(p) for p in players if not _is_my_team(p, team_id)}
        self.base_players = players
        self.opp_team = 1 - team_id
        self.tasks = [("PRESS", self.carrier_idx), ("COVER", -1)]
        self.csp = bool(CSP.get("on"))
        self.d_carrier_goal = _d(self.carrier, self.own_goal)
        if self.csp and self.d_carrier_goal <= CSP["double_m"]:
            self.tasks.append(("PRESS2", self.carrier_idx))
        self.tasks += [("MARK", j) for j in self.opps if j not in (0, self.carrier_idx)]
        self.order = sorted(self.mine, key=lambda i: _d(self.mine[i], self.carrier))
        self.targets = {t: _task_target(t, self.carrier, self.opps, self.own_goal)
                        for t in self.tasks}
        self.evals = 0

    def allowed(self, i, t):
        """Unary constraint: may player i take task t?"""
        if not self.csp:
            return True
        if i == 1 and t[0] in ("PRESS", "PRESS2") and self.d_carrier_goal > CSP["def_press_m"]:
            return False
        return True

    def feasible(self, assign):
        """Global constraint on a complete plan."""
        if not self.csp:
            return True
        kinds = [t[0] for t in assign.values()]
        if self.d_carrier_goal <= CSP["cover_m"] and "COVER" not in kinds:
            return False
        return True

    def repair(self, assign):
        """Make a plan feasible: the nearest non-presser takes COVER."""
        if self.feasible(assign):
            return assign
        cover = ("COVER", -1)
        cands = [i for i, t in assign.items() if t[0] not in ("PRESS",)]
        if not cands:
            return assign
        i = min(cands, key=lambda i: _d(self.mine[i], self.targets[cover]))
        out = dict(assign)
        out[i] = cover
        return out

    def greedy(self, players, taken):
        """Heuristic: each remaining player takes the nearest free task."""
        out = {}
        used = set(taken)
        for i in players:
            free = [t for t in self.tasks if t not in used and self.allowed(i, t)]
            if not free:
                break
            t = min(free, key=lambda t: _d(self.mine[i], self.targets[t]))
            out[i] = t
            used.add(t)
        return out

    def threat(self, assign):
        """Opponent carrier's best EV after our players move toward their tasks."""
        self.evals += 1
        moved = {i: _toward(self.mine[i], self.targets[t]) for i, t in assign.items()}
        players = []
        for p in self.base_players:
            from state import _player_idx, _is_my_team
            idx = _player_idx(p)
            if _is_my_team(p, self.team) and idx in moved:
                q = dict(p)
                q["position"] = {"x": moved[idx][0], "y": moved[idx][1]}
                players.append(q)
            else:
                players.append(p)
        gs = dict(self.gs)
        gs["players"] = players
        ctx = evp._Ctx(gs, self.opp_team, self.carrier_idx, self.m)
        best = max(o[0] for o in ctx.options())
        pressers = sum(1 for i, t in assign.items()
                       if t[0] in ("PRESS", "PRESS2") and _d(moved[i], self.carrier) < 2.5)
        if pressers >= 2:
            return best * (1 - PRESS_WIN) * (1 - CSP["double_win"] * PRESS_WIN)
        return best * (1 - PRESS_WIN if pressers else 1.0)

    def solve(self, budget=BUDGET_S, max_evals=None):
        max_evals = MAX_EVALS if max_evals is None else max_evals
        t_end = time.perf_counter() + max(budget, SAFETY_S)
        best_assign = self.repair(self.greedy(self.order, set()))
        best_val = self.threat(best_assign)
        evals = 1
        depth_done = 0
        for depth in range(1, len(self.order) + 1):
            head, tail = self.order[:depth], self.order[depth:]
            timed_out = False

            def bt(k, partial, used):
                nonlocal best_val, best_assign, timed_out, evals
                if evals >= max_evals or time.perf_counter() > t_end:
                    timed_out = True
                    return
                if k == len(head):
                    evals += 1
                    full = dict(partial)
                    full.update(self.greedy(tail, used))
                    full = self.repair(full)
                    if not self.feasible(full):
                        return
                    v = self.threat(full)
                    if v < best_val - 1e-9:
                        best_val, best_assign = v, full
                    return
                i = head[k]
                cands = sorted((t for t in self.tasks if t not in used and self.allowed(i, t)),
                               key=lambda t: _d(self.mine[i], self.targets[t]))[:TOP_K]
                for t in cands:
                    partial[i] = t
                    used.add(t)
                    bt(k + 1, partial, used)
                    used.discard(t)
                    del partial[i]
                    if timed_out:
                        return

            bt(0, {}, set())
            if timed_out:
                break
            depth_done = depth
        self.evals = evals
        return best_assign, best_val, depth_done


def assign(game_state, team_id, my_pid, budget=BUDGET_S, model=None):
    """Return (command, info) for my_pid, or (None, None) outside the
    opponent-possession phase."""
    if my_pid == 0:
        return None, None
    prob = _Problem(game_state, team_id, model or evp.MODEL)
    if not prob.ok or my_pid not in prob.mine:
        return None, None
    plan, val, depth = prob.solve(budget)
    task = plan.get(my_pid)
    if task is None:
        return None, None
    kind, j = task
    me = prob.mine[my_pid]
    if kind in ("PRESS", "PRESS2"):
        if _d(me, prob.carrier) <= 3.0:
            cmd = {"commandType": "SLIDE_TACKLE", "parameters": {"target_player_id": -1, "sprint": True}}
        else:
            cmd = {"commandType": "PRESS_BALL", "parameters": {"intensity": 1.0}}
    elif kind == "MARK":
        cmd = {"commandType": "MARK", "parameters": {"target_player_id": j, "tightness": "TIGHT"}}
    else:
        tx, ty = prob.targets[task]
        cmd = {"commandType": "MOVE_TO", "parameters": {"target_x": round(tx, 1),
                                                        "target_y": round(ty, 1), "sprint": True}}
    cmd.update(playerId=my_pid, teamId=team_id, duration=0)
    return cmd, {"task": f"{kind}{'' if j < 0 else j}", "threat": round(val, 4),
                 "depth": depth, "evals": prob.evals}


def gk_position(game_state, team_id):
    """Heuristic keeper spot: on the ball-goal line, up to 4 m off the line,
    narrowing the angle (further out when the ball is far)."""
    from state import get_goal_positions
    b = (game_state.get("ball") or {}).get("position") or {}
    bx, by = float(b.get("x", 0) or 0), float(b.get("y", 0) or 0)
    own_x, _ = get_goal_positions(team_id)
    dist = math.hypot(bx - own_x, by) or 1.0
    step = min(4.0, 0.12 * dist)
    return own_x + (bx - own_x) / dist * step, max(-6.0, min(6.0, by / dist * step * 3))


def gk_distribute(game_state, team_id, model=None):
    """Keeper with the ball: pass to the teammate with the best EV."""
    ctx = evp._Ctx(game_state, team_id, 0, model or evp.MODEL)
    if not ctx.ok:
        return None, None
    opts = [o for o in ctx.options() if o[1].startswith("pass")]
    if not opts:
        return None, None
    ev, label, cmd = max(opts, key=lambda o: o[0])
    prm = cmd["parameters"]
    method = "KICK" if prm["type"] in ("AERIAL", "THROUGH") else "THROW"
    out = {"commandType": "GK_DISTRIBUTE", "playerId": 0, "teamId": team_id, "duration": 0,
           "parameters": {"target_player_id": prm["target_player_id"], "method": method}}
    return out, {"ev": round(ev, 4), "choice": label}

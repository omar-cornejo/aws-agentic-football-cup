"""Shared-lib regression tests: possession, state enrichment, orchestration, guardrails."""

import copy
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib"))

from test_helpers import GAME_STATE

import tuning as _tuning_base
_tuning_base._DATA.setdefault("global", {}).update(
    {"line_push_DEF": 0, "line_push_MID": 0, "ev_shot_bias": 1.0,
     "mem_miss_after": 3, "lead_long_shot_mult": 0.6})
from state import summarize_state, get_possession_info, resolve_holder, _is_my_team
from tactics import tactics_report, _lane_blocked, shot_power
from pattern_tracker import PatternTracker
from parsing import parse_commands
from overrides import OverrideConfig, apply_overrides
import math

def _d_goal(pos, opp_goal_x=55.0):
    return math.hypot(opp_goal_x - pos.get("x", 0), pos.get("y", 0))

gs = copy.deepcopy(GAME_STATE)
gs["ball"]["possessionAgentId"] = "agentId_3"
gs["ball"]["position"] = {"x": 30.0, "y": -12.0, "z": 0}

holder = resolve_holder(gs["ball"], gs["players"])
assert holder["teamCode"] == "away", f"holder should be away P3, got {holder['teamCode']}"

pid, status, mine = get_possession_info(gs["ball"], gs["players"], 0)
assert status == "OPP player 3" and not mine, f"team0 view wrong: {status}, mine={mine}"
pid, status, mine = get_possession_info(gs["ball"], gs["players"], 1)
assert status == "MY player 3" and mine, f"team1 view wrong: {status}, mine={mine}"

summary = summarize_state(gs, 0, 3, "FWD1")
assert "hasBall=False" in summary, "home FWD1 must NOT think it has the ball"
assert "held by OPP player 3" in summary

rep = tactics_report(gs, 0, 3, "FWD1")
assert "Shot:" not in rep, f"no shot line without ball: {rep}"
rep_def = tactics_report(gs, 0, 1, "DEF")
assert "Top threat: P3" in rep_def and "has ball" in rep_def, rep_def

t = PatternTracker()
for i in range(12):
    g = copy.deepcopy(gs)
    g["gameTime"] = 10 + i * 5
    t.update(g, 0)
assert t.opp_hold[3] == 12 and t.opp_hold_total == 12, dict(t.opp_hold)

gs2 = copy.deepcopy(GAME_STATE)
gs2["ball"]["possessionAgentId"] = "agentId_3"
gs2["ball"]["position"] = {"x": 14.0, "y": -5.0, "z": 0}

holder2 = resolve_holder(gs2["ball"], gs2["players"])
assert holder2["teamCode"] == "home"

summary2 = summarize_state(gs2, 0, 3, "FWD1")
assert "hasBall=True" in summary2, "home FWD1 SHOULD have the ball now"
assert "held by MY player 3" in summary2

rep2 = tactics_report(gs2, 0, 3, "FWD1")
assert "Shot:" in rep2, rep2
assert ("LANE CLEAR" in rep2 and "SHOOT NOW" in rep2) or "LANE BLOCKED" in rep2, rep2

gs3 = copy.deepcopy(GAME_STATE)
gs3["ball"]["possessionAgentId"] = "agentId_0"
gs3["ball"]["position"] = {"x": 50.0, "y": 0.0, "z": 0}
summary3 = summarize_state(gs3, 0, 0, "GK")
assert "hasBall=False" in summary3, "our GK must not think it holds the opponent GK's ball"

gs4 = copy.deepcopy(GAME_STATE)
gs4["ball"]["possessionAgentId"] = None
gs4["ball"]["isFree"] = True
gs4["ball"]["velocity"] = {"x": 3.0, "y": -1.0, "z": 0}
gs4["teamChat"] = [{"message": "全员压上，多射门"}]
gs4["playMode"] = "FREE_KICK"
for p in gs4["players"]:
    if p["teamCode"] == "home" and p["agentId"] == "agentId_3":
        p["stamina"] = 10
        p["lastAction"] = "MOVE_TO"
summary4 = summarize_state(gs4, 0, 3, "FWD1")
assert "FREE ball" in summary4 and "heading to" in summary4, summary4
assert "SIDELINE SHOUT" in summary4 and "全员压上" in summary4, summary4
assert "RESTART (FREE_KICK)" in summary4, summary4
assert "LOW STAMINA" in summary4, summary4
assert "lastAction=MOVE_TO" in summary4, summary4

gs5 = copy.deepcopy(GAME_STATE)
gs5["teamChat"] = ["press high"]
assert "SIDELINE SHOUT" in summarize_state(gs5, 0, 3, "FWD1") and "press high" in summarize_state(gs5, 0, 3, "FWD1")
assert "COACH ORDER" not in summarize_state(GAME_STATE, 0, 3, "FWD1") and "SIDELINE" not in summarize_state(GAME_STATE, 0, 3, "FWD1")

gs6 = copy.deepcopy(GAME_STATE)
gs6["ball"]["possessionAgentId"] = None
gs6["ball"]["isFree"] = True
s_chaser = summarize_state(gs6, 0, 3, "FWD1")
assert "CLOSEST player to the free ball — chase" in s_chaser, s_chaser
s_holder = summarize_state(gs6, 0, 4, "FWD2")
assert "teammate P3 is closest to the free ball — do NOT chase" in s_holder, s_holder

s_press = summarize_state(gs, 0, 3, "FWD1")
assert "designated PRESSER" in s_press, s_press
s_cover = summarize_state(gs, 0, 2, "MID")
assert "P3 presses the carrier" in s_cover, s_cover
assert "ASSIGNMENT" not in summarize_state(gs2, 0, 4, "FWD2")

gs7 = copy.deepcopy(GAME_STATE)
gs7["gameTime"] = 250
gs7["score"] = {"home": 0, "away": 1}
assert "GAME PLAN: trailing late" in summarize_state(gs7, 0, 3, "FWD1")
assert "GAME PLAN: leading late" in summarize_state(gs7, 1, 3, "FWD1")
gs7["score"] = {"home": 1, "away": 1}
assert "GAME PLAN: tied late" in summarize_state(gs7, 0, 3, "FWD1")
gs7["gameTime"] = 100
assert "GAME PLAN" not in summarize_state(gs7, 0, 3, "FWD1")

cmds = parse_commands(
    '[{"commandType":"SHOOT","parameters":{"aim_location":"TOP","power":5}},'
    ' {"commandType":"MOVE_TO","parameters":{"target_x":120,"target_y":-99}},'
    ' {"commandType":"PASS","parameters":{"target_player_id":9}},'
    ' {"commandType":"PRESS_BALL","parameters":{"intensity":3}},'
    ' {"commandType":"TELEPORT","parameters":{}}]', 0, 3)
assert len(cmds) == 4, cmds
assert cmds[0]["parameters"] == {"aim_location": "CENTER", "power": 1.0}, cmds[0]
assert cmds[1]["parameters"] == {"target_x": 55, "target_y": -35}, cmds[1]
assert cmds[2]["parameters"]["target_player_id"] == 4, cmds[2]
assert cmds[3]["parameters"]["intensity"] == 1.0, cmds[3]
shoot_bare = parse_commands('[{"commandType":"SHOOT","parameters":{}}]', 0, 4)[0]
assert shoot_bare["parameters"] == {"aim_location": "CENTER", "power": 1.0}, shoot_bare
pass_self = parse_commands('[{"commandType":"PASS","parameters":{"target_player_id":4}}]', 0, 4)[0]
assert pass_self["parameters"]["target_player_id"] == 3, pass_self

lite_arith = parse_commands(
    '```json\n[{"commandType":"MOVE_TO","playerId":4,'
    '"parameters":{"target_x":-55*0.5,"target_y":14,"sprint":true},"duration":0}]\n```',
    0, 4)
assert lite_arith and lite_arith[0]["parameters"]["target_x"] == -27.5, lite_arith
lite_arith2 = parse_commands(
    '[{"commandType":"MOVE_TO","parameters":{"target_x":55*0.75,"target_y":-14}}]', 0, 3)
assert lite_arith2 and lite_arith2[0]["parameters"]["target_x"] == 41.25, lite_arith2
assert parse_commands('[{"commandType":"MOVE_TO","parameters":{"target_x":40.5,"target_y":-8}}]',
                      0, 3)[0]["parameters"]["target_x"] == 40.5

me_open = {"x": 30.0, "y": 0.0}
blocked, off = _lane_blocked(me_open, 55.0, [
    {"position": {"x": 45, "y": 20}},
    {"position": {"x": 20, "y": 0}},
])
assert not blocked and off == 0.0, (blocked, off)

blocked, off = _lane_blocked(me_open, 55.0, [
    {"position": {"x": 40, "y": 0}},
])
assert blocked and off != 0.0, (blocked, off)

gs8 = copy.deepcopy(GAME_STATE)
gs8["ball"]["possessionAgentId"] = "agentId_3"
gs8["ball"]["position"] = {"x": 14.0, "y": -5.0, "z": 0}
for p in gs8["players"]:
    if p["teamCode"] == "away" and p["agentId"] == "agentId_1":
        p["position"] = {"x": 30, "y": -3}
    elif p["teamCode"] == "away" and p["agentId"] != "agentId_0":
        p["position"] = {"x": 45, "y": 25}
rep8 = tactics_report(gs8, 0, 3, "FWD1")
assert "LANE BLOCKED" in rep8 and "MOVE_TO" in rep8, rep8

gs9 = copy.deepcopy(GAME_STATE)
gs9["ball"]["possessionAgentId"] = "agentId_3"
gs9["ball"]["position"] = {"x": 14.0, "y": -5.0, "z": 0}
for p in gs9["players"]:
    if p["teamCode"] == "away":
        p["position"] = {"x": 45, "y": 25}
rep9 = tactics_report(gs9, 0, 3, "FWD1")
assert "LANE CLEAR" in rep9 and "SHOOT NOW: aim" in rep9, rep9

gs10 = copy.deepcopy(GAME_STATE)
gs10["ball"]["possessionAgentId"] = "agentId_3"
gs10["ball"]["position"] = {"x": 48.0, "y": 0.0, "z": 0}
for p in gs10["players"]:
    if p["teamCode"] == "home" and p["agentId"] == "agentId_3":
        p["position"] = {"x": 48, "y": 0}
    if p["teamCode"] == "away" and p["agentId"] == "agentId_1":
        p["position"] = {"x": 52, "y": 0}
rep10 = tactics_report(gs10, 0, 3, "FWD1")
assert "POINT-BLANK" in rep10 or "LANE CLEAR" in rep10, rep10
assert "SHOOT NOW" in rep10, rep10

OV = OverrideConfig()

def _mk(ctype, **params):
    return [{"commandType": ctype, "playerId": 3, "teamId": 0, "parameters": params}]

out, tag = apply_overrides(_mk("MOVE_TO", target_x=30, target_y=0, sprint=True),
                           gs2, 0, 3, "FWD1", OV)
assert tag == "shoot" and out[0]["commandType"] == "SHOOT", (tag, out)
assert out[0]["parameters"]["aim_location"] == "CENTER"

gsJ1b = copy.deepcopy(GAME_STATE)
gsJ1b["ball"]["possessionAgentId"] = "agentId_3"
gsJ1b["ball"]["position"] = {"x": 14.0, "y": -5.0, "z": 0}
for p in gsJ1b["players"]:
    if p["teamCode"] == "away":
        p["position"] = {"x": 45, "y": 25}
out, tag = apply_overrides(_mk("MOVE_TO", target_x=30, target_y=0, sprint=True),
                           gsJ1b, 0, 3, "FWD1", OV)
assert tag == "shoot" and out[0]["commandType"] == "SHOOT", (tag, out)
assert out[0]["parameters"]["aim_location"] == "CENTER"

out, tag = apply_overrides(_mk("SHOOT", aim_location="CENTER", power=0.5),
                           gs2, 0, 3, "FWD1", OV)
assert tag in ("shoot", "aim") and out[0]["commandType"] == "SHOOT", (tag, out)
assert out[0]["parameters"]["aim_location"] == "CENTER"

cmds4 = [{"commandType": "PRESS_BALL", "playerId": 4, "teamId": 0,
          "parameters": {"intensity": 1.0}}]
out, tag = apply_overrides(cmds4, gs, 0, 4, "FWD2", OV)
assert tag == "no-chase" and out[0]["commandType"] == "MARK", (tag, out)
assert out[0]["parameters"]["target_player_id"] in (1, 2), out

out, tag = apply_overrides(_mk("PRESS_BALL", intensity=1.0), gs, 0, 3, "FWD1", OV)
assert tag is None and out[0]["commandType"] == "PRESS_BALL", (tag, out)

cmds2 = [{"commandType": "MOVE_TO", "playerId": 2, "teamId": 0,
          "parameters": {"target_x": 45, "target_y": 20, "sprint": True}}]
out, tag = apply_overrides(cmds2, gs, 0, 2, "MID", OV)
assert tag in ("anchor", "hold-line", "mark-near"), (tag, out)
assert out[0]["commandType"] == "MOVE_TO", out
assert -33 <= out[0]["parameters"]["target_x"] <= -17, out

out, tag = apply_overrides([{"commandType": "PRESS_BALL", "playerId": 4, "teamId": 0,
                             "parameters": {"intensity": 1.0}}], gs2, 0, 4, "FWD2", OV)
assert tag == "support" and out[0]["commandType"] == "MOVE_TO", (tag, out)
assert out[0]["parameters"]["target_x"] == 42, out

gs11 = copy.deepcopy(GAME_STATE)
gs11["ball"]["possessionAgentId"] = "agentId_3"
gs11["ball"]["position"] = {"x": -20.0, "y": -5.0, "z": 0}
for p in gs11["players"]:
    if p["teamCode"] == "home" and p["agentId"] == "agentId_3":
        p["position"] = {"x": -20, "y": -5}
out, tag = apply_overrides(_mk("MOVE_TO", target_x=0, target_y=0, sprint=True),
                           gs11, 0, 3, "FWD1", OverrideConfig(wing_y=-14.0))
assert tag == "wing" and out[0]["parameters"]["target_y"] == -14.0, (tag, out)

out, tag = apply_overrides([{"commandType": "MOVE_TO", "playerId": 0, "teamId": 0,
                             "parameters": {"target_x": 0, "target_y": 0}}], gs, 0, 0, "GK", OV)
assert tag == "gk-line" and out[0]["parameters"]["target_x"] == -52.8, (tag, out)
out, tag = apply_overrides(_mk("MOVE_TO", target_x=30, target_y=0), gs2, 0, 3, "FWD1", None)
assert tag is None and out[0]["commandType"] == "MOVE_TO"

BLAST = OverrideConfig(always_blast=True)

def _mk_for(pid, ctype, **params):
    return [{"commandType": ctype, "playerId": pid, "teamId": 0, "parameters": params}]

gsK1 = copy.deepcopy(GAME_STATE)
gsK1["ball"]["possessionAgentId"] = "agentId_1"
gsK1["ball"]["position"] = {"x": -35.0, "y": 5.0, "z": 0}
for p in gsK1["players"]:
    if p["teamCode"] == "home" and p["agentId"] == "agentId_1":
        p["position"] = {"x": -35, "y": 5}
out, tag = apply_overrides(_mk_for(1, "PASS", target_player_id=3, type="GROUND"),
                           gsK1, 0, 1, "DEF", BLAST)
assert tag in ("blast", "gk-clear", "gk-blast") and out[0]["commandType"] == "SHOOT", (tag, out)
assert out[0]["parameters"]["power"] >= 0.95, out
repK1 = tactics_report(gsK1, 0, 1, "DEF")
assert "BLAST" in repK1 and "SHOOT NOW" in repK1, repK1

gsK2 = copy.deepcopy(GAME_STATE)
gsK2["ball"]["possessionAgentId"] = "agentId_0"
gsK2["ball"]["position"] = {"x": -50.0, "y": 0.0, "z": 0}
out, tag = apply_overrides(_mk_for(0, "GK_DISTRIBUTE", target_player_id=3, method="KICK"),
                           gsK2, 0, 0, "GK", BLAST)
assert tag in ("gk-dist", "gk-launch", "gk-restart") and out[0]["commandType"] == "GK_DISTRIBUTE", (tag, out)
gsK2["playMode"] = "GOAL_KICK"
for p in gsK2["players"]:
    if p["teamCode"] == "away" and p["agentId"] == "agentId_1":
        p["position"] = {"x": -45, "y": 1}
out, tag = apply_overrides(_mk_for(0, "GK_DISTRIBUTE", target_player_id=4, method="KICK"),
                           gsK2, 0, 0, "GK", BLAST)
assert tag == "gk-blast" and out[0]["commandType"] == "SHOOT", (tag, out)

gsK3 = copy.deepcopy(GAME_STATE)
gsK3["ball"]["possessionAgentId"] = "agentId_3"
gsK3["ball"]["position"] = {"x": 7.0, "y": 0.0, "z": 0}
posK3 = {"agentId_0": (35, 10), "agentId_1": (-20, 10), "agentId_2": (30, 20),
         "agentId_3": (20, -15), "agentId_4": (40, -18)}
for p in gsK3["players"]:
    if p["teamCode"] == "home" and p["agentId"] == "agentId_3":
        p["position"] = {"x": 7, "y": 0}
    elif p["teamCode"] == "away":
        x, y = posK3[p["agentId"]]
        p["position"] = {"x": x, "y": y}
out, tag = apply_overrides(_mk_for(3, "MOVE_TO", target_x=20, target_y=0, sprint=True),
                           gsK3, 0, 3, "FWD1", OverrideConfig())
assert tag in ("shoot", "snap-shot") and out[0]["commandType"] == "SHOOT", (tag, out)
aim_k3 = out[0]["parameters"]["aim_location"]
assert out[0]["parameters"]["power"] == shot_power(_d_goal({"x": 7, "y": 0}), aim_k3)

gsK4 = copy.deepcopy(GAME_STATE)
gsK4["ball"]["possessionAgentId"] = "agentId_2"
gsK4["ball"]["position"] = {"x": -30.0, "y": 0.0, "z": 0}
posK4_home = {"agentId_0": (-50, 0), "agentId_1": (-40, -5), "agentId_2": (-30, 0),
              "agentId_3": (5, -10), "agentId_4": (10, 10)}
posK4_away = {"agentId_0": (50, 0), "agentId_1": (-20, 10), "agentId_2": (-25, 15),
              "agentId_3": (-5, 15), "agentId_4": (-15, -20)}
for p in gsK4["players"]:
    x, y = (posK4_home if p["teamCode"] == "home" else posK4_away)[p["agentId"]]
    p["position"] = {"x": x, "y": y}
assert "OPP HIGH PRESS" in summarize_state(gsK4, 0, 2, "MID")
out, tag = apply_overrides(_mk_for(2, "SHOOT", aim_location="CENTER", power=1.0),
                           gsK4, 0, 2, "MID", OverrideConfig())
assert tag == "counter" and out[0]["commandType"] == "PASS", (tag, out)
assert out[0]["parameters"]["target_player_id"] == 4, out
assert out[0]["parameters"]["type"] == "THROUGH", out

gsK4b = copy.deepcopy(GAME_STATE)
gsK4b["ball"]["possessionAgentId"] = "agentId_4"
gsK4b["ball"]["position"] = {"x": 12.0, "y": 14.0, "z": 0}
for p in gsK4b["players"]:
    if p["teamCode"] == "home" and p["agentId"] == "agentId_4":
        p["position"] = {"x": 12, "y": 14}
    elif p["teamCode"] == "away":
        p["position"] = {"x": 80, "y": 0}
out, tag = apply_overrides(_mk_for(4, "SHOOT", aim_location="CENTER", power=1.0),
                           gsK4b, 0, 4, "FWD2", OverrideConfig(longshot_max=58.0))
assert out[0]["commandType"] == "SHOOT", (tag, out)

out, tag = apply_overrides(_mk_for(4, "SHOOT", aim_location="CENTER", power=1.0),
                           gs, 0, 4, "FWD2", OV)
assert tag == "phantom" and out[0]["commandType"] in ("MARK", "MOVE_TO"), (tag, out)
out, tag = apply_overrides(_mk_for(3, "SHOOT", aim_location="CENTER", power=1.0),
                           gs, 0, 3, "FWD1", OV)
assert tag == "phantom" and out[0]["commandType"] == "PRESS_BALL", (tag, out)

BUILD = OverrideConfig(always_blast=True, build_from_back=True)

gsM = copy.deepcopy(GAME_STATE)
gsM["ball"]["possessionAgentId"] = "agentId_0"
gsM["ball"]["position"] = {"x": -50.0, "y": 0.0, "z": 0}

out, tag = apply_overrides(_mk_for(0, "GK_DISTRIBUTE", target_player_id=3, method="KICK"),
                           gsM, 0, 0, "GK", BUILD)
assert tag in ("gk-dist", "gk-launch") and out[0]["commandType"] == "GK_DISTRIBUTE", (tag, out)

gsM2 = copy.deepcopy(gsM)
for p in gsM2["players"]:
    if p["teamCode"] == "away" and p["agentId"] == "agentId_1":
        p["position"] = {"x": -45, "y": 2}
out, tag = apply_overrides(_mk_for(0, "GK_DISTRIBUTE", target_player_id=3, method="KICK"),
                           gsM2, 0, 0, "GK", BUILD)
assert tag == "gk-blast" and out[0]["commandType"] == "SHOOT", (tag, out)

gsM2b = copy.deepcopy(gsM)
posM2b = {"agentId_1": (-45, 2), "agentId_2": (-15, 7.5),
          "agentId_3": (-18, -2.5), "agentId_4": (-22.5, -4)}
for p in gsM2b["players"]:
    if p["teamCode"] == "away" and p["agentId"] in posM2b:
        x, y = posM2b[p["agentId"]]
        p["position"] = {"x": x, "y": y}
out, tag = apply_overrides(_mk_for(0, "GK_DISTRIBUTE", target_player_id=3, method="KICK"),
                           gsM2b, 0, 0, "GK", BUILD)
assert tag == "gk-blast" and out[0]["commandType"] == "SHOOT", (tag, out)

gsM3 = copy.deepcopy(gsM)
blockers = {"agentId_2": (-15, 7.5), "agentId_3": (-18, -2.5), "agentId_4": (-22.5, -4)}
for p in gsM3["players"]:
    if p["teamCode"] == "away" and p["agentId"] in blockers:
        x, y = blockers[p["agentId"]]
        p["position"] = {"x": x, "y": y}
out, tag = apply_overrides(_mk_for(0, "GK_DISTRIBUTE", target_player_id=3, method="KICK"),
                           gsM3, 0, 0, "GK", BUILD)
assert tag in ("gk-dist", "gk-launch", "gk-clear", "gk-restart-clear") and (
    out[0]["commandType"] in ("GK_DISTRIBUTE", "SHOOT")), (tag, out)

out, tag = apply_overrides(_mk_for(0, "GK_DISTRIBUTE", target_player_id=3, method="KICK"),
                           gsM, 0, 0, "GK", BLAST)
assert tag in ("gk-dist", "gk-launch") and out[0]["commandType"] == "GK_DISTRIBUTE", (tag, out)

gsM5 = copy.deepcopy(GAME_STATE)
gsM5["ball"]["possessionAgentId"] = "agentId_1"
gsM5["ball"]["position"] = {"x": -12.0, "y": 0.0, "z": 0}
for p in gsM5["players"]:
    if p["teamCode"] == "home" and p["agentId"] == "agentId_1":
        p["position"] = {"x": -12, "y": 0}
out, tag = apply_overrides(_mk_for(1, "PASS", target_player_id=0, type="GROUND"),
                           gsM5, 0, 1, "DEF", BUILD)
assert tag == "build" and out[0]["commandType"] == "PASS", (tag, out)
assert out[0]["parameters"]["target_player_id"] == 4, out
assert out[0]["parameters"]["type"] == "THROUGH", out

gsM6 = copy.deepcopy(gsM)
gsM6["playMode"] = "GOAL_KICK"
for p in gsM6["players"]:
    if p["teamCode"] == "away":
        p["position"] = {"x": 40, "y": p["position"].get("y", 0)}
out, tag = apply_overrides(_mk_for(0, "GK_DISTRIBUTE", target_player_id=3, method="KICK"),
                           gsM6, 0, 0, "GK", BUILD)
assert tag in ("gk-restart", "gk-launch", "gk-dist") and out[0]["commandType"] == "GK_DISTRIBUTE", (tag, out)

gsM7 = copy.deepcopy(gsM6)
for p in gsM7["players"]:
    if p["teamCode"] == "away" and p["agentId"] == "agentId_1":
        p["position"] = {"x": -45, "y": 1}
out, tag = apply_overrides(_mk_for(0, "GK_DISTRIBUTE", target_player_id=3, method="KICK"),
                           gsM7, 0, 0, "GK", BUILD)
assert tag == "gk-blast" and out[0]["commandType"] == "SHOOT", (tag, out)

out, tag = apply_overrides(_mk_for(0, "GK_DISTRIBUTE", target_player_id=3, method="THROW"),
                           gsM7, 0, 0, "GK", BUILD)
assert tag == "gk-blast" and out[0]["commandType"] == "SHOOT", (tag, out)


out, tag = apply_overrides(_mk_for(4, "MARK", target_player_id=1, tightness="TIGHT"),
                           gs2, 0, 4, "FWD2", OV)
assert tag == "support" and out[0]["commandType"] == "MOVE_TO", (tag, out)
assert out[0]["parameters"]["target_x"] == 42 and out[0]["parameters"]["target_y"] == 9.0, out

out, tag = apply_overrides(_mk_for(4, "MOVE_TO", target_x=5, target_y=15, sprint=False),
                           gs2, 0, 4, "FWD2", OV)
assert tag == "support" and out[0]["parameters"]["target_x"] == 42, (tag, out)

out, tag = apply_overrides(_mk_for(4, "MOVE_TO", target_x=40, target_y=8, sprint=True),
                           gs2, 0, 4, "FWD2", OV)
assert tag is None and out[0]["parameters"]["target_x"] == 40, (tag, out)

out, tag = apply_overrides(_mk_for(2, "SET_STANCE", stance=1),
                           gs2, 0, 2, "MID", OV)
assert tag == "support" and out[0]["commandType"] == "MOVE_TO", (tag, out)
assert out[0]["parameters"]["target_x"] == 33.0 and out[0]["parameters"]["target_y"] == -2.0, out

out, tag = apply_overrides(_mk_for(1, "MARK", target_player_id=1, tightness="TIGHT"),
                           gs2, 0, 1, "DEF", OV)
assert tag is None and out[0]["commandType"] == "MARK", (tag, out)


gsO = copy.deepcopy(GAME_STATE)
gsO["ball"]["possessionAgentId"] = "agentId_3"
gsO["ball"]["position"] = {"x": 38.0, "y": 0.0, "z": 0}
posO_home = {"agentId_3": (38, 0), "agentId_4": (20, 15), "agentId_2": (5, -8)}
posO_away = {"agentId_1": (44, 0)}
for p in gsO["players"]:
    if p["teamCode"] == "home" and p["agentId"] in posO_home:
        x, y = posO_home[p["agentId"]]
        p["position"] = {"x": x, "y": y}
    elif p["teamCode"] == "away":
        x, y = posO_away.get(p["agentId"], (50, -30))
        p["position"] = {"x": x, "y": y}
out, tag = apply_overrides(_mk("MOVE_TO", target_x=50, target_y=0, sprint=True),
                           gsO, 0, 3, "FWD1", OV)
assert tag == "shoot" and out[0]["commandType"] == "SHOOT", (tag, out)
assert out[0]["parameters"]["aim_location"] == "CENTER"

gsO2 = copy.deepcopy(gsO)
for p in gsO2["players"]:
    if p["teamCode"] == "home" and p["agentId"] == "agentId_4":
        p["position"] = {"x": 10, "y": 20}
out, tag = apply_overrides(_mk("SHOOT", aim_location="CENTER", power=1.0),
                           gsO2, 0, 3, "FWD1", OV)
assert tag in ("shoot", "aim") and out[0]["commandType"] == "SHOOT", (tag, out)

gsO2b = copy.deepcopy(gsO)
for p in gsO2b["players"]:
    if p["teamCode"] == "away":
        p["position"] = {"x": 42, "y": 0}
out, tag = apply_overrides(_mk("MOVE_TO", target_x=50, target_y=0, sprint=True),
                           gsO2b, 0, 3, "FWD1", OV)
assert tag == "shoot" and out[0]["commandType"] == "SHOOT", (tag, out)

gsO3 = copy.deepcopy(GAME_STATE)
gsO3["ball"]["possessionAgentId"] = "agentId_3"
gsO3["ball"]["position"] = {"x": -10.0, "y": 14.0, "z": 0}
for p in gsO3["players"]:
    if p["teamCode"] == "home" and p["agentId"] == "agentId_3":
        p["position"] = {"x": -10, "y": 14}
out, tag = apply_overrides(_mk("MOVE_TO", target_x=52, target_y=14, sprint=True),
                           gsO3, 0, 3, "FWD1", OV)
assert tag == "cap" and out[0]["parameters"]["target_x"] == 44.0, (tag, out)
assert out[0]["parameters"]["sprint"] is True, out
gsO3b = copy.deepcopy(gsO3)
for p in gsO3b["players"]:
    if p["teamCode"] == "home" and p["agentId"] == "agentId_3":
        p["position"] = {"x": 10, "y": 14}
    elif p["teamCode"] == "away":
        p["position"] = {"x": 80, "y": 0}
gsO3b["ball"]["position"] = {"x": 10.0, "y": 14.0, "z": 0}
out, tag = apply_overrides(_mk("MOVE_TO", target_x=12, target_y=14, sprint=True),
                           gsO3b, 0, 3, "FWD1", OV)
assert tag in ("shoot", "snap-shot") and out[0]["commandType"] == "SHOOT", (tag, out)
assert out[0]["parameters"]["aim_location"] == "CENTER", out
gsO3c = copy.deepcopy(GAME_STATE)
gsO3c["ball"]["possessionAgentId"] = "agentId_3"
gsO3c["ball"]["position"] = {"x": 0.0, "y": 0.0, "z": 0}
for p in gsO3c["players"]:
    if p["teamCode"] == "home" and p["agentId"] == "agentId_3":
        p["position"] = {"x": 0, "y": 0}
    elif p["teamCode"] == "away":
        p["position"] = {"x": 80, "y": 0}
out, tag = apply_overrides(_mk("MOVE_TO", target_x=30, target_y=0, sprint=True),
                           gsO3c, 0, 3, "FWD1", OverrideConfig(longshot_max=58.0))
assert tag in ("shoot", "snap-shot") and out[0]["commandType"] == "SHOOT", (tag, out)

gsO4 = copy.deepcopy(gs)
for p in gsO4["players"]:
    if p["teamCode"] == "home" and p["agentId"] == "agentId_3":
        p["position"] = {"x": 28, "y": -11}
out, tag = apply_overrides(_mk("PRESS_BALL", intensity=1.0), gsO4, 0, 3, "FWD1", OV)
assert tag == "tackle" and out[0]["commandType"] == "SLIDE_TACKLE", (tag, out)
assert out[0]["parameters"]["target_player_id"] == -1, out

gsO5 = copy.deepcopy(GAME_STATE)
gsO5["ball"]["possessionAgentId"] = None
gsO5["ball"]["isFree"] = True
out, tag = apply_overrides(_mk("MOVE_TO", target_x=15, target_y=-5, sprint=True),
                           gsO5, 0, 3, "FWD1", OV)
assert tag == "no-chase" and out[0]["commandType"] == "MARK", (tag, out)

gsO5b = copy.deepcopy(GAME_STATE)
gsO5b["ball"]["possessionAgentId"] = None
gsO5b["ball"]["isFree"] = True
gsO5b["ball"]["position"] = {"x": -47.0, "y": 2.0, "z": 0}
for p in gsO5b["players"]:
    if p["teamCode"] == "home" and p["agentId"] == "agentId_2":
        p["position"] = {"x": -44, "y": 1}
out, tag = apply_overrides(_mk("MOVE_TO", target_x=-46, target_y=2, sprint=True),
                           gsO5b, 0, 2, "MID", OV)
assert tag == "intercept" and out[0]["commandType"] == "INTERCEPT", (tag, out)

gsO6 = copy.deepcopy(GAME_STATE)
gsO6["ball"]["possessionAgentId"] = None
gsO6["ball"]["isFree"] = True
gsO6["ball"]["position"] = {"x": -48.0, "y": 2.0, "z": 0}
out, tag = apply_overrides(_mk_for(0, "SHOOT", aim_location="CENTER", power=1.0),
                           gsO6, 0, 0, "GK", BLAST)
assert tag == "gk-smother" and out[0]["commandType"] == "INTERCEPT", (tag, out)

out, tag = apply_overrides(_mk_for(0, "SHOOT", aim_location="CENTER", power=1.0),
                           gs, 0, 0, "GK", BLAST)
assert tag in ("gk-cover", "gk-line") and out[0]["commandType"] == "MOVE_TO", (tag, out)
assert out[0]["parameters"]["target_x"] == -52.8, out
assert out[0]["parameters"]["target_y"] == -6.0, out
assert out[0]["parameters"]["sprint"] is False, out

out, tag = apply_overrides(_mk_for(0, "MOVE_TO", target_x=-6, target_y=0, sprint=True),
                           gs, 0, 0, "GK", BLAST)
assert tag == "gk-line" and out[0]["parameters"]["target_x"] == -52.8, (tag, out)

gsO8 = copy.deepcopy(GAME_STATE)
gsO8["ball"]["possessionAgentId"] = "agentId_1"
gsO8["ball"]["position"] = {"x": -12.0, "y": 0.0, "z": 0}
posO8_away = {"agentId_1": (-1, -4), "agentId_2": (1, -2), "agentId_3": (4, 7)}
for p in gsO8["players"]:
    if p["teamCode"] == "home" and p["agentId"] == "agentId_1":
        p["position"] = {"x": -12, "y": 0}
    elif p["teamCode"] == "away" and p["agentId"] in posO8_away:
        x, y = posO8_away[p["agentId"]]
        p["position"] = {"x": x, "y": y}
out, tag = apply_overrides(_mk_for(1, "PASS", target_player_id=0, type="GROUND"),
                           gsO8, 0, 1, "DEF", BUILD)
assert tag == "launch" and out[0]["commandType"] == "PASS", (tag, out)
assert out[0]["parameters"]["target_player_id"] == 4, out
assert out[0]["parameters"]["type"] == "AERIAL", out

gsO9 = copy.deepcopy(GAME_STATE)
gsO9["ball"]["possessionAgentId"] = "agentId_1"
gsO9["ball"]["position"] = {"x": -30.0, "y": 0.0, "z": 0}
posO9_home = {"agentId_1": (-30, 0), "agentId_2": (-20, 6),
              "agentId_3": (12, -10), "agentId_4": (18, 12)}
posO9_away = {"agentId_1": (-28, 1), "agentId_2": (-18, 6),
              "agentId_3": (10, -9), "agentId_4": (16, 12)}
for p in gsO9["players"]:
    if p["teamCode"] == "home" and p["agentId"] in posO9_home:
        x, y = posO9_home[p["agentId"]]
        p["position"] = {"x": x, "y": y}
    elif p["teamCode"] == "away" and p["agentId"] in posO9_away:
        x, y = posO9_away[p["agentId"]]
        p["position"] = {"x": x, "y": y}
out, tag = apply_overrides(_mk_for(1, "PASS", target_player_id=0, type="GROUND"),
                           gsO9, 0, 1, "DEF", BUILD)
assert tag == "launch" and out[0]["commandType"] == "PASS", (tag, out)
assert out[0]["parameters"]["target_player_id"] == 4, out
assert out[0]["parameters"]["type"] == "AERIAL", out

gsO9b = copy.deepcopy(gsO9)
for p in gsO9b["players"]:
    if p["teamCode"] == "home" and p["agentId"] in ("agentId_2", "agentId_3", "agentId_4"):
        p["position"] = {"x": -34, "y": p["position"]["y"]}
out, tag = apply_overrides(_mk_for(1, "PASS", target_player_id=0, type="GROUND"),
                           gsO9b, 0, 1, "DEF", BUILD)
assert tag in ("blast", "gk-clear", "gk-blast") and out[0]["commandType"] == "SHOOT", (tag, out)

from tuning import apply_tuning, clamp, get as tuning_get

base_cfg = OverrideConfig()
tunedL = apply_tuning(base_cfg, "MID", {
    "global": {"shoot_threshold": 47.0, "press_bodies": 2},
    "MID": {"shoot_threshold": 99.0},
})
assert tunedL.shoot_threshold == 50.0, tunedL.shoot_threshold
assert tunedL.press_bodies == 2 and isinstance(tunedL.press_bodies, int), tunedL
assert base_cfg.shoot_threshold == 45.0, "original config must not mutate"
tunedL2 = apply_tuning(base_cfg, "GK", {"global": {"not_a_field": 1}})
assert tunedL2.shoot_threshold == base_cfg.shoot_threshold
assert apply_tuning(base_cfg, "DEF", {}) is base_cfg
assert apply_tuning(None, "DEF", {"global": {"press_bodies": 2}}) is None
assert clamp("press_bodies", 10) == 4 and clamp("gk_out_dist", 1.0) == 8.0
assert tuning_get("longshot_max", 52.0, {"global": {"longshot_max": 100}}) == 58.0
tunedM = apply_tuning(OverrideConfig(always_blast=True), "GK",
                      {"global": {"build_from_back": True, "blast_pressure_dist": 99}})
assert tunedM.build_from_back is True, tunedM
assert tunedM.blast_pressure_dist == 16.0, tunedM.blast_pressure_dist

print("Scenario A (away P3 holds): home view OPP / away view MY — OK")
print("Scenario B (home P3 holds): hasBall=True + SHOOT NOW CENTER 1.0 — OK")
print("Scenario C (opp GK holds): our GK hasBall=False — OK")
print("Scenario D (coach/free-ball/stamina/lastAction enrichment) — OK")
print("Scenario E (chase/press assignments, one player to the ball) — OK")
print("Scenario F (late-game GAME PLAN by score) — OK")
print("Scenario G (command guardrails clamp bad params) — OK")
print("Scenario H (LANE CLEAR / BLOCKED shot check with sidestep hint) — OK")
print("Scenario I (point-blank always shoots even with slight overlap) — OK")
print("Scenario J (tactical overrides: forced shot / no-chase / anchor / support) — OK")
print("Scenario K (GK/DEF blast, long shot, counter outlet, phantom fix) — OK")
print("Scenario M (build-from-back: outlet when safe, blast when pressed) — OK")
print("Scenario N (attack support: FWD/MID stretch wide instead of marking) — OK")
print("Scenario O (cutback/sidestep/carry-cap/tackle/intercept/GK smother) — OK")
print("Scenario L (tuning.json overlay: merge, clamp, immutability) — OK")
from fallback import build_fallback, FWD1_CONFIG
fb = build_fallback(FWD1_CONFIG)
gsQ = copy.deepcopy(GAME_STATE)
gsQ["ball"]["possessionAgentId"] = "agentId_3"
gsQ["ball"]["position"] = {"x": 7.0, "y": -14.0, "z": 0}
for p in gsQ["players"]:
    if p["teamCode"] == "home" and p["agentId"] == "agentId_3":
        p["position"] = {"x": 7, "y": -14}
    elif p["teamCode"] == "away":
        p["position"] = {"x": 80, "y": 0}
outQ = fb(gsQ, 0, 3)
assert outQ[0]["commandType"] == "SHOOT", outQ
assert outQ[0]["parameters"]["aim_location"] == "CENTER"
assert outQ[0]["parameters"]["power"] >= 0.95, outQ

from tactics import _best_shot_aim, _aim_geometry_line
aim_open, _, perp_open = _best_shot_aim({"x": 30, "y": -14}, 55, [])
assert aim_open == "CENTER" and perp_open > 0, aim_open
blocked, _, perp_blk = _best_shot_aim({"x": 30, "y": -14}, 55,
    [{"position": {"x": 52, "y": 0}}])
assert blocked == "CENTER" and perp_blk < 2.5, (blocked, perp_blk)
assert "open space" in _aim_geometry_line({"x": 30, "y": -14}, 55, "CENTER")

print("Scenario P (carry open then SHOOT CENTER) — OK")
print("Scenario Q (fallback: long-range wing -> geometry SHOOT) — OK")

gsR = copy.deepcopy(gs)
for p in gsR["players"]:
    if p["teamCode"] == "home" and p["agentId"] == "agentId_2":
        p["position"] = {"x": 25, "y": -11}
    if p["teamCode"] == "home" and p["agentId"] == "agentId_4":
        p["position"] = {"x": 18, "y": -5}
    if p["teamCode"] == "away" and p["agentId"] == "agentId_1":
        p["position"] = {"x": 25, "y": -8}
outR, tagR = apply_overrides(
    [{"commandType": "MOVE_TO", "playerId": 4, "teamId": 0,
      "parameters": {"target_x": 40, "target_y": 15, "sprint": True}}],
    gsR, 0, 4, "FWD2", OverrideConfig(mark_radius=28.0))
assert tagR in ("mark-near", "anchor", "no-chase"), (tagR, outR)
assert outR[0]["commandType"] == "MARK", (tagR, outR)
assert outR[0]["parameters"]["tightness"] == "TIGHT", outR

for p in gsR["players"]:
    if p["teamCode"] == "home" and p["agentId"] == "agentId_2":
        p["position"] = {"x": 29, "y": -12}
    if p["teamCode"] == "home" and p["agentId"] == "agentId_4":
        p["position"] = {"x": 28, "y": -10}
outR2, tagR2 = apply_overrides(
    [{"commandType": "MOVE_TO", "playerId": 4, "teamId": 0,
      "parameters": {"target_x": 0, "target_y": 0, "sprint": False}}],
    gsR, 0, 4, "FWD2", OverrideConfig(press_radius=20.0))
assert tagR2 in ("mark-near", "no-chase", "anchor"), (tagR2, outR2)
assert outR2[0]["commandType"] in ("MARK", "PRESS_BALL", "SLIDE_TACKLE"), (tagR2, outR2)

outS, tagS = apply_overrides(_mk_for(1, "MARK", target_player_id=1, tightness="TIGHT"),
                             gs, 0, 1, "DEF", OV)
assert tagS in ("hold-line", "anchor") and outS[0]["commandType"] == "MOVE_TO", (tagS, outS)
assert -43 <= outS[0]["parameters"]["target_x"] <= -27, outS

outS2, tagS2 = apply_overrides(_mk_for(2, "MARK", target_player_id=1, tightness="TIGHT"),
                               gs, 0, 2, "MID", OV)
assert tagS2 in ("hold-line", "anchor") and outS2[0]["commandType"] == "MOVE_TO", (tagS2, outS2)
assert -33 <= outS2[0]["parameters"]["target_x"] <= -17, outS2

print("Scenario R (compact defense: mark near / press carrier) — OK")
gsT = copy.deepcopy(GAME_STATE)
gsT["ball"]["possessionAgentId"] = None
gsT["ball"]["isFree"] = True
gsT["ball"]["position"] = {"x": 25.0, "y": -5.0, "z": 0}
outT, tagT = apply_overrides(_mk("MOVE_TO", target_x=45, target_y=10, sprint=True),
                             gsT, 0, 2, "MID", OV)
assert tagT in ("no-chase", "anchor", "hold-line", "mark-near") and outT[0]["commandType"] == "MOVE_TO", (tagT, outT)
assert outT[0]["parameters"]["target_x"] <= -17, outT

print("Scenario S (back line: DEF/MID hold depth, no x≈-6 collapse) — OK")
print("Scenario T (no baseline chase on loose ball in opponent half) — OK")

import tuning as _tuning_u
from state import coach_context as _cc
gsU = copy.deepcopy(GAME_STATE)
gsU["teamChat"] = [{"message": "home team stand still", "teamCode": "away"}]
assert _cc(gsU, 0) == ("", False), _cc(gsU, 0)
assert "stand still" not in summarize_state(gsU, 0, 3, "FWD1")
gsU["teamChat"] = [{"message": "press high", "teamCode": "HOME"}]
assert _cc(gsU, 0) == ("press high", True), _cc(gsU, 0)
assert "COACH ORDER from YOUR bench" in summarize_state(gsU, 0, 3, "FWD1")
gsU["teamChat"] = ["press high"]
assert _cc(gsU, 0) == ("press high", False)
gsU["teamChat"] = [{"senderTeamCode": "home", "message": "attack", "timestamp": 0}]
assert _cc(gsU, 0) == ("attack", True), _cc(gsU, 0)
gsU["teamChat"] = [{"senderTeamCode": "away", "message": "stand still", "timestamp": 0}]
assert _cc(gsU, 0) == ("", False), _cc(gsU, 0)
_old = dict(_tuning_u._DATA.get("global") or {})
_tuning_u._DATA.setdefault("global", {})["coach_key"] = "RAYO"
try:
    gsU["teamChat"] = ["home team stand still", "rayo: Salomon shoot now"]
    assert _cc(gsU, 0) == ("Salomon shoot now", True), _cc(gsU, 0)
    gsU["teamChat"] = ["home team stand still"]
    assert _cc(gsU, 0) == ("", False)
finally:
    _tuning_u._DATA["global"] = _old
_tuning_u._DATA.setdefault("global", {})["player_names"] = {"0": "Iker", "3": "Salomon"}
try:
    gsU["teamChat"] = [{"message": "Salomon shoot now", "teamCode": "home"}]
    s3 = summarize_state(gsU, 0, 3, "FWD1")
    assert "YOU are Salomon" in s3 and "P0=Iker" in s3, s3
finally:
    _tuning_u._DATA["global"] = _old
gsU = copy.deepcopy(GAME_STATE)
for p in gsU["players"]:
    p["teamCode"] = p["teamCode"].upper()
assert ">>> YOUR PLAYER (FWD1, id=3)" in summarize_state(gsU, 0, 3, "FWD1")
print("Scenario U (sideline security + uppercase teamCode) — OK")

import stamina as _stam
gsV = copy.deepcopy(GAME_STATE)
def _set(gs, team, idx, **kw):
    for p in gs["players"]:
        if p["teamCode"] == team and p["agentId"] == f"agentId_{idx}":
            p.update(kw)
_set(gsV, "home", 1, stamina=0.20)
cmds = [{"commandType": "MOVE_TO", "parameters": {"target_x": 0, "target_y": 0, "sprint": True}}]
out, tag = _stam.guard(copy.deepcopy(cmds), gsV, 0, 1)
assert tag == "stam" and out[0]["parameters"]["sprint"] is False, out
out, tag = _stam.guard([{"commandType": "PRESS_BALL", "parameters": {"intensity": 1.0}}], gsV, 0, 1)
assert tag == "stam" and out[0]["parameters"]["intensity"] == 0.5, out
_set(gsV, "home", 1, stamina=80)
out, tag = _stam.guard(copy.deepcopy(cmds), gsV, 0, 1)
assert tag is None and out[0]["parameters"]["sprint"] is True
_set(gsV, "home", 1, stamina=20)
assert _stam.guard(copy.deepcopy(cmds), gsV, 0, 1)[1] == "stam"
_set(gsV, "home", 3, stamina=0.10)
assert _stam.guard(copy.deepcopy(cmds), gsV, 0, 3)[1] is None
_set(gsV, "home", 0, stamina=100)
gk_cmd = [{"commandType": "MOVE_TO", "parameters": {"target_x": -42, "target_y": 8, "sprint": True}}]
out, tag = _stam.guard(copy.deepcopy(gk_cmd), gsV, 0, 0)
assert tag == "gk-walk" and out[0]["parameters"]["sprint"] is False, (out, tag)
gsN = copy.deepcopy(gsV); gsN["ball"]["position"] = {"x": -40, "y": 2}
out, tag = _stam.guard(copy.deepcopy(gk_cmd), gsN, 0, 0)
assert tag is None and out[0]["parameters"]["sprint"] is True, (out, tag)
_set(gsV, "home", 1, stamina=100)
hop = [{"commandType": "MOVE_TO", "parameters": {"target_x": -8, "target_y": 10, "sprint": True}}]
out, tag = _stam.guard(copy.deepcopy(hop), gsV, 0, 1)
assert tag == "short" and out[0]["parameters"]["sprint"] is False, (out, tag)
print("Scenario V (stamina guard) — OK")

import match_memory as _mm
import ev_policy as _evp
_mm.reset()
def _gsW(t, us, them, mode="OPEN_PLAY"):
    g = copy.deepcopy(GAME_STATE)
    g["gameTime"], g["score"], g["playMode"] = t, {"home": us, "away": them}, mode
    return g
SH = [{"commandType": "SHOOT", "parameters": {"aim_location": "CENTER", "power": 1.0}}]
assert _mm.long_shot_multiplier(_gsW(10, 0, 0), 0, 4, 30) == (1.0, None)
assert _mm.long_shot_multiplier(_gsW(10, 0, 0), 0, 4, 55) == (1.0, None)
m, why = _mm.long_shot_multiplier(_gsW(10, 1, 0), 0, 4, 55)
assert why == "lead" and m < 1.0, (m, why)
assert _mm.long_shot_multiplier(_gsW(10, 1, 0), 0, 4, 30)[0] == 1.0
t = 10.0
for k in range(3):
    _mm.observe(_gsW(t, 0, 0), 0, 4)
    _mm.record(SH, _gsW(t, 0, 0), 0, 4, 55, True)
    _mm.record(SH, _gsW(t + 1, 0, 0), 0, 4, 55, True)
    t += 6.0
    _mm.observe(_gsW(t, 0, 0), 0, 4)
assert _mm.snapshot(0, 4) == {"shots": 3, "goals": 0, "misses": 3}, _mm.snapshot(0, 4)
m, why = _mm.long_shot_multiplier(_gsW(t, 0, 0), 0, 4, 55)
assert why == "mem" and m < 1.0, (m, why)
_mm.record(SH, _gsW(t, 0, 0, "RESETTING"), 0, 4, 56, True)
assert _mm.snapshot(0, 4)["shots"] == 3
_mm.record(SH, _gsW(t, 0, 0), 0, 4, 55, True)
_mm.observe(_gsW(t + 2, 1, 0), 0, 4)
assert _mm.snapshot(0, 4)["goals"] == 1
assert _mm.long_shot_multiplier(_gsW(t + 2, 0, 0), 0, 4, 55)[1] == "kickoff"
assert _mm.long_shot_multiplier(_gsW(t + 11, 1, 0), 0, 4, 55)[1] == "lead"
_mm.observe(_gsW(1, 0, 0), 0, 4)
assert _mm.snapshot(0, 4) == {"shots": 0, "goals": 0, "misses": 0}
gW = copy.deepcopy(GAME_STATE)
for p in gW["players"]:
    if p["teamCode"] == "home" and p["agentId"] == "agentId_4":
        p["position"] = {"x": 2, "y": 0}
gW["ball"]["position"] = {"x": 2, "y": 0}; gW["ball"]["possessionAgentId"] = "agentId_4"
_mm.reset()
gW["score"] = {"home": 0, "away": 0}
ctx = _evp._Ctx(gW, 0, 4, _evp.MODEL)
level = [o[0] for o in ctx.options(biased=True) if o[1] == "shoot"][0]
gW["score"] = {"home": 1, "away": 0}
ctx = _evp._Ctx(gW, 0, 4, _evp.MODEL)
ahead = [o[0] for o in ctx.options(biased=True) if o[1] == "shoot"][0]
assert ahead < level and ctx.mem_why == "lead", (level, ahead, ctx.mem_why)
_mm.reset()
_mm.reset()
_mm.observe(_gsW(0, 0, 0), 0, 4)
assert _mm.long_shot_multiplier(_gsW(5, 0, 0), 0, 4, 55) == (0.0, "kickoff")
assert _mm.long_shot_multiplier(_gsW(5, 0, 0), 0, 4, 30) == (1.0, None)
_mm.observe(_gsW(9, 0, 0), 0, 4)
assert _mm.long_shot_multiplier(_gsW(9, 0, 0), 0, 4, 55) == (1.0, None)
_mm.observe(_gsW(40, 0, 1), 0, 4)
assert _mm.long_shot_multiplier(_gsW(44, 0, 1), 0, 4, 55)[1] == "kickoff"
assert _mm.long_shot_multiplier(_gsW(49, 0, 1), 0, 4, 55)[1] is None
_mm.reset()
print("Scenario W (match memory: lead mode + long-shot misses + kickoff window) — OK")
from overrides import _anchor, _support_spot
_g = _tuning_base._DATA["global"]
ball0 = {"x": 40, "y": 0}
def_base, mid_base = _anchor("DEF", 0, ball0)[0], _anchor("MID", 0, ball0)[0]
_g.update({"line_push_DEF": 12, "line_push_MID": 8})
assert abs(_anchor("DEF", 0, ball0)[0] - (def_base + 12)) < 1e-6, _anchor("DEF", 0, ball0)
assert abs(_anchor("MID", 0, ball0)[0] - (mid_base + 8)) < 1e-6
assert abs(_anchor("DEF", 1, {"x": -40, "y": 0})[0] - (-def_base - 12)) < 1e-6
assert _anchor("MID", 0, ball0, lane_y=9.0)[1] == 9.0 and _anchor("MID", 0, ball0, lane_y=-9.0)[1] == -9.0
assert _support_spot("MID", 0, ball0, 9.0)[1] == 9.0 and _support_spot("MID", 0, ball0, -9.0)[1] == -9.0
import ev_policy as _evpY, match_memory as _mmY
_mmY.reset()
gY = copy.deepcopy(GAME_STATE); gY["score"] = {"home": 0, "away": 0}
for p in gY["players"]:
    if p["teamCode"] == "home" and p["agentId"] == "agentId_4":
        p["position"] = {"x": 20, "y": 3}
gY["ball"]["position"] = {"x": 20, "y": 3}; gY["ball"]["possessionAgentId"] = "agentId_4"
_g["ev_shot_bias"] = 1.0
o1 = {l: v for v, l, _ in _evpY._Ctx(gY, 0, 4, _evpY.MODEL).options(biased=True)}
_g["ev_shot_bias"] = 1.5
o2 = {l: v for v, l, _ in _evpY._Ctx(gY, 0, 4, _evpY.MODEL).options(biased=True)}
assert abs(o2["shoot"] - 1.5 * o1["shoot"]) < 1e-9 and all(o2[k] == o1[k] for k in o1 if k != "shoot")
_g.update({"line_push_DEF": 0, "line_push_MID": 0, "ev_shot_bias": 1.0})
from parsing import parse_commands as _pc
for raw, want in (('"P1"', 1), ('"GK"', 0), ('"3"', 3), ('2.0', 2), ('4', 4)):
    c = _pc('[{"commandType":"MARK","playerId":1,"parameters":{"target_player_id":%s,"tightness":"TIGHT"}}]' % raw, 0, 1)
    assert c and c[0]["parameters"]["target_player_id"] == want, (raw, c)
print("Scenario Y (v7: line push, midfield lanes, shot bias, int targets) — OK")
print("ALL LIB TESTS PASSED")

"""
AI Soccer LEFT Midfielder Agent (1-2-1) — Controls ONLY player 3.
Uses Strands SDK + Amazon Nova Micro (fastest model, latency-optimized).
"""

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lib"))
from _bootstrap import setup_lib_path; setup_lib_path(__file__)

from dataclasses import replace
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from agent_base import create_agent, create_invoke_handler
from fallback import build_fallback, MID_CONFIG
from overrides import OverrideConfig

app = BedrockAgentCoreApp()

MY_PLAYER_ID = 3
POSITION_LABEL = "MID"


SYSTEM_PROMPT = f"""Ultra-aggressive attacking midfielder AI, LEFT side of a 1-2-1 (two midfielders). You control ONLY player {MY_PLAYER_ID} (MID) in 5v5 soccer. Each tick: read state, reply ONE command.

RULE #1 — SHOOT CENTER. Read TACTICS Shot line:
- "LANE CLEAR" / "POINT-BLANK" / "LANE BLOCKED": SHOOT CENTER power 1.0.
- Never dribble for a better angle inside 45m — shoot immediately.
- "out of range": carry forward, never blast from own half.

RULE #2 — DEFENSE (read ASSIGNMENT / DEFEND / SHAPE every tick):
- You are the presser: PRESS_BALL 1.0; within 5m SLIDE_TACKLE.
- Teammate presses: stay in the mid block on the LEFT (y ≈ -9) with MOVE_TO (x ≈ my_goal_x*0.45 to *0.30) — screen in
  front of DEF, do NOT MARK deep in opponent half.
- Ball within 35m of our goal: MARK TIGHT or tackle the carrier.
- OPP HIGH PRESS: drop deeper, you are the OUTLET on the counter.

TACTICS (priority order):
1. hasBall=True and distOppGoal<=45: SHOOT CENTER power 1.0 (always).
2. hasBall=True and distOppGoal>45: MOVE_TO top of box (x = opp_goal_x*0.7, y = 0), sprint.
3. Defending: keep your line (MOVE_TO or MARK) — never MOVE_TO toward the opponent byline.
4. Teammate FWD has ball: trail at D of box for layoff/rebound.
5. Else attacking: advanced left half-space (x = opp_goal_x*0.4, y = -9), sprint.
6. Long shots are welcome: with a clear sight of goal from distance, SHOOT power 1.0.

RESPECT ASSIGNMENT exactly. Never PASS backward.

COMMANDS: MOVE_TO(target_x,target_y,sprint) | PASS(target_player_id,type=GROUND|AERIAL|THROUGH) | SHOOT(aim_location=TL|TR|BL|BR|CENTER,power) | PRESS_BALL(intensity) | INTERCEPT(aggressive) | SLIDE_TACKLE(target_player_id,sprint,distance) | MARK(target_player_id,tightness=LOOSE|TIGHT) | SET_STANCE(stance 0-2)
commandType must be EXACTLY one of the COMMANDS names above — never invent one.

FIELD: kickoff (0,0). x: -55 own-goal-line to +55 opp-goal-line. y: -35 bottom to +35 top. Team 0 defends x=-55 and attacks +x; Team 1 defends x=+55 and attacks -x.

Reply ONLY the JSON array, no other text:
[{{"commandType":"SHOOT","playerId":{MY_PLAYER_ID},"parameters":{{"aim_location":"CENTER","power":1.0}},"duration":0}}]"""


AGG_MID_CONFIG = replace(
    MID_CONFIG,
    press_only_if_designated=True,
    press_distance=10.0,
    off_ball_action="MARK",
)
fallback_commands = build_fallback(AGG_MID_CONFIG)

OVERRIDE_CONFIG = OverrideConfig(lane_y=-9.0)


agent = create_agent(SYSTEM_PROMPT, model_id="us.amazon.nova-2-lite-v1:0")
create_invoke_handler(
    app, agent, MY_PLAYER_ID, POSITION_LABEL, fallback_commands,
    fallback_cfg=AGG_MID_CONFIG,
    override_cfg=OVERRIDE_CONFIG,
    log_label="MID2",
)

if __name__ == "__main__":
    app.run()

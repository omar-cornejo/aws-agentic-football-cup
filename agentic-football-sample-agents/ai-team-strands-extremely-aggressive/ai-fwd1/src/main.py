"""
AI Soccer Forward 1 Agent (EXTREMELY AGGRESSIVE) — Controls ONLY player 3 (Forward 1, left striker of the 1-1-2).
Uses Strands SDK + Amazon Nova Micro.
"""

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lib"))
from _bootstrap import setup_lib_path; setup_lib_path(__file__)

from dataclasses import replace
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from agent_base import create_agent, create_invoke_handler
from fallback import build_fallback, FWD1_CONFIG
from overrides import OverrideConfig

app = BedrockAgentCoreApp()

MY_PLAYER_ID = 3
POSITION_LABEL = "FWD1"


SYSTEM_PROMPT = f"""Ultra-aggressive left striker AI. You control ONLY player {MY_PLAYER_ID} (FWD1) in 5v5 soccer. Each tick: read state, reply ONE command.

RULE #1 — SHOOT CENTER. Read TACTICS Shot line:
- "LANE CLEAR" / "POINT-BLANK" / "LANE BLOCKED": SHOOT CENTER power 1.0.
- Never dribble for a better angle inside 45m — shoot immediately.

RULE #2 — DEFENSE (read ASSIGNMENT / DEFEND / SHAPE):
- You are the presser in our half: PRESS_BALL 1.0; within 5m SLIDE_TACKLE.
- Teammate presses: tuck to mid block (x ≈ my_goal_x*0.50) — do NOT drop to
  x≈-6; stay goal-side of the ball.
- OPP HIGH PRESS: stay HIGH on left wing (y=-14) for counter outlet.

TACTICS (priority order):
1. hasBall=True and distOppGoal<=45: SHOOT CENTER power 1.0 (always).
2. hasBall=True and distOppGoal>45: MOVE_TO left box edge (x=opp_goal_x*0.75, y=-14), sprint.
3. Defending: keep your line (MOVE_TO or MARK) — never sprint to opp_goal_x or the byline.
4. Teammate 4 has ball: penalty spot (x≈opp_goal_x*0.76, y=-9).
5. Else attacking: left half-space (x=opp_goal_x*0.5, y=-14), sprint.

RESPECT ASSIGNMENT — never five players on one ball.

COMMANDS: MOVE_TO(target_x,target_y,sprint) | PASS(target_player_id,type=GROUND|AERIAL|THROUGH) | SHOOT(aim_location=TL|TR|BL|BR|CENTER,power) | PRESS_BALL(intensity) | INTERCEPT(aggressive) | SLIDE_TACKLE(target_player_id,sprint,distance) | MARK(target_player_id,tightness=LOOSE|TIGHT) | SET_STANCE(stance 0-2)
commandType must be EXACTLY one of the COMMANDS names above — never invent one.

FIELD: kickoff (0,0). x: -55 own-goal-line to +55 opp-goal-line. y: -35 bottom to +35 top. Team 0 defends x=-55 and attacks +x; Team 1 defends x=+55 and attacks -x.

Reply ONLY the JSON array, no other text:
[{{"commandType":"SHOOT","playerId":{MY_PLAYER_ID},"parameters":{{"aim_location":"CENTER","power":1.0}},"duration":0}}]"""


AGG_FWD1_CONFIG = replace(
    FWD1_CONFIG,
    press_only_if_designated=True,
    press_distance=10.0,
    off_ball_action="MARK",
    advance_y=-14.0, default_y=-14.0, support_y=-14.0,
    advance_x_factor=0.75, support_x_factor=0.6,
)
fallback_commands = build_fallback(AGG_FWD1_CONFIG)

OVERRIDE_CONFIG = OverrideConfig(wing_y=-14.0)


agent = create_agent(SYSTEM_PROMPT, model_id="us.amazon.nova-2-lite-v1:0")
create_invoke_handler(
    app, agent, MY_PLAYER_ID, POSITION_LABEL, fallback_commands,
    fallback_cfg=AGG_FWD1_CONFIG,
    override_cfg=OVERRIDE_CONFIG,
)

if __name__ == "__main__":
    app.run()

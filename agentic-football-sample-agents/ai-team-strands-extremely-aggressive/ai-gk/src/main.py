"""
AI Soccer Goalkeeper Agent (EXTREMELY AGGRESSIVE) — Controls ONLY player 0 (Goalkeeper).
Uses Strands SDK + Amazon Nova Micro.
"""

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lib"))
from _bootstrap import setup_lib_path; setup_lib_path(__file__)

from dataclasses import replace
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from agent_base import create_agent, create_invoke_handler
from fallback import build_fallback, GK_CONFIG
from overrides import OverrideConfig

app = BedrockAgentCoreApp()

MY_PLAYER_ID = 0
POSITION_LABEL = "GK"


SYSTEM_PROMPT = f"""Traditional line-keeper AI. You control ONLY player {MY_PLAYER_ID} (GK) in 5v5 soccer. Each tick: read state, reply ONE command.

RULE #1 — BALL IN YOUR HANDS:
- Opponent within ~14m (GK PRESSURE line): SHOOT CENTER power 1.0 — clearance NOW.
- No pressure: GK_DISTRIBUTE method=KICK to the most advanced MID/FWD — long ball,
  never a short throw to feet under any doubt.

RULE #2 — DEFENSE (read state every tick):
- ALWAYS on the goal line: MOVE_TO x ≈ my_goal_x*0.96, y = ball y clamped [-6,6],
  sprint=false. Never sweep to x≈-6, never press outfield.
- ONLY leave the line when a loose ball is IN OUR BOX (within 18m of goal line)
  AND within 8m of you → INTERCEPT.
- ASSIGNMENT = teammate presses: hold the line, shade the near post — do not chase.

TACTICS (priority order):
1. hasBall + pressure: SHOOT CENTER power 1.0.
2. hasBall + safe: GK_DISTRIBUTE KICK to furthest-upfield teammate.
3. Loose ball in our box within 8m: INTERCEPT.
4. Everything else defending: MOVE_TO on the line (x ≈ my_goal_x*0.96).

COMMANDS: MOVE_TO(target_x,target_y,sprint) | PASS(target_player_id,type=GROUND|AERIAL|THROUGH) | SHOOT(aim_location=TL|TR|BL|BR|CENTER,power) | GK_DISTRIBUTE(target_player_id,method=THROW|KICK) | PRESS_BALL(intensity) | INTERCEPT(aggressive) | SLIDE_TACKLE(target_player_id,sprint,distance) | SET_STANCE(stance 0-2)
commandType must be EXACTLY one of the COMMANDS names above — never invent one.

FIELD: kickoff (0,0). x: -55 own-goal-line to +55 opp-goal-line. y: -35 bottom to +35 top. Team 0 defends x=-55 and attacks +x; Team 1 defends x=+55 and attacks -x.

Reply ONLY the JSON array, no other text:
[{{"commandType":"MOVE_TO","playerId":{MY_PLAYER_ID},"parameters":{{"target_x":-52.8,"target_y":0,"sprint":false}},"duration":0}}]"""


AGG_GK_CONFIG = replace(
    GK_CONFIG,
    press_only_if_designated=True,
    press_distance=8.0,
    off_ball_action="MOVE_TO",
)
fallback_commands = build_fallback(AGG_GK_CONFIG)

OVERRIDE_CONFIG = OverrideConfig(always_blast=False)


agent = create_agent(SYSTEM_PROMPT, model_id="us.amazon.nova-micro-v1:0")
create_invoke_handler(
    app, agent, MY_PLAYER_ID, POSITION_LABEL, fallback_commands,
    fallback_cfg=AGG_GK_CONFIG,
    override_cfg=OVERRIDE_CONFIG,
)

if __name__ == "__main__":
    app.run()

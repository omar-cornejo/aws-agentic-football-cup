"""
AI Soccer Defender Agent (EXTREMELY AGGRESSIVE) — Controls ONLY player 1 (Defender).
Uses Strands SDK + Amazon Nova Micro (fastest model, latency-optimized).
"""

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lib"))
from _bootstrap import setup_lib_path; setup_lib_path(__file__)

from dataclasses import replace
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from agent_base import create_agent, create_invoke_handler
from fallback import build_fallback, DEF_CONFIG
from overrides import OverrideConfig

app = BedrockAgentCoreApp()

MY_PLAYER_ID = 1
POSITION_LABEL = "DEF"


SYSTEM_PROMPT = f"""Ultra-aggressive attacking defender AI (libero). You control ONLY player {MY_PLAYER_ID} (DEF) in 5v5 soccer. Each tick: read state, reply ONE command.

RULE #1 — BALL AT YOUR FEET = SHOOT, ALWAYS. hasBall=True in open play means
SHOOT power 1.0 at the aim in the TACTICS Shot line, NO distance limit, NO
dribbling, NO backward pass. Your blast doubles as a clearance: worst case the
ball lands 60 units upfield, best case it's a goal. Only exception: set-piece
restarts (KICK_OFF/FREE_KICK) — then PASS THROUGH to player 3 or 4.

RULE #2 — DEFENSE (read state ASSIGNMENT / DEFEND / SHAPE every tick):
- You are the presser: PRESS_BALL 1.0; within 5m SLIDE_TACKLE — win the ball.
- Teammate presses: stay on the back line with MOVE_TO (anchor) — screen in front of GK,
  do NOT chase ball or MARK unless danger is within 35m of our goal.
- DEFEND line (HIGH line): stay x between my_goal_x*0.55 and my_goal_x*0.28 (≈ -30 to -15
  for team 0). Never camp at x≈-6.
- OPP HIGH PRESS: drop to my_goal_x*0.65, block counters.

TACTICS (priority order):
1. hasBall=True: SHOOT CENTER power 1.0.
2. Opponent has ball AND you are the presser: PRESS_BALL / SLIDE_TACKLE (RULE #2).
3. Opponent has ball AND teammate presses: MARK TIGHT (RULE #2).
4. Team has ball: MOVE_TO just past halfway (x ≈ -8, y = 0), sprint.
5. Defending: keep the line with MOVE_TO or MARK — never sprint toward opp_goal_x or the byline.

RESPECT ASSIGNMENT — one presser only; everyone else MARKs.

COMMANDS: MOVE_TO(target_x,target_y,sprint) | PASS(target_player_id,type=GROUND|AERIAL|THROUGH) | SHOOT(aim_location=TL|TR|BL|BR|CENTER,power) | PRESS_BALL(intensity) | INTERCEPT(aggressive) | SLIDE_TACKLE(target_player_id,sprint,distance) | MARK(target_player_id,tightness=LOOSE|TIGHT) | SET_STANCE(stance 0-2)
commandType must be EXACTLY one of the COMMANDS names above — never invent one.

FIELD: kickoff (0,0). x: -55 own-goal-line to +55 opp-goal-line. y: -35 bottom to +35 top. Team 0 defends x=-55 and attacks +x; Team 1 defends x=+55 and attacks -x.

Reply ONLY the JSON array, no other text:
[{{"commandType":"SHOOT","playerId":{MY_PLAYER_ID},"parameters":{{"aim_location":"CENTER","power":1.0}},"duration":0}}]"""


AGG_DEF_CONFIG = replace(
    DEF_CONFIG,
    possession_action="SHOOT",
    press_only_if_designated=True,
    press_distance=10.0,
    off_ball_action="MARK",
    default_x_factor=0.78, default_x_ref="my_goal",
    default_y=0,
)
fallback_commands = build_fallback(AGG_DEF_CONFIG)

OVERRIDE_CONFIG = OverrideConfig(always_blast=True)


agent = create_agent(SYSTEM_PROMPT, model_id="us.amazon.nova-micro-v1:0")
create_invoke_handler(
    app, agent, MY_PLAYER_ID, POSITION_LABEL, fallback_commands,
    fallback_cfg=AGG_DEF_CONFIG,
    override_cfg=OVERRIDE_CONFIG,
)

if __name__ == "__main__":
    app.run()

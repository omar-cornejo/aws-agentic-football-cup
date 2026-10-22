"""
Prompt-only team (strategy llm, 1-1-2) — controls ONLY player 1 (DEF).
No Markov / CSP / backtracking / overrides: the LLM decides every tick from
lib/llm_prompts.py (football principles) + lib/llm_state.py (facts) + memory v2.
"""

import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "lib"))
from _bootstrap import setup_lib_path; setup_lib_path(__file__)

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from llm_agent import create_llm_handler

app = BedrockAgentCoreApp()
MY_PLAYER_ID = 1
POSITION_LABEL = "DEF"

create_llm_handler(app, MY_PLAYER_ID, POSITION_LABEL)

if __name__ == "__main__":
    app.run()

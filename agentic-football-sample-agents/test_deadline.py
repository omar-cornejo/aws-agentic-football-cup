"""Regression test for the LLM deadline guard in lib/agent_base.py."""
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib"))

from test_helpers import mock_agentcore, GAME_STATE, TEAM_ID  # noqa: E402

mock_agentcore()

import agent_base  # noqa: E402
from fallback import build_fallback, MID_CONFIG  # noqa: E402
from overrides import OverrideConfig  # noqa: E402

PLAYER = 2
DECISIONS = []


class FakeApp:
    class logger:
        @staticmethod
        def info(msg):
            if msg.startswith("DECISION "):
                DECISIONS.append(json.loads(msg[len("DECISION "):]))

        @staticmethod
        def warn(msg):
            pass

        @staticmethod
        def error(msg):
            print("  [ERROR]", msg)

    def entrypoint(self, fn):
        self.handler = fn
        return fn


class FakeAgent:
    """Mimics a Strands Agent: callable, has .messages."""

    def __init__(self):
        self.messages = []
        self.delay = 0.0
        self.reply = ""

    def __call__(self, text):
        time.sleep(self.delay)
        return self.reply


async def invoke(app):
    payload = {"prompt": json.dumps({"gameState": GAME_STATE, "teamId": TEAM_ID,
                                     "myPlayers": [PLAYER]})}
    out = [chunk async for chunk in app.handler(payload, None)]
    return json.loads(out[0])


def run():
    app, agent = FakeApp(), FakeAgent()
    agent_base.create_invoke_handler(
        app, agent, PLAYER, "MID", build_fallback(MID_CONFIG),
        fallback_cfg=MID_CONFIG, override_cfg=OverrideConfig())
    budget = float(agent_base.tuning_mod.get("llm_deadline_ms", 800)) / 1000

    def tick(delay, reply):
        agent.delay, agent.reply = delay, reply
        t0 = time.perf_counter()
        cmds = asyncio.run(invoke(app))
        elapsed = time.perf_counter() - t0
        assert len(cmds) == 1 and cmds[0]["playerId"] == PLAYER, cmds
        return DECISIONS[-1]["source"], elapsed, cmds[0]["commandType"]

    good = json.dumps([{"commandType": "MOVE_TO", "playerId": PLAYER,
                        "parameters": {"target_x": 10, "target_y": 0, "sprint": True},
                        "duration": 0}])

    src, el, cmd = tick(0.05, good)
    assert src == "llm", src
    print(f"  fast answer      -> {src:17s} {el*1000:5.0f} ms  {cmd}")

    src, el, cmd = tick(budget + 0.7, good)
    assert src == "timeout-fallback", src
    assert el < budget + 0.15, f"took {el:.2f}s, deadline {budget:.2f}s"
    print(f"  slow answer      -> {src:17s} {el*1000:5.0f} ms  {cmd}")

    src, el, cmd = tick(0.05, good)
    assert src == "busy-fallback", src
    print(f"  still busy       -> {src:17s} {el*1000:5.0f} ms  {cmd}")

    time.sleep(1.0)
    src, el, cmd = tick(0.05, "Sure! I would press the ball now.")
    assert src == "parse-fallback", src
    print(f"  chatty answer    -> {src:17s} {el*1000:5.0f} ms  {cmd}")

    src, el, cmd = tick(0.05, good)
    assert src == "llm", src
    print(f"  recovered        -> {src:17s} {el*1000:5.0f} ms  {cmd}")
    print(f"DEADLINE GUARD OK (deadline {budget*1000:.0f} ms)")


if __name__ == "__main__":
    run()

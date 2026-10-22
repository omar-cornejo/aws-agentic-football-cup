"""Handler for the prompt-only team (strategy `llm`)."""
from __future__ import annotations

import asyncio
import copy
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

from strands import Agent
from strands.models import BedrockModel

import llm_prompts
import llm_state
import memory_v2
import stamina as stamina_mod
import tuning as tuning_mod
import units
from parsing import parse_commands
from state import observability_context, possession_context

BALL_CMDS = {"PASS", "SHOOT", "GK_DISTRIBUTE"}
DEFAULT_MODEL = "us.anthropic.claude-sonnet-4-6"
TERSE = ("\n\nOUTPUT RULE: your entire reply is the JSON array and nothing else. "
         "No reasoning, no explanation, no markdown. Start with '[' and end with ']'.")

REPLY_RULE = ("\n\nDecide now. Reply with ONLY the JSON array of one command — "
              "no analysis, no words. Your reply must start with [")

HOLD = {"GK": (-52, 0), "DEF": (-32, 0), "MID": (-12, 0), "FWD1": (2, 12), "FWD2": (2, -12)}


def build_agent(label: str, pid: int) -> Agent:
    model_id = str(tuning_mod.get(f"model_{label}", "") or "").strip() or DEFAULT_MODEL
    kw = {"model_id": model_id, "temperature": 0.3, "max_tokens": 160}
    if "anthropic" in model_id:
        kw["cache_prompt"] = "default"
    prompt = llm_prompts.system_prompt(label, pid) + TERSE
    return Agent(model=BedrockModel(**kw), system_prompt=prompt, callback_handler=None)


def _brief(cmd: dict) -> str:
    ct = cmd.get("commandType")
    p = cmd.get("parameters") or {}
    if ct == "MOVE_TO":
        return f"MOVE_TO ({p.get('target_x')},{p.get('target_y')}) sprint={p.get('sprint')}"
    if ct in ("PASS", "GK_DISTRIBUTE"):
        return f"{ct} to P{p.get('target_player_id')} {p.get('type') or p.get('method') or ''}".strip()
    if ct in ("MARK", "SLIDE_TACKLE"):
        return f"{ct} P{p.get('target_player_id')}"
    return str(ct)


def create_llm_handler(app, my_player_id: int, label: str):
    log = app.logger
    agent = build_agent(label, my_player_id)
    log.info(f"LLM-ONLY {label}: model {agent.model.config.get('model_id')}")
    deadline_s = float(tuning_mod.get("llm_deadline_ms", 1800)) / 1000.0
    pool = ThreadPoolExecutor(max_workers=1)
    inflight = {"f": None}
    last = {"cmd": None, "brief": None, "chat": "[]"}
    usage = {"prev": (0, 0), "d": (None, None)}

    def _run(text):
        agent.messages = []
        try:
            res = agent(text + REPLY_RULE)
        except Exception as e:
            part = ""
            try:
                part = "".join(c.get("text", "") for c in agent.messages[-1]["content"])
            except Exception:
                pass
            log.warn(f"{label} LLM stopped early ({type(e).__name__}); partial {len(part)} chars")
            return part
        try:
            u = res.metrics.accumulated_usage
            tin, tout = int(u.get("inputTokens", 0)), int(u.get("outputTokens", 0))
            p = usage["prev"] if tin >= usage["prev"][0] else (0, 0)
            usage["d"], usage["prev"] = (tin - p[0], tout - p[1]), (tin, tout)
        except Exception:
            pass
        return str(res)

    async def call(text):
        prev = inflight["f"]
        if prev is not None and not prev.done():
            return None, "busy-fallback"
        fut = pool.submit(_run, text)
        inflight["f"] = fut
        try:
            return await asyncio.wait_for(asyncio.wrap_future(fut), deadline_s), None
        except asyncio.TimeoutError:
            return None, "timeout-fallback"

    def safety(facts, team_id, pid):
        """Previous LLM command if still legal, else the role spot."""
        prev = last["cmd"]
        if prev is not None and (prev.get("commandType") not in BALL_CMDS or facts["i_have"]):
            return [copy.deepcopy(prev)], "repeat"
        x, y = HOLD.get(label, (0, 0))
        return [{"commandType": "MOVE_TO", "playerId": pid, "teamId": team_id, "duration": 0,
                 "parameters": {"target_x": x, "target_y": y, "sprint": False}}], "hold"

    def log_decision(source, cmds, ms, gs, team_id, pid, chars, nb, why=None, want=None):
        try:
            hb, dg = possession_context(gs, team_id, pid)
            d = {"pos": label, "tick": gs.get("tick"), "t": round(gs.get("gameTime", 0) or 0),
                 "source": source, "cmd": cmds[0].get("commandType") if cmds else None,
                 "latency_ms": ms, "prompt_chars": chars, "hb": hb, "dg": dg, "strat": "llm", "tm": team_id}
            d.update(observability_context(gs, team_id, pid))
            if why:
                d["sf"] = why
            if want:
                d["want"] = want
            if source == "llm" and usage["d"][0] is not None:
                d["tin"], d["tout"] = usage["d"]
                usage["d"] = (None, None)
            st = stamina_mod.stamina_pct(stamina_mod._me(gs, team_id, pid))
            if st is not None:
                d["st"] = round(st)
            if d["cmd"] == "SHOOT":
                aim = (cmds[0].get("parameters") or {}).get("aim_location")
                if aim:
                    d["aim"] = aim
            if nb is not None:
                memory_v2.after_decision(nb, cmds[0] if cmds else {}, bool(hb), dg, source)
                d.update(memory_v2.log_fields(nb))
            chat = json.dumps((gs.get("teamChat") or [])[-2:], ensure_ascii=False)[:160]
            if chat != last["chat"]:
                last["chat"] = chat
                d["chat"] = chat
            coach, trusted = llm_state.coach_context(gs, team_id)
            if coach:
                d["co"] = 1 if trusted else 2
            log.info("DECISION " + json.dumps(d, separators=(",", ":")))
        except Exception as e:
            log.warn(f"log failed: {e}")

    @app.entrypoint
    async def invoke(payload, context):
        prompt = (payload or {}).get("prompt", "{}")
        try:
            pd = prompt if isinstance(prompt, dict) else json.loads(prompt)
        except (TypeError, ValueError):
            pd = {}
        team_id = pd.get("teamId", 0) if isinstance(pd, dict) else 0
        mp = (pd.get("myPlayers") if isinstance(pd, dict) else None) or [my_player_id]
        pid = mp[0]
        gs = {}
        try:
            gs, _ = units.to_field(pd.get("gameState", {}) or {}, force=True)
            nb = None
            if memory_v2.enabled():
                try:
                    nb = memory_v2.notebook(memory_v2.match_key_from(
                        getattr(context, "session_id", None), gs), team_id, pid, label)
                    nb.observe(gs, team_id)
                except Exception as e:
                    nb = None
                    log.warn(f"memory_v2 failed: {e}")
            text, facts = llm_state.summarize(gs, team_id, pid, label, last["brief"])
            if nb is not None:
                blk = nb.block()
                if blk:
                    text = f"{text}\n\n{blk}"
            t0 = time.perf_counter()
            resp, skipped = await call(text)
            ms = round((time.perf_counter() - t0) * 1000)
            cmds = parse_commands(resp, team_id, pid) if resp is not None else []
            source, why, want = "llm", None, None
            if cmds:
                want = cmds[0].get("commandType")
                if want in BALL_CMDS and not facts["i_have"]:
                    cmds, why = safety(facts, team_id, pid)
                    why, source = f"illegal-{why}", "llm-illegal"
                else:
                    cmds = cmds[:1]
                    last["cmd"] = copy.deepcopy(cmds[0])
                    last["brief"] = _brief(cmds[0])
            else:
                source = skipped or "parse-fallback"
                cmds, why = safety(facts, team_id, pid)
            out = llm_state.to_engine(copy.deepcopy(cmds), team_id)
            for c in out:
                c["playerId"], c["teamId"] = pid, team_id
                c.setdefault("duration", 0)
            log_decision(source, out, ms, gs, team_id, pid, len(text), nb, why,
                         want if want != (out[0].get("commandType") if out else None) else None)
            yield json.dumps(out)
        except Exception as e:
            log.error(f"{label} llm agent error: {e}")
            x, y = HOLD.get(label, (0, 0))
            s = 1.0 if team_id == 0 else -1.0
            cmd = {"commandType": "MOVE_TO", "playerId": pid, "teamId": team_id, "duration": 0,
                   "parameters": {"target_x": x * s, "target_y": y * s, "sprint": False}}
            log_decision("error-fallback", [cmd], None, gs or {}, team_id, pid, None, None, "hold")
            yield json.dumps([cmd])

    return invoke

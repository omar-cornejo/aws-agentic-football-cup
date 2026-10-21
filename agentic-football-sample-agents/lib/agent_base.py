"""Base agent factory for AI soccer position agents."""

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable
from strands import Agent
from strands.models import BedrockModel

from parsing import parse_commands
from pattern_tracker import PatternTracker
from state import (summarize_state, possession_context, observability_context,
                   coach_context, get_goal_positions)
from forecast import effective_d_goal, carry_lead_m
import tuning as tuning_mod
from tactics import tactics_report
from fallback import FallbackConfig, build_last_resort
from overrides import OverrideConfig, apply_overrides
from tuning import apply_tuning
import copy
import ev_policy
import stamina as stamina_mod
import match_memory
import units
import agentcore_memory
import memory_v2
import mem3


CODE_FIRST_TAGS = {"ev", "dsearch", "gk-ev", "gk-angle", "ev-support"}


def create_agent(
    system_prompt: str,
    model_id: str = "us.amazon.nova-micro-v1:0",
    temperature: float = 0.2,
    max_tokens: int = 200,
) -> Agent:
    """Create a Strands Agent with the given system prompt."""
    model = BedrockModel(model_id=model_id, temperature=temperature, max_tokens=max_tokens)
    return Agent(model=model, system_prompt=system_prompt, callback_handler=None)


TERSE_RULE = ("\n\nOUTPUT RULE: your entire reply is the JSON array and nothing else. "
              "No reasoning, no explanation, no markdown. Start with '[' and end with ']'.")


REPLY_RULE = ("\n\nDecide now. Reply with ONLY the JSON array of one command — "
              "no analysis, no words. Your reply must start with [")

MEMORY_RULE = ("\n\nMEMORY: the prompt may end with MATCH MEMORY (what your actions achieved "
               "in THIS match) and LESSONS FROM YOUR LAST MATCHES. Use them: repeat what "
               "works, stop what keeps failing (e.g. intercepted passes, missed long "
               "shots), and cover the side they keep scoring from.")


def model_override(agent, position_label: str, log=None):
    """Swap the agent's model from tuning `model_<POS>` (e.g. model_MID)."""
    model_id = str(tuning_mod.get(f"model_{position_label}", "") or "").strip()
    if not model_id or not isinstance(agent, Agent):
        return agent
    sp = getattr(agent, "system_prompt", "") or ""
    if "anthropic" in model_id:
        new = create_agent(sp + TERSE_RULE, model_id=model_id, max_tokens=150)
    else:
        new = create_agent(sp, model_id=model_id)
    if log is not None:
        log.info(f"MODEL override for {position_label}: {model_id}")
    return new


def parse_prompt(payload) -> dict:
    """Parse the invocation payload's prompt, accepting either str or dict."""
    prompt = (payload or {}).get("prompt", "{}")
    if isinstance(prompt, dict):
        return prompt
    try:
        parsed = json.loads(prompt)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def team_id_from(prompt_data: dict, log, label: str) -> int:
    """teamId from the payload; HOME only as a loudly logged last resort."""
    team_id = prompt_data.get("teamId")
    if team_id is None:
        log.error(f"{label}: payload contained no teamId; falling back to HOME (0). "
                  "Commands this tick may be attributed to the wrong team.")
        return 0
    return team_id


def create_invoke_handler(
    app,
    agent: Agent,
    my_player_id: int,
    position_label: str,
    fallback_fn: Callable[[dict, int, int], list[dict]],
    fallback_cfg: FallbackConfig,
    override_cfg: OverrideConfig | None = None,
    log_label: str | None = None,
):
    """Create and register the @app.entrypoint invoke handler."""
    log = app.logger
    if memory_v2.enabled() and isinstance(agent, Agent):
        agent = create_agent((getattr(agent, "system_prompt", "") or "") + MEMORY_RULE,
                             model_id=agent.model.config.get("model_id"))
    add31 = mem3.mem31.prompt_addendum(position_label)
    if add31 and isinstance(agent, Agent):
        agent = create_agent((getattr(agent, "system_prompt", "") or "") + add31,
                             model_id=agent.model.config.get("model_id"))
    agent = model_override(agent, position_label, log)
    last_chat = {"v": "[]"}
    last_resort = build_last_resort(fallback_cfg, my_player_id)
    tracker = PatternTracker()

    deadline_s = float(tuning_mod.get("llm_deadline_ms", 1800)) / 1000.0
    llm_pool = ThreadPoolExecutor(max_workers=1)
    inflight = {"future": None}
    current = {"agent": agent, "key": None, "mem": False, "building": None}
    mem_pool = ThreadPoolExecutor(max_workers=1)

    def maybe_switch_memory(session_key):
        if not session_key or not agentcore_memory.enabled():
            return
        b = current["building"]
        if b is not None and b[1].done():
            try:
                current["agent"], current["mem"] = b[1].result(), True
                log.info(f"MEMORY on for {position_label}, session {b[0]}")
            except Exception as mem_error:
                current["agent"], current["mem"] = agent, False
                log.warn(f"MEMORY build failed for {position_label}: {mem_error}")
            current["building"] = None
        if session_key != current["key"] and current["building"] is None:
            current["key"] = session_key
            current["agent"], current["mem"] = agent, False
            current["building"] = (session_key, mem_pool.submit(
                agentcore_memory.build, agent, session_key, position_label))

    usage = {"in": 0, "out": 0, "last_in": None, "last_out": None}

    def _usage_delta(ag, res):
        """Tokens of this call (Strands accumulates usage across calls)."""
        try:
            u = res.metrics.accumulated_usage
            tin, tout = int(u.get("inputTokens", 0)), int(u.get("outputTokens", 0))
        except Exception:
            return
        key = id(ag)
        prev = usage.get(key, (0, 0))
        if tin < prev[0]:
            prev = (0, 0)
        usage[key] = (tin, tout)
        usage["last_in"], usage["last_out"] = tin - prev[0], tout - prev[1]

    nb_ref = {"nb": None}

    async def call_llm(text: str):
        """Return (response_text | None, source_if_skipped | None)."""
        prev = inflight["future"]
        if prev is not None and not prev.done():
            return None, "busy-fallback"

        ag, with_mem = current["agent"], current["mem"]

        def _run():
            if not with_mem:
                ag.messages = []
            mid = str(getattr(ag, "model", None) and ag.model.config.get("model_id") or "")
            res = ag(text + REPLY_RULE if "anthropic" in mid else text)
            _usage_delta(ag, res)
            return str(res)

        fut = llm_pool.submit(_run)
        inflight["future"] = fut
        try:
            return await asyncio.wait_for(asyncio.wrap_future(fut), deadline_s), None
        except asyncio.TimeoutError:
            return None, "timeout-fallback"

    tuned = apply_tuning(override_cfg, position_label)
    if tuned is not override_cfg:
        log.info(f"TUNING applied for {position_label}: {tuned}")
    override_cfg = tuned

    def log_decision(source, commands, latency_ms, game_state, prompt_chars,
                     effective_pid=my_player_id, team_id=0, ov=None, want=None,
                     proposed=None, stg=None):
        """One structured line per tick for CloudWatch Logs Insights."""
        try:
            hb, dg = possession_context(game_state, team_id, effective_pid)
            payload = {
                "pos": log_label or position_label,
                "tick": game_state.get("tick"),
                "t": round(game_state.get("gameTime", 0) or 0),
                "source": source,
                "cmd": commands[0].get("commandType") if commands else None,
                "latency_ms": latency_ms,
                "prompt_chars": prompt_chars,
                "hb": hb,
                "dg": dg,
                "tm": team_id,
            }
            strat = memory_v2.strategy_name()
            if strat != "default":
                payload["strat"] = strat
            payload.update(observability_context(game_state, team_id, effective_pid))
            if hb == 1:
                _, opp_gx = get_goal_positions(team_id)
                ticks = int(tuning_mod.get("shoot_lead_ticks", 2))
                per = float(tuning_mod.get("shoot_lead_per_tick", 6.0))
                lead = carry_lead_m(ticks, per, None, {}, toward_goal=True)
                mx, my = payload.get("mx"), payload.get("my")
                if isinstance(mx, (int, float)) and isinstance(my, (int, float)):
                    payload["edg"] = round(
                        effective_d_goal({"x": mx, "y": my}, opp_gx, lead))
            if hb == 1 and commands and effective_pid != 0:
                crit = ev_policy.evaluate(game_state, team_id, effective_pid, commands[0])
                if crit:
                    payload.update(ev=crit["ev"], evb=crit["best"], evr=crit["regret"],
                                   evx=crit["best_choice"], xt=crit["v_here"])
                    if proposed is not None:
                        lc = ev_policy.evaluate(game_state, team_id, effective_pid, proposed)
                        if lc:
                            payload["evl"] = lc["ev"]
            if ov:
                payload["ov"] = ov
            if hb == 1 and commands:
                payload.update(mem3.mem31.log_fields(game_state, team_id, effective_pid,
                                                     commands[0]))
            if source == "llm" and usage["last_in"] is not None:
                payload["tin"], payload["tout"] = usage["last_in"], usage["last_out"]
                usage["last_in"] = None
            nb = nb_ref["nb"]
            if nb is not None:
                try:
                    memory_v2.after_decision(nb, commands[0] if commands else {},
                                             bool(hb), dg, source)
                    payload.update(memory_v2.log_fields(nb))
                except Exception:
                    pass
            try:
                match_memory.record(commands, game_state, team_id, effective_pid,
                                    dg, bool(hb))
                snap = match_memory.snapshot(team_id, effective_pid)
                if snap.get("shots"):
                    payload["ms"] = f'{snap["shots"]}/{snap["goals"]}/{snap["misses"]}'
            except Exception:
                pass
            me_st = stamina_mod.stamina_pct(stamina_mod._me(game_state, team_id, effective_pid))
            if me_st is not None:
                payload["st"] = round(me_st)
            if stg:
                payload["stg"] = stg
            if want is not None:
                payload["want"] = want
                if want != payload.get("cmd"):
                    payload["fix"] = 1
            if payload.get("cmd") == "SHOOT" and commands:
                aim = (commands[0].get("parameters") or {}).get("aim_location")
                if aim:
                    payload["aim"] = aim
            raw_chat = json.dumps((game_state.get("teamChat") or [])[-2:],
                                  ensure_ascii=False)[:160]
            if raw_chat != last_chat["v"]:
                last_chat["v"] = raw_chat
                payload["chat"] = raw_chat
            coach, trusted = coach_context(game_state, team_id)
            if coach:
                payload["co"] = 1 if trusted else 2
            log.info("DECISION " + json.dumps(payload, separators=(",", ":")))
        except Exception:
            pass

    @app.entrypoint
    async def invoke(payload, context):
        prompt_data = parse_prompt(payload)
        team_id = team_id_from(prompt_data, log, position_label)
        try:
            game_state = prompt_data.get("gameState", {}) or {}
            game_state, _converted = units.to_field(game_state)
            if memory_v2.enabled():
                try:
                    _mp0 = prompt_data.get("myPlayers", [my_player_id])
                    _pid0 = _mp0[0] if _mp0 else my_player_id
                    nb_ref["nb"] = memory_v2.notebook(
                        memory_v2.match_key_from(getattr(context, "session_id", None),
                                                 game_state),
                        team_id, _pid0, log_label or position_label)
                    nb_ref["nb"].observe(game_state, team_id)
                except Exception as mem_error:
                    nb_ref["nb"] = None
                    log.warn(f"memory_v2 failed: {mem_error}")
            try:
                maybe_switch_memory(getattr(context, "session_id", None)
                                    or game_state.get("matchId"))
            except Exception as mem_error:
                log.warn(f"memory switch failed: {mem_error}")
            try:
                _mp = prompt_data.get("myPlayers", [my_player_id])
                match_memory.observe(game_state, team_id, _mp[0] if _mp else my_player_id)
            except Exception as mem_error:
                log.warn(f"match_memory.observe failed: {mem_error}")

            my_players = prompt_data.get("myPlayers", [my_player_id])
            effective_pid = my_players[0] if my_players else my_player_id

            state_summary = summarize_state(
                game_state, team_id, effective_pid, position_label
            )

            tracker.update(game_state, team_id)
            scout = tracker.report(game_state, team_id, position_label)
            if scout:
                state_summary = f"{state_summary}\n\n{scout}"

            tactics = tactics_report(game_state, team_id, effective_pid, position_label)
            if tactics:
                state_summary = f"{state_summary}\n\n{tactics}"

            log.info(f"{position_label} agent invoked for team {team_id}, controlling player {effective_pid}")

            m3, m3_tag = mem3.decide(game_state, team_id, effective_pid, position_label)
            if m3 is not None:
                m3_cmds, stg = stamina_mod.guard([m3], game_state, team_id, effective_pid)
                log_decision(m3_tag, m3_cmds, 0, game_state, len(state_summary),
                             effective_pid, team_id, ov=m3_tag, stg=stg)
                yield json.dumps(m3_cmds)
                return

            if override_cfg is not None and any(getattr(override_cfg, k, False)
                                                for k in ("ev_select", "def_search", "gk_ev", "ev_support")):
                stub = [{"commandType": "SET_STANCE", "playerId": effective_pid,
                         "teamId": team_id, "parameters": {}, "duration": 0}]
                ev_cmds, ov = apply_overrides(copy.deepcopy(stub), game_state, team_id,
                                              effective_pid, position_label, override_cfg)
                if ov in CODE_FIRST_TAGS:
                    ov = mem3.shape(ev_cmds, game_state, team_id, effective_pid,
                                    position_label) or ov
                    ev_cmds, stg = stamina_mod.guard(ev_cmds, game_state, team_id, effective_pid)
                    log_decision(ov, ev_cmds, 0, game_state, len(state_summary),
                                 effective_pid, team_id, ov=ov, stg=stg)
                    yield json.dumps(ev_cmds)
                    return

            if nb_ref["nb"] is not None:
                mem_block = nb_ref["nb"].block()
                if mem_block:
                    state_summary = f"{state_summary}\n\n{mem_block}"

            t0 = time.perf_counter()
            response_text, skipped = await call_llm(state_summary)
            llm_ms = round((time.perf_counter() - t0) * 1000)

            commands = (parse_commands(response_text, team_id, effective_pid)
                        if response_text is not None else [])

            if commands:
                want = commands[0].get("commandType")
                proposed = copy.deepcopy(commands[0])
                commands, ov = apply_overrides(
                    commands, game_state, team_id, effective_pid,
                    position_label, override_cfg)
                ov = mem3.shape(commands, game_state, team_id, effective_pid,
                                position_label) or ov
                if ov:
                    log.info(f"OVERRIDE {ov}: LLM said something else, enforcing "
                             f"{commands[0].get('commandType')}")
                log.info(f"LLM returned {len(commands)} commands: "
                         f"{[c.get('commandType') for c in commands]}")
                commands, stg = stamina_mod.guard(commands, game_state, team_id, effective_pid)
                log_decision("llm", commands, llm_ms, game_state, len(state_summary),
                             effective_pid, team_id, ov=ov, want=want, proposed=proposed,
                             stg=stg)
                yield json.dumps(commands)
            else:
                source = skipped or "parse-fallback"
                if skipped:
                    log.warn(f"LLM {source} after {llm_ms}ms, using rule fallback")
                else:
                    log.warn(f"LLM parse failed, using fallback. Response: {response_text[:200]}")
                commands = fallback_fn(game_state, team_id, effective_pid)
                fb_cmd = commands[0].get("commandType") if commands else None
                commands, ov = apply_overrides(
                    commands, game_state, team_id, effective_pid,
                    position_label, override_cfg)
                ov = mem3.shape(commands, game_state, team_id, effective_pid,
                                position_label) or ov
                log.info(f"Fallback returned {len(commands)} commands")
                commands, stg = stamina_mod.guard(commands, game_state, team_id, effective_pid)
                log_decision(source, commands, llm_ms, game_state,
                             len(state_summary), effective_pid, team_id, ov=ov,
                             want=fb_cmd, stg=stg)
                yield json.dumps(commands)

        except Exception as e:
            log.error(f"{position_label} agent error: {e}")
            try:
                my_players = prompt_data.get("myPlayers", [my_player_id])
                effective_pid = my_players[0] if my_players else my_player_id
                gs_err, _ = units.to_field(prompt_data.get("gameState", {}) or {})
                commands = fallback_fn(
                    gs_err,
                    team_id,
                    effective_pid,
                )
                fb_cmd = commands[0].get("commandType") if commands else None
                commands, ov = apply_overrides(
                    commands, gs_err, team_id,
                    effective_pid, position_label, override_cfg)
                log_decision("error-fallback", commands, None,
                             gs_err, None,
                             effective_pid, team_id, ov=ov, want=fb_cmd)
                yield json.dumps(commands)
            except Exception as fallback_error:
                log.error(f"{position_label} fallback also failed: {fallback_error}")
                cmd = dict(last_resort)
                cmd["teamId"] = team_id
                log_decision("last-resort", [cmd], None, {}, None, team_id=team_id)
                yield json.dumps([cmd])

    return invoke

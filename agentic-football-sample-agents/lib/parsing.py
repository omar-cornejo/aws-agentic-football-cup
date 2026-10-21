"""Response parsing utilities for AI soccer agents."""

import re

from json_tolerant import parse_json_tolerant

VALID_COMMANDS = {
    "MOVE_TO", "PASS", "SHOOT", "SLIDE_TACKLE", "PRESS_BALL", "INTERCEPT", "MARK",
    "FOLLOW_PLAYER", "GK_DISTRIBUTE", "SET_STANCE", "CLEAR_OVERRIDE", "RESET",
}

_ARITH = re.compile(
    r'(?<=[:\s\[,])(-?\d+(?:\.\d+)?)\s*\*\s*(-?\d+(?:\.\d+)?)(?=\s*[,}\]])')


def _fold_arithmetic(text: str) -> str:
    return _ARITH.sub(
        lambda m: repr(float(m.group(1)) * float(m.group(2))), text)


def parse_commands(text: str, team_id: int, my_player_id: int, on_recovered=None) -> list[dict]:
    """Extract commands from LLM response, forcing the given player ID on all commands."""
    text = _fold_arithmetic(text or "")
    for candidate in _iter_array_candidates(text):
        parsed = _loads(candidate, on_recovered)
        if not isinstance(parsed, list):
            continue
        commands = _tag_commands(parsed, team_id, my_player_id)
        if commands:
            return commands

    parsed = _loads(text, on_recovered)
    if isinstance(parsed, list):
        return _tag_commands(parsed, team_id, my_player_id)
    if isinstance(parsed, dict) and "commandType" in parsed:
        return _tag_commands([parsed], team_id, my_player_id)

    return []


def _iter_array_candidates(text: str):
    """Yield each balanced ``[...]`` span in ``text``, left to right."""
    for start, ch in enumerate(text):
        if ch != "[":
            continue
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            c = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif c == "\\":
                    escaped = True
                elif c == '"':
                    in_string = False
                continue
            if c == '"':
                in_string = True
            elif c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    yield text[start : i + 1]
                    break


def _loads(candidate: str, on_recovered):
    """Parse one candidate payload; None when it does not parse even after normalisation."""
    result = parse_json_tolerant(candidate)
    if result is None:
        return None
    value, recovered = result
    if recovered and on_recovered is not None:
        on_recovered(candidate)
    return value


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _tag_commands(commands: list, team_id: int, my_player_id: int) -> list[dict]:
    """Add teamId and playerId to each command, filtering to valid ones."""
    result = []
    for cmd in commands:
        if not isinstance(cmd, dict):
            continue
        cmd["teamId"] = team_id
        cmd["playerId"] = my_player_id
        if "commandType" in cmd:
            if cmd["commandType"] not in VALID_COMMANDS:
                continue
            params = cmd.get("parameters") or {}
            cmd["parameters"] = params
            if cmd["commandType"] == "MOVE_TO":
                if isinstance(params.get("target_x"), (int, float)):
                    params["target_x"] = _clamp(params["target_x"], -55, 55)
                if isinstance(params.get("target_y"), (int, float)):
                    params["target_y"] = _clamp(params["target_y"], -35, 35)
            if cmd["commandType"] == "SHOOT":
                if params.get("aim_location") not in ("TL", "TR", "BL", "BR", "CENTER"):
                    params["aim_location"] = "CENTER"
                power = params.get("power")
                params["power"] = _clamp(power, 0.3, 1.0) if isinstance(power, (int, float)) else 1.0
            if cmd["commandType"] == "PRESS_BALL":
                inten = params.get("intensity")
                params["intensity"] = _clamp(inten, 0.1, 1.0) if isinstance(inten, (int, float)) else 1.0
            if cmd["commandType"] in ("PASS", "MARK", "FOLLOW_PLAYER", "GK_DISTRIBUTE", "SLIDE_TACKLE"):
                target = params.get("target_player_id")
                if isinstance(target, str):
                    m = re.fullmatch(r"\s*(?:P|ID\s*=?\s*)?(\d)\s*", target, re.I)
                    if m:
                        target = int(m.group(1))
                    elif target.strip().upper() == "GK":
                        target = 0
                    params["target_player_id"] = target
                elif isinstance(target, float) and target.is_integer():
                    target = int(target)
                    params["target_player_id"] = target
                bad_pass_target = (
                    not isinstance(target, int) or not 0 <= target <= 4
                    or target == my_player_id
                )
                if cmd["commandType"] in ("PASS", "GK_DISTRIBUTE") and bad_pass_target:
                    params["target_player_id"] = 3 if my_player_id != 3 else 4
                elif target is None:
                    if cmd["commandType"] == "SLIDE_TACKLE":
                        params["target_player_id"] = -1
                    else:
                        params["target_player_id"] = 0
            result.append(cmd)
    return result

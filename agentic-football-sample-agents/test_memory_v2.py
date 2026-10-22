"""Memory v2 notebook: outcomes, prompt block, lessons (no AWS needed)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib"))
os.environ.pop("MEMORY_ID", None)

import tuning  # noqa: E402

tuning._DATA.setdefault("global", {})["memory_v2"] = True
import memory_v2 as m  # noqa: E402


def P(team, idx, x, y):
    return {"agentId": f"{'home' if team == 0 else 'away'}_{idx}",
            "teamCode": "home" if team == 0 else "away", "position": {"x": x, "y": y}}


def gs(t, holder=None, ball=(0, 0), score=(0, 0), extra=None):
    players = [P(0, i, -40 + 10 * i, 0) for i in range(5)] + \
              [P(1, i, 40 - 10 * i, 5) for i in range(5)]
    for p in extra or []:
        players = [q for q in players if not (q["agentId"] == p["agentId"])] + [p]
    b = {"position": {"x": ball[0], "y": ball[1]}}
    if holder:
        b["possessionAgentId"] = holder
    return {"gameTime": t, "players": players, "ball": b,
            "score": {"home": score[0], "away": score[1]}}


def run():
    nb = m.notebook("matchA", 0, 2, "MID")
    s = gs(10, "home_2", (-20, 0))
    nb.observe(s, 0)
    m.after_decision(nb, {"commandType": "PASS", "parameters": {"target_player_id": 4}}, True, 75, "llm")
    nb.observe(gs(12, "home_4", (0, 0)), 0)
    assert nb.s["pass"] == [1, 0], nb.s
    me = P(0, 2, 25, 0)
    nb.observe(gs(14, "home_2", (25, 0), extra=[me]), 0)
    m.after_decision(nb, {"commandType": "SHOOT", "parameters": {}}, True, 30, "ev")
    nb.observe(gs(16, None, (55, 0), score=(1, 0)), 0)
    assert nb.s["shot"]["mid"] == [1, 1], nb.s
    assert nb.s["gf"] == 1
    for t in (20, 22, 24):
        opp = P(1, 1, -30, 20)
        nb.observe(gs(t, "away_1", (-30, 20), score=(1, 0), extra=[opp]), 0)
    nb.observe(gs(26, None, (-55, 0), score=(1, 1)), 0)
    assert nb.s["ga"] == 1
    block = nb.block()
    print(block)
    assert "we scored" in block and "THEY scored" in block
    assert "passes 1/1" in block
    side = next(iter(nb.s["ga_side"]))
    assert side in ("LEFT", "RIGHT"), nb.s["ga_side"]

    snaps = [{"m": f"x{i}", "ts": i, "s": {"pass": [3, 4], "shot": {"long": [5, 0], "mid": [2, 1]},
                                           "carry": [1, 4], "gf": 1, "ga": 2,
                                           "ga_side": {"LEFT": 1}}} for i in range(3)]
    les = m.lessons_from(snaps)
    print("\n".join(les))
    assert any("long shots" in x and "rarely" in x for x in les)
    assert any("LEFT" in x for x in les)
    assert m.prior_from(snaps) == {"long_n": 15, "long_g": 0}
    nb.prior = m.prior_from(snaps)
    mult, why = m.long_shot_prior_mult(0, 2)
    assert mult < 1 and why, (mult, why)

    nb2 = m.notebook("matchB", 0, 2, "MID")
    assert nb2 is not nb and nb2.s["gf"] == 0
    print("tally:", nb.tally(), "| fields:", m.log_fields(nb))
    print("MEMORY V2 OK")


if __name__ == "__main__":
    run()

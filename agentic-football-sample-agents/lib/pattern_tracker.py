"""In-process cross-tick pattern tracker — compact scouting memory."""

from collections import Counter, deque

from state import _player_idx, _is_my_team, _possession_idx, get_goal_positions, resolve_holder

DEF_THIRD_DEPTH = 36.7


class PatternTracker:
    """Tracks opponent tendencies across ticks for one agent runtime."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.ticks = 0
        self.last_game_time = None
        self.opp_hold = Counter()
        self.opp_hold_total = 0
        self.opp_side_y = Counter()
        self.gk_feed = Counter()
        self.gk_feed_total = 0
        self.prev_holder = None
        self.def_third = deque(maxlen=30)

    def update(self, game_state: dict, team_id: int) -> None:
        """Ingest one game-state snapshot."""
        game_time = game_state.get("gameTime", 0) or 0
        if self.last_game_time is not None and game_time + 5 < self.last_game_time:
            self.reset()
        self.last_game_time = game_time
        self.ticks += 1

        ball = game_state.get("ball", {}) or {}
        players = game_state.get("players", []) or []
        ball_pos = ball.get("position", {}) or {}

        holder = None
        p = resolve_holder(ball, players)
        if p is not None:
            holder = (not _is_my_team(p, team_id), _player_idx(p))

        if holder and holder[0]:
            self.opp_hold[holder[1]] += 1
            self.opp_hold_total += 1
            self.opp_side_y["y<0" if (ball_pos.get("y", 0) or 0) < 0 else "y>0"] += 1

        if holder and self.prev_holder and self.prev_holder != holder:
            prev_is_opp, prev_idx = self.prev_holder
            cur_is_opp, cur_idx = holder
            if prev_is_opp and cur_is_opp and prev_idx == 0 and cur_idx != 0:
                self.gk_feed[cur_idx] += 1
                self.gk_feed_total += 1
        if holder:
            self.prev_holder = holder

        my_goal_x, _ = get_goal_positions(team_id)
        ball_x = ball_pos.get("x", 0) or 0
        self.def_third.append(1 if abs(ball_x - my_goal_x) < DEF_THIRD_DEPTH else 0)

    def report(self, game_state: dict, team_id: int, position_label: str) -> str:
        """Distill counters into <=4 prompt lines. Empty string until signal exists."""
        if self.ticks < 6:
            return ""

        lines = []

        if self.opp_hold_total >= 5:
            idx, n = self.opp_hold.most_common(1)[0]
            pct = round(100 * n / self.opp_hold_total)
            if pct >= 35:
                lines.append(
                    f"- Opp P{idx} carries the ball most ({pct}% of their possession):"
                    f" main threat, press/mark P{idx} first"
                )
            side, side_n = self.opp_side_y.most_common(1)[0]
            side_pct = round(100 * side_n / self.opp_hold_total)
            if side_pct >= 60:
                lines.append(
                    f"- Opp attacks mostly on the {side} side ({side_pct}%):"
                    f" shade your positioning toward that side"
                )

        if position_label in ("MID", "FWD1", "FWD2") and self.gk_feed_total >= 3:
            feed_idx, feed_n = self.gk_feed.most_common(1)[0]
            lines.append(
                f"- Their GK usually feeds P{feed_idx} ({feed_n}/{self.gk_feed_total}):"
                f" anticipate and intercept that outlet"
            )

        if position_label in ("GK", "DEF") and len(self.def_third) >= 10:
            press_pct = round(100 * sum(self.def_third) / len(self.def_third))
            if press_pct >= 50:
                lines.append(
                    f"- Ball in OUR defensive third {press_pct}% of recent ticks:"
                    f" high danger, prioritize defending"
                )

        score = game_state.get("score", {}) or {}
        home, away = score.get("home", 0), score.get("away", 0)
        mine, theirs = (home, away) if team_id == 0 else (away, home)
        if mine < theirs:
            lines.append(f"- Score {mine}-{theirs}: LOSING, take more risks and push for goals")
        elif mine > theirs:
            lines.append(f"- Score {mine}-{theirs}: WINNING, do not concede cheap counters")

        if not lines:
            return ""
        return "SCOUTING REPORT (match-long stats):\n" + "\n".join(lines[:4])

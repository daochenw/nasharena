#!/usr/bin/env python3
"""Score how exciting a game is for a human spectator, 1-10.

Usage:  python3 excitement.py games/game-001.json [more.json ...]

Reads the sidecar JSON produced by generate_games.py (no engine calls;
the only dependency is python-chess, used to replay the moves).

Eight factors, each normalized to [0, 1]. The first five follow the spec,
the last three are spectator staples. All material factors share one
replayed material/eval timeline but measure orthogonal things:

  sacrifice     discrete events: a side voluntarily sheds material while its
                own engine still likes its position (intent, not a blunder)
  compensation  how long a side sits >=2 pawns down while the engines'
                consensus says it is doing fine
  comeback      outcome bonus: the long-material-down side wins (or holds)
  forced        longest run of consecutive plies where the mover had
                essentially no choice
  divergence    Leela and Stockfish disagree about the same position
  swings        momentum: total eval variation plus lead changes
  king_danger   checks, sustained king hunts, mate on the board
  finish        decisive > drawn; checkmate beats resignation-style ends

The weighted raw score maps to the integer via fixed anchors (a quiet,
correct engine draw lands ~2-3, a sharp decisive game ~5-6, sac + comeback
+ disagreement ~7-8, immortal-game material 9-10).
"""

import json
import sys
from pathlib import Path

import chess

PIECE_VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
               chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}

WEIGHTS = {
    "sacrifice": 0.22, "compensation": 0.14, "comeback": 0.14,
    "forced": 0.10, "divergence": 0.10,
    "swings": 0.12, "king_danger": 0.10, "finish": 0.08,
}

# A sacrifice / deficit must be at least this many pawns of material.
SAC_PAWNS = 2
# Consensus expected score (mover POV) above which a material deficit
# counts as "the engines think there is compensation".
COMP_OK = 0.45


def clamp01(x):
    return max(0.0, min(1.0, x))


def material_balance(board):
    """White material minus black material, in pawns."""
    bal = 0
    for piece_type, value in PIECE_VALUE.items():
        bal += value * len(board.pieces(piece_type, chess.WHITE))
        bal -= value * len(board.pieces(piece_type, chess.BLACK))
    return bal


def pov(eval_white, side):
    return eval_white if side == "white" else 1.0 - eval_white


def own_eval(move):
    e = move["evalLc0"] if move["engine"] == "Lc0" else move["evalSF"]
    return pov(e, move["side"])


def replay(moves):
    """Per-ply timeline: material balance, legal-move count, check flags."""
    board = chess.Board()
    timeline = []
    for m in moves:
        legal = board.legal_moves.count()
        board.push_uci(m["uci"])
        timeline.append({
            "balance": material_balance(board),   # after the move, white POV
            "only_move": legal == 1,
            "gives_check": board.is_check(),
            "is_mate": board.is_checkmate(),
        })
    return timeline


def is_forced(move, tl_entry):
    """Did the mover have essentially no choice?"""
    if tl_entry["only_move"]:
        return True
    cands = move.get("candidates") or []
    if move["engine"] == "Lc0":
        return bool(cands) and cands[0].get("share", 0) >= 0.85
    extra = move.get("extra") or {}
    return extra.get("changes", 99) == 0 and extra.get("marginCp", 0) >= 120


def pov_balance(timeline, idx, side):
    bal = timeline[idx]["balance"]
    return bal if side == "white" else -bal


def score_sacrifices(moves, timeline):
    """Voluntary, confident material give-ups. Returns (score, events).

    "Given" material is measured against the mover's BEST balance once the
    exchange window settles - a piece that comes straight back via recapture
    is a trade, not a sacrifice.
    """
    events = []
    for i, m in enumerate(moves):
        if m.get("book"):
            continue  # a book gambit is theory, not the engine's own daring
        before = pov_balance(timeline, i - 1, m["side"]) if i else 0
        window = range(min(i + 2, len(timeline) - 1),
                       min(i + 7, len(timeline)))
        after = max(pov_balance(timeline, j, m["side"]) for j in window)
        given = before - after
        if given < SAC_PAWNS or is_forced(m, timeline[i]):
            continue
        # Intent, not blunder: the mover's own engine stays confident now
        # and hasn't collapsed a few moves later.
        now = own_eval(m)
        later_idx = min(i + 8, len(moves) - 1)
        later = pov(moves[later_idx]["evalLc0" if m["engine"] == "Lc0" else "evalSF"],
                    m["side"])
        if now >= 0.40 and later >= now - 0.20:
            events.append((m["ply"], m["san"], given))
    score = clamp01(sum(min(g, 6) / 6 * 0.6 for _, _, g in events))
    return score, events


def deficit_stretches(moves, timeline):
    """Consecutive-ply stretches where one side is >=SAC_PAWNS down.

    Yields (side, length, compensated_plies) per stretch.
    """
    stretches = []
    side, length, comp = None, 0, 0
    for m, t in zip(moves, timeline):
        down = ("black" if t["balance"] >= SAC_PAWNS else
                "white" if t["balance"] <= -SAC_PAWNS else None)
        if down != side:
            if side and length:
                stretches.append((side, length, comp))
            side, length, comp = down, 0, 0
        if down:
            length += 1
            consensus = pov((m["evalLc0"] + m["evalSF"]) / 2, down)
            if consensus >= COMP_OK:
                comp += 1
    if side and length:
        stretches.append((side, length, comp))
    return stretches


def score_compensation(stretches, n_plies):
    comp_plies = sum(c for _, _, c in stretches)
    # ~a third of the game spent material-down-but-fine saturates the factor.
    return clamp01(comp_plies / (n_plies / 3))


def score_comeback(stretches, result):
    winner = {"1-0": "white", "0-1": "black"}.get(result)
    best = 0.0
    for side, length, comp in stretches:
        if length < 10 or comp < length / 2:
            continue  # brief or hopeless deficits are not a comeback story
        depth = clamp01(length / 30)
        if side == winner:
            best = max(best, 0.6 + 0.4 * depth)
        elif winner is None:
            best = max(best, 0.25 + 0.15 * depth)  # held a long deficit
    return best


def score_forced(moves, timeline):
    longest = run = 0
    for m, t in zip(moves, timeline):
        run = run + 1 if is_forced(m, t) else 0
        longest = max(longest, run)
    # A 4-ply forced burst barely registers; 12 consecutive plies saturate.
    return clamp01((longest - 3) / 9), longest


def score_divergence(moves):
    """Largest sustained Leela-vs-Stockfish disagreement (rolling 3 plies)."""
    diffs = [abs(m["evalLc0"] - m["evalSF"]) for m in moves]
    if len(diffs) < 3:
        return 0.0, max(diffs, default=0.0)
    peak = max(sum(diffs[i:i + 3]) / 3 for i in range(len(diffs) - 2))
    # 0.08 is engine noise; a 0.38 expected-score gap is a shouting match.
    return clamp01((peak - 0.08) / 0.30), peak


def score_swings(moves):
    evals = [(m["evalLc0"] + m["evalSF"]) / 2 for m in moves]
    smooth = [sum(evals[max(0, i - 2):i + 1]) / len(evals[max(0, i - 2):i + 1])
              for i in range(len(evals))]
    variation = sum(abs(b - a) for a, b in zip(smooth, smooth[1:]))
    lead_changes = 0
    state = None
    for e in smooth:
        s = "w" if e > 0.58 else "b" if e < 0.42 else state
        if s != state and state is not None and s is not None:
            lead_changes += 1
        state = s
    return clamp01(variation / 1.5 + lead_changes * 0.25)


def score_king_danger(timeline):
    checks = sum(t["gives_check"] for t in timeline)
    # Longest one-sided checking spree (mover checks every other ply).
    hunt = run = 0
    for i, t in enumerate(timeline):
        if t["gives_check"] and (i < 2 or timeline[i - 2]["gives_check"]):
            run += 1
        else:
            run = 1 if t["gives_check"] else 0
        hunt = max(hunt, run)
    base = clamp01(checks / 10) * 0.6 + clamp01((hunt - 1) / 4) * 0.4
    if timeline and timeline[-1]["is_mate"]:
        base = max(base, 0.8)
    return clamp01(base)


def score_finish(result, timeline):
    if timeline and timeline[-1]["is_mate"]:
        return 1.0
    return 0.6 if result in ("1-0", "0-1") else 0.2


def to_integer(raw):
    """Fixed anchors: raw 0.05 -> 1 ... raw 0.80 -> 10."""
    return max(1, min(10, round(1 + 9 * clamp01((raw - 0.05) / 0.75))))


def evaluate(path):
    game = json.loads(Path(path).read_text())
    moves = game["moves"]
    timeline = replay(moves)
    stretches = deficit_stretches(moves, timeline)

    sac_score, sac_events = score_sacrifices(moves, timeline)
    forced_score, forced_run = score_forced(moves, timeline)
    div_score, div_peak = score_divergence(moves)
    factors = {
        "sacrifice": sac_score,
        "compensation": score_compensation(stretches, len(moves)),
        "comeback": score_comeback(stretches, game["result"]),
        "forced": forced_score,
        "divergence": div_score,
        "swings": score_swings(moves),
        "king_danger": score_king_danger(timeline),
        "finish": score_finish(game["result"], timeline),
    }
    raw = sum(WEIGHTS[k] * v for k, v in factors.items())

    print(f"\n{path}: {game['white']} vs {game['black']}  "
          f"{game['result']}  ({len(moves)} plies)")
    for k, v in factors.items():
        note = ""
        if k == "sacrifice" and sac_events:
            note = "  " + ", ".join(f"ply {p} {s} (-{g})" for p, s, g in sac_events)
        if k == "forced":
            note = f"  longest run {forced_run}"
        if k == "divergence":
            note = f"  peak {div_peak:.2f}"
        print(f"  {k:<13} {v:5.2f}  x{WEIGHTS[k]:.2f}{note}")
    score = to_integer(raw)
    print(f"  raw {raw:.3f}  ->  excitement: {score}/10")
    return score, raw


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: excitement.py game.json [game.json ...]")
    for path in sys.argv[1:]:
        evaluate(path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Retro-apply the resignation rule (RESIGN_WP) to games already on disk.

Existing JSONs store both engines' eval bars after every ply, so resignation
can be replayed offline: find the first post-book ply where the side to move
would see its OWN win prob below RESIGN_WP, and end the game there — the move
list is truncated, termination becomes "resigned", and the PGN is rewritten
to match.

Safety: a game whose FINAL result differs from what resignation would give
(a sub-threshold comeback, e.g. a held fortress) is left untouched and
reported — rewriting it would erase a real save. The strict rule still
applies natively to newly generated games.

Idempotent; run after generation:
  python3 gen/apply_resignations.py
"""
import json
import sys
from pathlib import Path

import chess
import chess.pgn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_games import RESIGN_WP, GAME_RE, game_result, normalize_game

GAMES = Path(__file__).resolve().parent.parent / "games"


def find_resign_ply(g):
    """Index into g["moves"] AFTER which the side to move resigns, plus
    (side, own_wp), or None. Book plies are exempt (gambit dips are theory)."""
    book = g.get("bookPlies", 0)
    board = chess.Board()
    for i, m in enumerate(g["moves"]):
        board.push(chess.Move.from_uci(m["uci"]))
        if m["ply"] <= book:
            continue
        if game_result(board) is not None:
            continue  # game actually over here; nobody left to resign
        next_white = board.turn == chess.WHITE
        next_engine = g["white"] if next_white else g["black"]
        ev = m["evalLc0"] if next_engine == "Lc0" else m["evalSF"]
        own = ev if next_white else 1.0 - ev
        if own < RESIGN_WP:
            return i, ("white" if next_white else "black"), own
    return None


def rewrite_pgn(pgn_path, n_plies, result):
    """Truncate the PGN mainline to n_plies and restamp Result/Termination."""
    with open(pgn_path) as f:
        game = chess.pgn.read_game(f)
    new = chess.pgn.Game()
    new.headers.update(game.headers)
    new.headers["Result"] = result
    new.headers["Termination"] = "resignation"
    node = new
    for i, mv in enumerate(game.mainline_moves()):
        if i >= n_plies:
            break
        node = node.add_variation(mv)
    pgn_path.write_text(str(new) + "\n", encoding="utf-8")


def main():
    files = sorted(f for f in GAMES.glob("game-*.json") if GAME_RE.search(f.name))
    applied = skipped_flip = untouched = 0
    for f in files:
        g = json.loads(f.read_text())
        if g.get("termination") == "resigned":
            untouched += 1
            continue
        hit = find_resign_ply(g)
        if hit is None:
            untouched += 1
            continue
        i, side, own = hit
        new_result = "0-1" if side == "white" else "1-0"
        if new_result != g["result"]:
            print(f"{f.name}: SKIPPED — {side} dipped to {own:.1%} after ply "
                  f"{g['moves'][i]['ply']} but came back ({g['result']}); "
                  f"left as-is")
            skipped_flip += 1
            continue
        old_plies = len(g["moves"])
        g["moves"] = g["moves"][:i + 1]
        g["result"] = new_result
        g["termination"] = "resigned"
        g["resignSide"] = side
        normalize_game(g)  # difficulty/delayMs renormalize within the cut game
        f.write_text(json.dumps(g, indent=1) + "\n", encoding="utf-8")
        rewrite_pgn(f.with_suffix(".pgn"), i + 1, new_result)
        print(f"{f.name}: {side} resigns after ply {g['moves'][-1]['ply']} "
              f"(own wp {own:.1%}) — {old_plies} -> {len(g['moves'])} plies, "
              f"{new_result}")
        applied += 1
    print(f"\ndone: {applied} resigned, {skipped_flip} comeback(s) left alone, "
          f"{untouched} untouched")


if __name__ == "__main__":
    main()

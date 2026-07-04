#!/usr/bin/env python3
"""Pick the top-3 featured games — Leela (Lc0) comebacks — and rewrite
games/manifest.json to list ONLY those three, in rank order.

Metric (per game, Leela's perspective, values P1 N3 B3 R5 Q9):
  deficit[p]   = opponent material - Leela material after ply p
  sustained[p] = min(deficit[p .. p+3])   # must survive ~2 full moves, so
                                          # pending recaptures don't count
  score        = max(sustained)           # "deepest real hole she climbed out of"
  integral     = sum of positive sustained values  # tie-break: suffered longest

Ranking (strict tiers — a -1 win outranks a -9 draw):
  1. Leela WINS,  by (score, integral, plies) descending
  2. DRAWS,       same order, only to fill remaining slots
  3. Losses never qualify.

The full corpus stays on disk; only the manifest is trimmed. Rerun
generate_games.py --finalize-only first if delays need re-normalizing.
"""
import json
import sys
from pathlib import Path

import chess

GAMES = Path(__file__).resolve().parent.parent / "games"
VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
          chess.ROOK: 5, chess.QUEEN: 9}
WINDOW = 4  # plies the deficit must persist
TOP_N = 3


def material(board, color):
    return sum(v * len(board.pieces(pt, color)) for pt, v in VALUES.items())


def game_metrics(data):
    leela_color = chess.WHITE if data["white"] == "Lc0" else chess.BLACK
    board = chess.Board()
    deficit = []
    for m in data["moves"]:
        board.push(chess.Move.from_uci(m["uci"]))
        deficit.append(material(board, not leela_color) - material(board, leela_color))
    sustained = [min(deficit[i:i + WINDOW]) for i in range(len(deficit))]
    leela_result = "1-0" if leela_color == chess.WHITE else "0-1"
    return {
        "leelaWin": data["result"] == leela_result,
        "draw": data["result"] == "1/2-1/2",
        "score": max(sustained) if sustained else 0,
        "integral": sum(s for s in sustained if s > 0),
        "plies": len(deficit),
        "peakPly": sustained.index(max(sustained)) + 1 if sustained else 0,
    }


def main():
    manifest = json.loads((GAMES / "manifest.json").read_text())
    rows = []
    for entry in manifest["games"]:
        data = json.loads((GAMES / entry["data"]).read_text())
        m = game_metrics(data)
        rows.append((entry, data, m))
        tag = "LEELA WIN" if m["leelaWin"] else data["result"]
        print(f"{entry['data']}  {tag:>9}  score={m['score']:>3}  "
              f"integral={m['integral']:>4}  plies={m['plies']:>3}  "
              f"peak at ply {m['peakPly']}")

    rank_key = lambda r: (r[2]["score"], r[2]["integral"], r[2]["plies"])
    wins = sorted((r for r in rows if r[2]["leelaWin"]), key=rank_key, reverse=True)
    draws = sorted((r for r in rows if r[2]["draw"]), key=rank_key, reverse=True)

    featured = wins[:TOP_N]
    if len(featured) < TOP_N:
        fill = draws[:TOP_N - len(featured)]
        featured += fill
        print(f"\nOnly {len(wins)} Leela win(s); filling with {len(fill)} draw(s).")
    if not featured:
        print("\nNo Leela wins or draws in the corpus; manifest left untouched.")
        sys.exit(1)

    for i, (entry, data, _) in enumerate(featured, 1):
        entry["label"] = f"Game {i}: {data['white']} vs {data['black']} ({data['result']})"
    manifest["games"] = [entry for entry, _, _ in featured]
    (GAMES / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                         encoding="utf-8")

    print(f"\nManifest now lists {len(featured)} game(s):")
    for entry, _, m in featured:
        print(f"  {entry['data']}  sustained deficit {m['score']} "
              f"(peak ply {m['peakPly']}, integral {m['integral']})")
    if featured and featured[0][2]["score"] <= 0:
        print("Note: the top game was never truly materially behind.")


if __name__ == "__main__":
    main()

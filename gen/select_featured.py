#!/usr/bin/env python3
"""Pick the top-N most exciting games (default 5; pass N as the only
argument) and rewrite games/manifest.json to list ONLY those, in rank order.

Excitement is scored by excitement.py (eight spectator factors: sacrifices,
compensation, comebacks, forced sequences, engine divergence, eval swings,
king danger, finish). Ranking is by raw score; at a tie a decisive game
outranks a draw.

The full corpus stays on disk; only the manifest is trimmed. Meant to run
after an overnight generate_games.py session: overproduce, then cull.
"""
import json
import sys
from pathlib import Path

from excitement import evaluate

GAMES = Path(__file__).resolve().parent.parent / "games"
TOP_N = int(sys.argv[1]) if len(sys.argv) > 1 else 5


def main():
    manifest = json.loads((GAMES / "manifest.json").read_text())
    rows = []
    for entry in manifest["games"]:
        data = json.loads((GAMES / entry["data"]).read_text())
        score, raw = evaluate(GAMES / entry["data"])
        decisive = data["result"] in ("1-0", "0-1")
        rows.append((entry, data, score, raw, decisive))
    if not rows:
        sys.exit("manifest lists no games")

    rows.sort(key=lambda r: (r[3], r[4]), reverse=True)
    featured = rows[:TOP_N]

    for i, (entry, data, score, _, _) in enumerate(featured, 1):
        vs = f"{data['white']} vs {data['black']}"
        if data.get("opening"):
            vs += f" — {data['opening']}"
        entry["label"] = f"Game {i}: {vs} ({data['result']})"
    manifest["games"] = [entry for entry, *_ in featured]
    (GAMES / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                         encoding="utf-8")

    print(f"\nManifest now lists {len(featured)} game(s):")
    for entry, data, score, raw, _ in featured:
        print(f"  {entry['data']}  {score:>2}/10 (raw {raw:.3f})  {entry['label']}")


if __name__ == "__main__":
    main()

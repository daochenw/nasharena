# nasharena

A small static website that visualizes chess games played by an **AlphaZero-style
engine (Leela Chess Zero / Lc0)** against a classical search engine (**Stockfish**).

No model training involved — you download ready-made Lc0 network weights and the
two engine binaries, generate a handful of games offline, and the site replays
them move by move.

## How it works

```
gen/generate_games.py   →  games/*.pgn + games/manifest.json  →  index.html (replay)
```

- **`index.html`** — a single static page (no build step). It loads
  [chessground](https://github.com/lichess-org/chessground) and
  [chess.js](https://github.com/jhlywa/chess.js) from a CDN, reads
  `games/manifest.json`, and lets you step through each game (arrow keys,
  slider, click moves). If no games are present it falls back to an embedded
  placeholder game so the page still works.
- **`gen/generate_games.py`** — drives Lc0 and Stockfish over UCI with
  [`python-chess`](https://python-chess.readthedocs.io/) and writes PGNs.

## View the site

Open `index.html` directly, or serve the folder (needed for the page to fetch
real games from `games/`):

```sh
python3 -m http.server 8000
# then visit http://localhost:8000
```

## Generate real games

1. Install engines + a network (no training):
   - **Stockfish** — https://stockfishchess.org/download/
   - **Lc0** — https://github.com/LeelaChessZero/lc0/releases
   - **Lc0 network (weights)** — https://lczero.org/play/networks/bestnets/

2. Install the Python dep:

   ```sh
   pip install -r gen/requirements.txt
   ```

3. Generate:

   ```sh
   python gen/generate_games.py \
     --lc0 /path/to/lc0 --lc0-weights /path/to/network.pb.gz \
     --stockfish /path/to/stockfish \
     --games 4 --movetime 1.0
   ```

   This writes `games/game-001.pgn …` and `games/manifest.json`. Reload the site.

   Strength/speed knobs: `--movetime` (seconds per move) or `--nodes`
   (fixed node count); `--swap-colors` to alternate which engine plays White.

## Note on "AI vs non-AI"

Lc0 is a neural-net engine (policy/value net + MCTS), close to AlphaZero in
spirit. Modern Stockfish uses a small **NNUE** evaluation net, so it isn't
purely handcrafted anymore — but it's still a classical alpha-beta search
engine, which is the contrast on display here. For a strictly non-neural
opponent, run Stockfish with `Use NNUE = false` or an older classical build.

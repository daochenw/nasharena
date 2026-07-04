#!/usr/bin/env python3
"""Play Lc0 (AlphaZero-style) vs Stockfish and write PGNs + telemetry for the site.

This captures *how each engine decided* every move, so the website can
anthropomorphize the players:

  Lc0 (MCTS)      -> visit-distribution entropy (how split its simulations were),
                     top-2 Q margin, policy->visit "changed its mind", WDL.
  Stockfish (a/b) -> search instability (how often its best move flipped while
                     deepening), top-2 eval margin, settle depth, eval, candidates.

Both engines run at a FIXED NODE BUDGET per move so the signals are comparable
across moves and games. For each game we write:

  games/game-00N.pgn   human-readable move list
  games/game-00N.json  per-move telemetry consumed by index.html
  games/manifest.json  index of games

Excitement engineering (the corpus is culled to a featured few, so quality
comes from volume + selection):

  * Every game is seeded from a SHARP OPENING BOOK (gambits, opposite-side
    castling storms). Book moves are stamped "book": true with eval bars but
    no thinking telemetry - the site plays them "from memory" - and the JSON
    carries "bookPlies" so the renderer knows where thinking begins. Each
    opening is played as a colour-reversed pair on consecutive attempts.
  * ADJUDICATION cuts games short once both engines' eval bars agree:
    win adjudicated after WIN_PLIES consecutive plies beyond WIN_WP;
    draw adjudicated after DRAW_PLIES consecutive dead-equal, low-divergence
    plies. An adjudicated draw whose evals NEVER left the drama band is a
    dud and is DISCARDED (not written) - its slot goes to a fresh attempt.
  * --hours N runs time-boxed: keep generating until the deadline (checked
    between games), then finalize. Overproduce, then cull with
    select_featured.py (top 5 by excitement.py score).

Checkpoint/resume: each game is written to disk the moment it finishes,
already complete: difficulty is normalized WITHIN the game (per engine, from
the raw signal kept as "rawDifficulty") and the eased render delay
(0.4s .. 4.0s) is stamped into every move. Killing the run costs at most the
in-flight game; rerunning the same command resumes, and finished games never
change when more games are added.

A finalize pass (run at the end, or standalone via --finalize-only) rebuilds
the manifest and re-stamps the per-game normalization for every game on disk
(idempotent; also upgrades games generated under the old global scheme).

Example (paths default to the bundled engines/ binaries):
  python3 gen/generate_games.py --games 20
"""
import argparse
import json
import math
import re
import time
from pathlib import Path

import chess
import chess.engine
import chess.pgn

REPO = Path(__file__).resolve().parent.parent
ENGINES = REPO / "engines"

# Render-delay mapping (shared by both engines so timing stays an honest readout).
DELAY_MIN_MS = 400
DELAY_MAX_MS = 4000
EASE_EXP = 1.8           # >1: most moves brisk, only hard ones visibly stall
MAX_CANDIDATES = 4       # ghost arrows per move

# Adjudication / boredom control. All thresholds act on the per-ply eval bars
# (White-POV win probs from BOTH engines), post-book plies only.
WIN_WP = 0.95            # both engines this sure, for the same side...
WIN_PLIES = 10           # ...this many consecutive plies -> adjudicate the win
DRAW_BAND = 0.055        # |wp - 0.5| within this = dead equal (per engine)
DRAW_DIV = 0.06          # and the engines agree within this
DRAW_PLIES = 40          # consecutive dead plies -> adjudicate the draw
DRAMA_WP = 0.15          # an adjudicated draw that never saw |wp - 0.5| >= this
                         # had no story: discard it entirely

GAME_RE = re.compile(r"game-(\d{3})\.json$")

# Sharp opening book: every line is a known theoretical battleground (gambits,
# sac lines, opposite-side castling races). From the standard start these two
# engines drift into correct, samey, drawish chess; the book forces imbalance
# and character. Each entry is played TWICE on consecutive attempts, colours
# reversed, TCEC-style. All lines are machine-verified legal (SAN from start).
OPENINGS = [
    ("King's Gambit, Kieseritzky", "e4 e5 f4 exf4 Nf3 g5 h4 g4 Ne5 Nf6"),
    ("Muzio Gambit", "e4 e5 f4 exf4 Nf3 g5 Bc4 g4 O-O gxf3 Qxf3 Qf6"),
    ("Evans Gambit", "e4 e5 Nf3 Nc6 Bc4 Bc5 b4 Bxb4 c3 Ba5 d4 exd4 O-O"),
    ("Traxler Counterattack", "e4 e5 Nf3 Nc6 Bc4 Nf6 Ng5 Bc5 Nxf7 Bxf2+ Kxf2 Nxe4+"),
    ("Frankenstein-Dracula", "e4 e5 Nc3 Nf6 Bc4 Nxe4 Qh5 Nd6 Bb3 Nc6 Nb5 g6 "
                             "Qf3 f5 Qd5 Qe7 Nxc7+ Kd8 Nxa8 b6"),
    ("Danish Gambit", "e4 e5 d4 exd4 c3 dxc3 Bc4 cxb2 Bxb2 d5"),
    ("Cochrane Gambit", "e4 e5 Nf3 Nf6 Nxe5 d6 Nxf7 Kxf7 d4"),
    ("Najdorf, Poisoned Pawn", "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Bg5 e6 "
                               "f4 Qb6 Qd2 Qxb2 Rb1 Qa3"),
    ("Dragon, Yugoslav Attack", "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 g6 Be3 Bg7 "
                                "f3 O-O Qd2 Nc6 Bc4 Bd7 O-O-O Rc8"),
    ("Sicilian Sveshnikov", "e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 Nf6 Nc3 e5 Ndb5 d6 "
                            "Bg5 a6 Na3 b5"),
    ("Velimirovic Attack", "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 Nc6 Bc4 e6 "
                           "Be3 Be7 Qe2 a6 O-O-O Qc7"),
    ("Marshall Attack", "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 O-O "
                        "c3 d5 exd5 Nxd5 Nxe5 Nxe5 Rxe5 c6"),
    ("Winawer, Poisoned Pawn", "e4 e6 d4 d5 Nc3 Bb4 e5 c5 a3 Bxc3+ bxc3 Ne7 "
                               "Qg4 Qc7 Qxg7 Rg8 Qxh7 cxd4"),
    ("Botvinnik Semi-Slav", "d4 d5 c4 c6 Nc3 Nf6 Nf3 e6 Bg5 dxc4 e4 b5 e5 h6 "
                            "Bh4 g5 Nxg5 hxg5 Bxg5 Nbd7"),
    ("King's Indian, Mar del Plata", "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 e5 "
                                     "O-O Nc6 d5 Ne7 Ne1 Nd7 Nd3 f5"),
    ("Benko Gambit Accepted", "d4 Nf6 c4 c5 d5 b5 cxb5 a6 bxa6 g6 Nc3 Bxa6 "
                              "e4 Bxf1 Kxf1 d6"),
    ("Grunfeld Exchange", "d4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5 e4 Nxc3 bxc3 Bg7 "
                          "Bc4 c5 Ne2 Nc6 Be3 O-O O-O Bg4 f3 Na5"),
    ("Modern Benoni, Flick-Knife", "d4 Nf6 c4 c5 d5 e6 Nc3 exd5 cxd5 d6 e4 g6 "
                                   "f4 Bg7 Bb5+ Nfd7"),
    ("Max Lange Attack", "e4 e5 Nf3 Nc6 Bc4 Nf6 d4 exd4 O-O Bc5 e5 d5 exf6 dxc4 "
                         "Re1+ Be6 Ng5 Qd5 Nc3 Qf5 Nce4"),
]

# Parses one Lc0 VerboseMoveStats line, e.g.:
#  "g1f3  (159 ) N:     193 (+ 6) (P: 15.68%) (WL:  0.07194) (D: 0.574) ... (Q:  0.07194) ..."
LC0_LINE = re.compile(
    r"^\s*(?P<move>[a-h][1-8][a-h][1-8][qrbn]?)\s+\(\s*\d+\s*\)\s+"
    r"N:\s*(?P<n>\d+).*?\(P:\s*(?P<p>[\d.]+)%\).*?"
    r"\(WL:\s*(?P<wl>-?[\d.]+)\).*?\(D:\s*(?P<d>-?[\d.]+)\)"
)


def sq(square):
    return chess.square_name(square)


def wp_from_cp(cp):
    """Centipawns (white POV) -> white win probability."""
    return 1.0 / (1.0 + 10 ** (-cp / 400.0))


def game_result(board):
    """Result string if the game is over, else None.

    Deliberately NOT is_game_over(claim_draw=True): that auto-claims a draw
    as soon as one is merely claimable - even a repetition that only WOULD
    occur one move ahead - handing the losing side a draw the winning side
    would never agree to (game-002: K+B+P vs bare king adjudicated 1/2-1/2).
    Draws are adjudicated only once they have actually occurred on the board
    (a genuine threefold repetition, or 100 halfmoves without progress);
    engines see both rules in search and steer clear while winning.
    """
    if board.is_game_over():
        return board.result()
    if board.is_repetition(3) or board.halfmove_clock >= 100:
        return "1/2-1/2"
    return None


def eval_wp(engine, board, nodes, res):
    """Quick White-POV win-prob estimate from one engine (own assessment).

    Used to give EACH engine an opinion on EVERY position, so the site can show
    Lc0's and Stockfish's evaluations as two separate bars (and the gap between
    them = how much the two minds disagree about who is winning).
    `res` is game_result(board), computed once by the caller for both engines.
    """
    if res is not None:
        return {"1-0": 1.0, "0-1": 0.0}.get(res, 0.5)
    info = engine.analyse(board, chess.engine.Limit(nodes=nodes))
    cp = info["score"].white().score(mate_score=100000)
    return round(wp_from_cp(cp), 4)


# --------------------------------------------------------------------------- #
# Lc0 analysis
# --------------------------------------------------------------------------- #
def analyse_lc0(engine, board, nodes):
    """Return telemetry dict for Lc0's decision, with best_move = most-visited."""
    rows = []  # list of (move_uci, N, P_frac, WL, D)
    with engine.analysis(board, chess.engine.Limit(nodes=nodes)) as a:
        for info in a:
            s = info.get("string")
            if not s:
                continue
            m = LC0_LINE.match(s)
            if m:
                rows.append((
                    m.group("move"),
                    int(m.group("n")),
                    float(m.group("p")) / 100.0,
                    float(m.group("wl")),
                    float(m.group("d")),
                ))

    # Keep only legal moves that actually got visits; pick the freshest stats
    # (Lc0 prints stats repeatedly while searching; later lines win).
    latest = {}
    for move, n, p, wl, d in rows:
        latest[move] = (n, p, wl, d)
    legal = {mv.uci() for mv in board.legal_moves}
    cand = [(mv, *latest[mv]) for mv in latest if mv in legal]
    if not cand:
        raise RuntimeError("Lc0 produced no parseable move stats")

    total_n = sum(c[1] for c in cand) or 1
    # Best move = most visited.
    cand.sort(key=lambda c: c[1], reverse=True)
    best_uci, best_n, best_p, best_wl, best_d = cand[0]

    # Visit-distribution entropy (normalized to 0..1 by log of #candidates).
    visited = [c for c in cand if c[1] > 0]
    probs = [c[1] / total_n for c in visited]
    H = -sum(p * math.log(p) for p in probs if p > 0)
    entropy_norm = H / math.log(len(visited)) if len(visited) > 1 else 0.0

    # Top-2 Q(=WL) margin (lower margin = harder choice).
    q_margin = abs(cand[0][3] - cand[1][3]) if len(cand) > 1 else 1.0

    # "Changed its mind": gut instinct (max policy) != most-visited move.
    policy_best = max(cand, key=lambda c: c[2])[0]
    changed_mind = policy_best != best_uci

    # White-POV expected score (W + D/2) from the most-visited move's WL.
    # WL is the side-to-move expectation in [-1, 1]; (WL+1)/2 maps to [0, 1].
    mover_score = (best_wl + 1) / 2.0
    wp_white = mover_score if board.turn == chess.WHITE else 1.0 - mover_score

    best_move = chess.Move.from_uci(best_uci)
    candidates = []
    for mv, n, p, wl, d in cand[:MAX_CANDIDATES]:
        m = chess.Move.from_uci(mv)
        candidates.append({
            "uci": mv, "from": sq(m.from_square), "to": sq(m.to_square),
            "san": board.san(m), "share": round(n / total_n, 4),
            "visits": n, "policy": round(p, 4), "q": round(wl, 4),
        })

    raw = entropy_norm  # primary difficulty signal for Lc0
    return {
        "best_move": best_move,
        "wpWhite": round(wp_white, 4),
        "changedMind": changed_mind,
        "candidates": candidates,
        "viz": "intuition",
        "extra": {"entropyNorm": round(entropy_norm, 4),
                  "qMargin": round(q_margin, 4),
                  "totalVisits": total_n},
        "raw": raw,
    }


# --------------------------------------------------------------------------- #
# Stockfish analysis
# --------------------------------------------------------------------------- #
def analyse_sf(engine, board, nodes, multipv=3):
    """Return telemetry dict for Stockfish, capturing iterative-deepening dynamics."""
    per_depth_root = {}                 # depth -> best root move (multipv 1)
    score_by_depth = {}                 # depth -> {multipv: cp_white}
    pv_by_depth = {}                    # depth -> {multipv: [uci,...]}
    with engine.analysis(board, chess.engine.Limit(nodes=nodes), multipv=multipv) as a:
        for info in a:
            d, pv, mpv, sc = (info.get("depth"), info.get("pv"),
                              info.get("multipv"), info.get("score"))
            if d is None or sc is None:
                continue
            cp = sc.white().score(mate_score=100000)
            if mpv == 1 and pv:
                per_depth_root[d] = pv[0].uci()
            if mpv in range(1, multipv + 1):
                score_by_depth.setdefault(d, {})[mpv] = cp
                if pv:
                    pv_by_depth.setdefault(d, {})[mpv] = [m.uci() for m in pv]

    if not per_depth_root:
        raise RuntimeError("Stockfish produced no PV")

    depths = sorted(per_depth_root)
    roots = [per_depth_root[d] for d in depths]
    max_depth = depths[-1]
    changes = sum(1 for i in range(1, len(roots)) if roots[i] != roots[i - 1])
    last_change_depth = next(
        (depths[i] for i in range(len(roots) - 1, 0, -1) if roots[i] != roots[i - 1]),
        depths[0])
    settle_frac = last_change_depth / max_depth if max_depth else 0.0

    final = score_by_depth[max_depth]
    best_uci = per_depth_root[max_depth]
    best_cp = final.get(1, 0)
    margin_cp = abs(final[1] - final[2]) if 2 in final else 1000

    # Difficulty: kept flipping (instability) + late settle + close runner-up.
    raw = (0.5 * min(changes / 8.0, 1.0)
           + 0.3 * settle_frac
           + 0.2 * math.exp(-margin_cp / 80.0))
    changed_mind = settle_frac >= 0.6 and changes > 0

    wp_white = wp_from_cp(best_cp)
    best_move = chess.Move.from_uci(best_uci)

    # Candidate ghosts: top-N final PVs, thickness via softmax over cp (mover POV).
    fin_pvs = pv_by_depth.get(max_depth, {})
    cps = []
    for mpv in sorted(fin_pvs):
        cp_mover = final[mpv] if board.turn == chess.WHITE else -final[mpv]
        cps.append((mpv, cp_mover, fin_pvs[mpv]))
    assert cps, "fin_pvs empty despite per_depth_root non-empty"
    mx = max(c[1] for c in cps)
    exps = [math.exp((c[1] - mx) / 80.0) for c in cps]
    ssum = sum(exps) or 1.0
    candidates = []
    for (mpv, cp_mover, pv), e in zip(cps, exps):
        m = chess.Move.from_uci(pv[0])
        candidates.append({
            "uci": pv[0], "from": sq(m.from_square), "to": sq(m.to_square),
            "san": board.san(m), "share": round(e / ssum, 4),
            "cp": (final[mpv] if board.turn == chess.WHITE else -final[mpv]),
            "line": pv[:6],
        })

    # Root-move history (for the "deepening" visualization).
    root_history = [[d, per_depth_root[d]] for d in depths]

    return {
        "best_move": best_move,
        "wpWhite": round(wp_white, 4),
        "changedMind": changed_mind,
        "candidates": candidates,
        "viz": "calculation",
        "extra": {"changes": changes, "settleDepth": last_change_depth,
                  "maxDepth": max_depth, "marginCp": margin_cp,
                  "rootHistory": root_history},
        "raw": raw,
    }


# --------------------------------------------------------------------------- #
# Game play
# --------------------------------------------------------------------------- #
def play_game(lc0, sf, lc0_white, lc0_nodes, sf_nodes,
              lc0_eval_nodes, sf_eval_nodes, round_no, opening):
    """Play one game from a book line. Returns (pgn_game, data);
    data is None when the game was adjudicated a dud and discarded."""
    opening_name, opening_sans = opening
    board = chess.Board()
    game = chess.pgn.Game()
    wn, bn = ("Lc0", "Stockfish") if lc0_white else ("Stockfish", "Lc0")
    game.headers.update({
        "Event": "nasharena: Lc0 vs Stockfish", "Site": "nasharena.ai",
        "Round": str(round_no), "White": wn, "Black": bn,
        "Opening": opening_name,
    })
    node = game
    moves = []
    ply = 0

    # Book prologue: both players bang out the prepared line from memory.
    # Eval bars are real (cheap probes) but there is no thinking telemetry.
    for san_in in opening_sans.split():
        ply += 1
        white_to_move = board.turn == chess.WHITE
        is_lc0 = white_to_move == lc0_white
        mv = board.parse_san(san_in)
        san = board.san(mv)
        board.push(mv)
        node = node.add_variation(mv)
        res = game_result(board)
        moves.append({
            "ply": ply,
            "side": "white" if white_to_move else "black",
            "engine": "Lc0" if is_lc0 else "Stockfish",
            "uci": mv.uci(), "san": san, "book": True,
            "evalLc0": eval_wp(lc0, board, lc0_eval_nodes, res),
            "evalSF": eval_wp(sf, board, sf_eval_nodes, res),
            "changedMind": False, "candidates": [], "viz": {}, "extra": {},
        })
    book_plies = ply

    # Adjudication state (book plies excluded: a gambit's eval dip is theory,
    # not drama, and prepared lines must not trip the dead-draw counter).
    result, termination = None, "natural"
    drama = 0.0          # furthest either eval bar ever strayed from 0.5
    dead_run = 0         # consecutive dead-equal, engines-agree plies
    w_run = b_run = 0    # consecutive both-engines-sure plies per side

    while game_result(board) is None:
        ply += 1
        white_to_move = board.turn == chess.WHITE
        is_lc0 = (white_to_move and lc0_white) or (not white_to_move and not lc0_white)
        engine_name = "Lc0" if is_lc0 else "Stockfish"
        print(f"    ply {ply:>3} {engine_name:<9} thinking...", flush=True)

        if is_lc0:
            t = analyse_lc0(lc0, board, lc0_nodes)
        else:
            t = analyse_sf(sf, board, sf_nodes)

        mv = t.pop("best_move")
        san = board.san(mv)
        board.push(mv)
        node = node.add_variation(mv)

        # Both engines assess the resulting position, on a shared win-prob scale.
        res = game_result(board)
        eval_lc0 = eval_wp(lc0, board, lc0_eval_nodes, res)
        eval_sf = eval_wp(sf, board, sf_eval_nodes, res)

        moves.append({
            "ply": ply,
            "side": "white" if white_to_move else "black",
            "engine": engine_name,
            "uci": mv.uci(), "san": san,
            "evalLc0": eval_lc0, "evalSF": eval_sf,
            "changedMind": t["changedMind"],
            "candidates": t["candidates"], "viz": t["viz"],
            "extra": t["extra"], "rawDifficulty": round(t["raw"], 6),
        })

        # ---- adjudication bookkeeping (skip if the game just ended) ----
        if res is not None:
            continue
        drama = max(drama, abs(eval_lc0 - 0.5), abs(eval_sf - 0.5))
        dead = (max(abs(eval_lc0 - 0.5), abs(eval_sf - 0.5)) <= DRAW_BAND
                and abs(eval_lc0 - eval_sf) <= DRAW_DIV)
        dead_run = dead_run + 1 if dead else 0
        w_run = w_run + 1 if min(eval_lc0, eval_sf) >= WIN_WP else 0
        b_run = b_run + 1 if max(eval_lc0, eval_sf) <= 1.0 - WIN_WP else 0
        if w_run >= WIN_PLIES or b_run >= WIN_PLIES:
            result = "1-0" if w_run >= WIN_PLIES else "0-1"
            termination = "adjudicated"
            print(f"    adjudicated {result}: both engines >= {WIN_WP:.0%} "
                  f"for {WIN_PLIES} plies", flush=True)
            break
        if dead_run >= DRAW_PLIES:
            result, termination = "1/2-1/2", "adjudicated"
            print(f"    adjudicated draw: dead equal for {DRAW_PLIES} plies",
                  flush=True)
            break

    if result is None:
        result = game_result(board)
    game.headers["Result"] = result
    if termination == "adjudicated":
        game.headers["Termination"] = "adjudication"
        # A cut-short draw that never had a story is a dud: reclaim the slot.
        if result == "1/2-1/2" and drama < DRAMA_WP:
            print(f"    discarded: no drama (peak dev from 0.5 was {drama:.3f})",
                  flush=True)
            return game, None
    return game, {"white": wn, "black": bn, "result": result,
                  "opening": opening_name, "bookPlies": book_plies,
                  "termination": termination, "moves": moves}


# --------------------------------------------------------------------------- #
# Normalization (per game) + finalize sweep (idempotent)
# --------------------------------------------------------------------------- #
def normalize_game(g):
    """Stamp difficulty/delayMs from rawDifficulty, normalized within THIS game
    (per engine): each game's hardest decision paces its own drama, and the
    JSON is self-contained the moment it is written. Book moves carry no
    difficulty signal (they are recalled, not computed) and are skipped."""
    real = [m for m in g["moves"] if not m.get("book")]
    by_engine = {}
    for m in real:
        by_engine.setdefault(m["engine"], []).append(m["rawDifficulty"])
    bounds = {eng: (min(v), max(v)) for eng, v in by_engine.items()}
    for m in real:
        lo, hi = bounds[m["engine"]]
        norm = (m["rawDifficulty"] - lo) / (hi - lo) if hi > lo else 0.0
        eased = norm ** EASE_EXP
        m["difficulty"] = round(norm, 4)
        m["delayMs"] = int(round(DELAY_MIN_MS + (DELAY_MAX_MS - DELAY_MIN_MS) * eased))


def finalize(out):
    """Re-stamp per-game normalization for ALL games on disk and rewrite the
    full manifest in numeric order."""
    files = sorted(f for f in out.glob("game-*.json") if GAME_RE.search(f.name))
    if not files:
        print("finalize: no games on disk, nothing to do")
        return
    games = [(f, json.loads(f.read_text())) for f in files]

    for f, g in games:
        normalize_game(g)
        f.write_text(json.dumps(g, indent=1) + "\n", encoding="utf-8")

    manifest = {"games": []}
    for i, (f, g) in enumerate(games, 1):
        vs = f"{g['white']} vs {g['black']}"
        if g.get("opening"):
            vs += f" — {g['opening']}"
        manifest["games"].append({
            "file": f.name.replace(".json", ".pgn"), "data": f.name,
            "label": f"Game {i}: {vs} ({g['result']})",
        })
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                       encoding="utf-8")
    print(f"finalize: normalized {len(games)} game(s), manifest lists all of them")
    print("          (run select_featured.py to trim the manifest to the top 5)")


def main():
    p = argparse.ArgumentParser(description="Generate Lc0 vs Stockfish games + telemetry.")
    p.add_argument("--lc0", default=str(ENGINES / "lc0-src/build/release/lc0"))
    p.add_argument("--lc0-weights",
                   default=str(ENGINES / "t1-512x15x8h-distilled-swa-3395000.pb.gz"))
    p.add_argument("--lc0-backend", default="metal")
    p.add_argument("--stockfish",
                   default=str(ENGINES / "stockfish/stockfish-macos-m1-apple-silicon"))
    p.add_argument("--games", type=int, default=None,
                   help="Target corpus size on disk (default: 20, or unlimited "
                        "when --hours is set).")
    p.add_argument("--hours", type=float, default=None,
                   help="Wall-clock budget: keep generating until the deadline "
                        "(checked between games; the in-flight game finishes).")
    p.add_argument("--lc0-nodes", type=int, default=50000)
    p.add_argument("--sf-nodes", type=int, default=100000)
    p.add_argument("--lc0-eval-nodes", type=int, default=256,
                   help="Nodes for Lc0's per-position eval bar (value head).")
    p.add_argument("--sf-eval-nodes", type=int, default=16000,
                   help="Nodes for Stockfish's per-position eval bar. Runs every "
                        "ply, so keep it far below --sf-nodes; 16k barely moves "
                        "the bar vs 100k and roughly halves SF's per-ply cost.")
    p.add_argument("--lc0-threads", type=int, default=2)
    p.add_argument("--out", default=str(REPO / "games"))
    p.add_argument("--finalize-only", action="store_true",
                   help="Skip generation; re-stamp per-game normalization and "
                        "rebuild the manifest.")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.finalize_only:
        finalize(out)
        return

    # Resume: games already checkpointed on disk count toward the target.
    done = {int(GAME_RE.search(f.name).group(1))
            for f in out.glob("game-*.json") if GAME_RE.search(f.name)}
    target = args.games if args.games is not None else (None if args.hours else 20)
    deadline = time.monotonic() + args.hours * 3600 if args.hours else None
    if done:
        print(f"resume: {len(done)} game(s) already on disk")
    if target is not None and len(done) >= target:
        finalize(out)
        return

    # Collision limits let MCTS gather GPU-sized batches; without them the
    # Metal backend is latency-bound at ~120 nps (14-35ms per tiny call).
    lc0 = chess.engine.SimpleEngine.popen_uci(
        [args.lc0, f"--weights={args.lc0_weights}", f"--backend={args.lc0_backend}",
         f"--threads={args.lc0_threads}", "--minibatch-size=1024",
         "--max-collision-events=32768", "--max-collision-visits=999999"])
    lc0.configure({"VerboseMoveStats": True})
    sf = chess.engine.SimpleEngine.popen_uci([args.stockfish])

    print(f"Lc0:       {args.lc0}  [{args.lc0_backend}]  ({args.lc0_nodes} nodes/move)")
    print(f"Stockfish: {args.stockfish}  ({args.sf_nodes} nodes/move)")
    if deadline is not None:
        print(f"deadline:  {args.hours:g}h from now (checked between games)")

    # Openings cycle by ATTEMPT (discards included, so a dud line is not
    # replayed forever): two consecutive attempts share an opening with
    # colours reversed. Resuming offsets the cycle past what's on disk.
    attempt = max(done, default=0)
    next_no = max(done, default=0) + 1
    saved = discarded = adjudicated = 0
    try:
        while True:
            if target is not None and len(done) >= target:
                break
            if deadline is not None and time.monotonic() >= deadline:
                print("deadline reached, stopping", flush=True)
                break
            opening = OPENINGS[(attempt // 2) % len(OPENINGS)]
            lc0_white = (attempt % 2 == 0)
            attempt += 1
            wn = "Lc0" if lc0_white else "Stockfish"
            bn = "Stockfish" if lc0_white else "Lc0"
            print(f"Game {next_no}: {wn} (W) vs {bn} (B) — {opening[0]}",
                  flush=True)
            game, data = play_game(lc0, sf, lc0_white, args.lc0_nodes,
                                   args.sf_nodes, args.lc0_eval_nodes,
                                   args.sf_eval_nodes, next_no, opening)
            if data is None:
                discarded += 1
                continue
            # Checkpoint immediately, fully normalized: a killed run costs at
            # most one game, and finished games are already display-ready.
            normalize_game(data)
            (out / f"game-{next_no:03d}.pgn").write_text(str(game) + "\n",
                                                         encoding="utf-8")
            (out / f"game-{next_no:03d}.json").write_text(
                json.dumps(data, indent=1) + "\n", encoding="utf-8")
            print(f"  result {data['result']}  ({len(data['moves'])} plies, "
                  f"{data['termination']})  [checkpointed]", flush=True)
            done.add(next_no)
            next_no += 1
            saved += 1
            if data["termination"] == "adjudicated":
                adjudicated += 1
    finally:
        lc0.quit()
        sf.quit()

    print(f"run: {saved} saved ({adjudicated} adjudicated), "
          f"{discarded} dud(s) discarded")
    finalize(out)


if __name__ == "__main__":
    main()

# Stockfish (WebAssembly)

The browser cannot spawn the native engine `gen/generate_games.py` drives, so
`stockfish.html` runs Stockfish compiled to WASM in a Web Worker instead.

- Build: `stockfish-18-lite-single` (single-threaded, so no COOP/COEP headers
  are needed; the lite net keeps the download at ~7 MB instead of ~113 MB).
- Source: npm [`stockfish@18.0.8`](https://www.npmjs.com/package/stockfish)
  (`bin/`), from https://github.com/nmrugg/stockfish.js — files copied verbatim.
- License: GPLv3, see `Copying.txt`.

To upgrade: `npm pack stockfish@<version>`, then copy
`package/bin/stockfish-<v>-lite-single.{js,wasm}` here and update the path in
`stockfish.html`.

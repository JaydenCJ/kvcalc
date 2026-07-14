# Contributing to kvcalc

Issues, discussions and pull requests are all welcome — this project aims
to stay small, zero-dependency at runtime, fully offline, and honest about
what a shape-level memory estimate can and cannot promise.

## Getting started

Requirements: Node.js >= 22.13 (for the stable `node:test` runner used by the suite).

```bash
git clone https://github.com/JaydenCJ/kvcalc.git
cd kvcalc
npm install            # installs typescript, the only devDependency
npm run build          # compile TypeScript to dist/
npm test               # build + 87 node:test tests
bash scripts/smoke.sh  # end-to-end CLI check against examples/
```

`scripts/smoke.sh` exercises the real CLI (the flagship 128k-in-24GiB
report and its exit codes, MLA and sliding-window arithmetic, the fit
solver, JSON determinism, the table gate) against the bundled example
configs and must print `SMOKE OK`.

## Before you open a pull request

1. `npx tsc -p tsconfig.json --noEmit` — the tree must type-check clean (strict mode is enforced).
2. `npm test` — all tests must pass.
3. `bash scripts/smoke.sh` — must print `SMOKE OK`.
4. Add tests for behavior changes; keep logic in pure, unit-testable
   modules (the math takes a normalized config and returns plain data —
   only `cli.ts` touches the filesystem or the process).
5. New expected values in tests must come with the paper arithmetic in a
   comment — no "expected = whatever the code printed" snapshots. A wrong
   constant here misinforms someone's hardware purchase.

## Ground rules

- **No runtime dependencies.** The zero-dependency install is a core
  feature; adding one needs justification in the PR and will usually be
  declined.
- No network calls, ever — kvcalc reads one local JSON file and prints
  numbers. It must work on an air-gapped box.
- Determinism is API: same config and flags, byte-identical output and
  exit code — no clocks, no randomness, no locale-dependent formatting.
- Stay key-driven: support new architectures by reading their config
  keys, never by matching `model_type` strings against a hardcoded list.
- Bits-per-weight constants must cite their block layout (or be flagged
  as measured averages) in `src/dtype.ts` and `docs/kv-math.md`.
- Code comments and doc comments are written in English.

## Reporting bugs

Please include: `kvcalc --version` output, the exact command line, the
config.json (or the relevant keys), and the number you expected with how
you derived it. Discrepancy reports against real weight files or a real
runtime's allocator are especially valuable — say which runtime and
version you compared against.

## Security

Do not open public issues for security problems (e.g. a crafted
config.json that hangs the parser); use GitHub private vulnerability
reporting on this repository instead.

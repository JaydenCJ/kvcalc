#!/usr/bin/env bash
# Smoke test for kvcalc: exercises the real CLI end to end against the
# committed example configs. No network, idempotent, runs from a clean
# checkout (after `npm install`). Prints "SMOKE OK" on success.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$(pwd)"

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

fail() {
  echo "SMOKE FAIL: $1" >&2
  exit 1
}

GQA=examples/gqa-8b.json
MLA=examples/mla-moe-236b.json
SWA=examples/swa-hybrid-12b.json

# 1. Build (idempotent).
npm run build >/dev/null 2>&1 || fail "npm run build failed"
CLI="node $ROOT/dist/cli.js"
echo "[smoke] build ok"

# 2. --version matches package.json; --help documents every command.
PKG_VERSION="$(node -p "require('$ROOT/package.json').version")"
CLI_VERSION="$($CLI --version)"
[ "$CLI_VERSION" = "$PKG_VERSION" ] || fail "--version mismatch: $CLI_VERSION != $PKG_VERSION"
HELP="$($CLI --help)"
for word in report table fit dtypes --vram --kv "Exit codes"; do
  echo "$HELP" | grep -q -- "$word" || fail "--help missing $word"
done
echo "[smoke] --help/--version ok ($CLI_VERSION)"

# 3. Error handling: bad flags, dtypes and inputs exit 2.
set +e
$CLI report "$GQA" --frobnicate >/dev/null 2>&1; [ $? -eq 2 ] || { set -e; fail "unknown flag should exit 2"; }
$CLI report >/dev/null 2>&1; [ $? -eq 2 ] || { set -e; fail "missing operand should exit 2"; }
$CLI report does-not-exist.json >/dev/null 2>&1; [ $? -eq 2 ] || { set -e; fail "missing file should exit 2"; }
$CLI report "$GQA" --weights q9_Z >/dev/null 2>&1; [ $? -eq 2 ] || { set -e; fail "bad dtype should exit 2"; }
printf '{not json' > "$WORKDIR/bad.json"
$CLI report "$WORKDIR/bad.json" >/dev/null 2>&1; [ $? -eq 2 ] || { set -e; fail "invalid JSON should exit 2"; }
set -e
echo "[smoke] error handling ok (exit 2)"

# 4. The flagship question: 128k ctx in 24GiB at q4_K_M weights — fits, exit 0.
REPORT="$($CLI report "$GQA" --ctx 128k --weights q4_K_M --vram 24GiB)" \
  || fail "flagship report should exit 0"
for want in "8.03 B" "16.00 GiB" "128.00 KiB/token" "FITS"; do
  echo "$REPORT" | grep -qF "$want" || fail "report output missing: $want"
done
echo "[smoke] report ok (128k @ q4_K_M fits 24GiB)"

# 5. The same question at bf16 weights does not fit — exit 1.
set +e
$CLI report "$GQA" --ctx 128k --vram 24GiB >/dev/null; RC=$?
set -e
[ "$RC" -eq 1 ] || fail "bf16 report should exit 1, got $RC"
echo "[smoke] budget gate ok (bf16 does not fit, exit 1)"

# 6. MLA and sliding-window architectures produce their distinctive numbers.
$CLI report "$MLA" --ctx 32k | grep -qF "67.50 KiB/token" || fail "MLA per-token cost wrong"
$CLI report "$SWA" --ctx 128k | grep -qF "40 layers capped @ 1024" || fail "SWA capping not reported"
echo "[smoke] MLA + sliding-window ok"

# 7. fit: full model context fits a 24GiB card at q4_K_M.
FIT="$($CLI fit "$GQA" --vram 24GiB --weights q4_K_M)" || fail "fit should exit 0"
echo "$FIT" | grep -qF "full model context FITS" || fail "fit verdict missing"
set +e
$CLI fit "$GQA" --vram 4GiB >/dev/null; RC=$?
set -e
[ "$RC" -eq 1 ] || fail "fit under weights should exit 1, got $RC"
echo "[smoke] fit ok (24GiB yes, 4GiB no)"

# 8. --json is valid, structurally intact and byte-identical across runs.
A="$($CLI report "$GQA" --ctx 128k --weights q4_K_M --json)"
B="$($CLI report "$GQA" --ctx 128k --weights q4_K_M --json)"
[ "$A" = "$B" ] || fail "report --json is not deterministic"
echo "$A" | node -e "
  const r = JSON.parse(require('fs').readFileSync(0, 'utf8'));
  if (r.tool !== 'kvcalc') throw new Error('tool');
  if (r.params.total !== 8030261248) throw new Error('params: ' + r.params.total);
  if (r.kv.bytes !== 16 * 1024 ** 3) throw new Error('kv: ' + r.kv.bytes);
  if (r.model.attention !== 'gqa') throw new Error('attention');
" || fail "report --json is not structurally intact"
echo "[smoke] --json + determinism ok"

# 9. table: grid marks fit/no-fit against the budget and gates the exit code.
set +e
TABLE="$($CLI table "$GQA" --weights q4_K_M --kv-list fp16,q4_0 --vram 8GiB)"; RC=$?
set -e
[ "$RC" -eq 1 ] || fail "table with an overflowing cell should exit 1, got $RC"
echo "$TABLE" | grep -q "✗" || fail "table missing ✗ marks"
echo "$TABLE" | grep -q "✓" || fail "table missing ✓ marks"
echo "$TABLE" | grep -q "128k" || fail "table missing the 128k row"
echo "[smoke] table ok (grid + exit gate)"

# 10. dtypes reference lists the block-quant receipts.
$CLI dtypes | grep -qF "256 weights in 210 bytes" || fail "dtypes missing q6_K provenance"
echo "[smoke] dtypes ok"

echo "SMOKE OK"

# Example configs

Four hand-written `config.json` files spanning the architecture space kvcalc
understands. The shapes are realistic (they match well-known public model
classes dimension-for-dimension) but the files are examples, not copies of
any distribution — a config is just geometry.

| File | Architecture | Params | Why it's here |
|---|---|---|---|
| `gqa-8b.json` | dense, GQA 32q/8kv | 8.03 B | the daily-driver local model; 128 KiB/token at fp16 |
| `mla-moe-236b.json` | MLA + MoE (160+2 experts) | 235.7 B (21.4 B active) | compressed KV: 67.5 KiB/token despite 128 heads |
| `swa-hybrid-12b.json` | GQA + 5:1 sliding windows, nested `text_config` | 11.8 B | 40 of 48 layers cap at 1024 tokens |
| `moe-a3b.json` | GQA + MoE (128 experts, top-8) | 30.5 B (3.3 B active) | big-total/small-active memory profile |

The test suite pins these numbers, so the examples double as regression
fixtures.

## Things to try

```bash
# The daily question, answered precisely:
kvcalc report examples/gqa-8b.json --ctx 128k --weights q4_K_M --vram 24GiB

# Why MLA exists — 60 layers, 128 heads, yet tiny per-token cache:
kvcalc report examples/mla-moe-236b.json --ctx 32k

# Sliding windows flatten the curve past 1024 tokens:
kvcalc table examples/swa-hybrid-12b.json --weights q4_K_M --vram 12GiB

# How much context a 24GiB card really buys at batch 4:
kvcalc fit examples/gqa-8b.json --vram 24GiB --weights q8_0 --batch 4 --overhead 1GiB

# Everything is scriptable:
kvcalc report examples/moe-a3b.json --json | node -pe \
  'JSON.parse(require("fs").readFileSync(0,"utf8")).params.active'
```

To analyze your own model, point kvcalc at the `config.json` you already
have next to the weights (or fetch it however you normally fetch files —
kvcalc itself never touches the network).

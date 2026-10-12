# Engineering log

One entry per problem: date, symptom, hypotheses, what fixed it. Week 5's debugging story comes from here.

## 2026-10-11 — Day 0
- **Symptom:** `uv init --package` created a `src/` layout and ran `git init` automatically.
- **Fix:** flattened to the spec layout (`tinyserve/` at repo root) via `[tool.uv.build-backend] module-root = ""`.

## 2026-10-11 — Step 1.4 logit equivalence
- **Result:** all 5 prompts within 1e-4 of HF on first run; max |diff| = 6.1e-05 on "The capital of France is".
- **Counterfactual:** swapping tanh-GELU for exact GELU raises max |diff| to 7.5e-02 (~1000x worse), so the test would catch it.
- **Note (weights):** checkpoint has 137.0M values vs 124M params: the extra 12.6M are 12 cached 1024x1024 causal masks (`h.N.attn.bias`), skipped by the loader.

## 2026-10-11 — Steps 1.5–1.7 sampler, KV cache, first graph
- **Result:** greedy output token-identical to HF `generate()` on 5 prompts (also with repetition_penalty=1.3); cached == uncached for greedy and seeded sampling; cached step logits match full forward within 1e-4.
- **Graph:** 512 tokens: 194.8 s uncached vs 21.0 s cached (9.3x). Cached per-token cost flat at ~41 ms; uncached grows 116 -> 381 ms.
- **Surprise:** cached cost at 32 tokens is 57 ms/token, higher than at 512. Fixed costs (prefill, cache allocation, first-step overhead) are spread over only 32 tokens.
- **Surprise:** uncached per-token cost is not a clean straight line at short lengths. On CPU a forward pass has a large fixed cost (reading all 124M weights from memory) that dominates until the sequence gets long.

## 2026-10-11 — Steps 2.1–2.2 static batching
- **Result:** batch-invariance passes first run at batch sizes 1, 2, 3, 8 (8 prompts, 1 to 40+ tokens, 30 greedy tokens), and with per-row seeds/temperatures/max_tokens.
- **Counterfactual 1:** positions from the column index (`arange`) instead of `mask.cumsum - 1` -> 2 of 3 padded rows produce different tokens. The test catches the classic bug.
- **Surprise (counterfactual 2):** removing the NaN guard produced *no* NaNs. torch 2.14 SDPA returns zeros for a fully masked query row; a hand-written `masked_fill(-inf).softmax()` returns NaN. Guard kept for custom attention (Qwen port), docstring corrected. Good design-doc item: the guide's warning was true for older PyTorch / manual attention, not for current SDPA.

## 2026-10-12 — Steps 2.3–2.4 slot KV cache, per-sequence decode
- **Design change:** model now calls `cache.advance(T)` instead of `cache.length += T`; with slots, "length" differs per row, so each cache type owns what advancing means.
- **Result:** one slot matches `generate()`; slots 0 and 5 holding 1-token and ~15-token prompts decoded together match each run alone.
- **Counterfactual:** drop the per-row key mask in decode -> the short row ("Hello") produces wrong tokens, the long row stays correct. The short row was attending to the unused tail of its slot (zeros / stale keys). Only the row shorter than L is hurt, which is why a test with equal lengths would never catch it.
- **Cost noted for the design doc:** each layer gathers `k[layer, slots, :, :L]`, which is a copy of every running sequence's history per layer per step. Paged attention kernels read blocks in place instead.

## 2026-10-12 — Steps 2.5–2.6 scheduler v1 (Week 2 done)
- **Result:** 20 requests (max_tokens 5–80, mixed prompts), max_slots=4: every output equals `generate()` alone, greedy and seeded (mixed temperature/top_p/top_k/repetition penalty). >= 10 requests joined mid-run; all 4 slots returned at the end. Stop-string request leaves early and frees its slot.
- **Counterfactual:** `alloc()` without `lengths[slot] = 0` -> crash "KV slot overflow: 274 > max_len 256" because reused slots kept growing. The overflow check turned a silent wrong-output bug into a loud one.
- **First look, NOT a benchmark** (one run, all 20 arrive at t=0, laptop CPU): continuous 260 steps, 15.9 s, 58.3 tok/s, mean batch 3.49; static (groups of 4) 344 steps, 18.6 s, 49.7 tok/s, 34% of computed rows wasted on finished sequences. Proper Poisson-arrival benchmark comes in week 4.

## 2026-10-12 — Steps 3.1–3.2 engine thread, random-arrival test
- **Design:** only the engine thread touches model/cache/scheduler (no locks). `submit()` validates synchronously (bad request -> ValueError -> 400) and `put_nowait` (full -> QueueFull -> 429). The engine pulls from the queue only while `len(waiting) < free slots`; draining everything would move the backlog into an unbounded list and make the queue limit meaningless.
- **Result:** 30 requests, shuffled order, Poisson gaps (mean 50 ms), half greedy / half seeded sampling, 4 slots: every output equals `generate()` alone, for 5 different arrival seeds.
- **Contention check (seed 0):** decode steps by batch size {1: 14, 2: 10, 3: 1, 4: 169}; peak queue depth 26. So the test really runs a full, backed-up batch, not one request at a time.
- **Production behaviours tested:** cancel mid-generation frees the slot within a step; deadline -> finish "timeout"; queue full -> QueueFull; an exception in one step fails only in-flight requests ("error") and the engine keeps serving.

## 2026-10-12 — Steps 3.3–3.5 OpenAI-compatible server, SSE, production behaviours
- **Result:** official `openai` client streams and non-streams unmodified; streamed text == non-streamed text == `generate()`; 400s in OpenAI error shape (n>1, temperature 5, empty messages, too long); 429 + Retry-After on full queue; disconnect after 3 chunks frees the slot within 5 s (200 tokens would take ~8 s).
- **Bug (mine, in a test):** 429 test got 400. The test built `Engine` without `decode`, and the server always adds the "\nUser:" stop string, so validation correctly rejected it. Fixed the test, not the server.
- **Design:** submit happens in the handler *before* the StreamingResponse starts, so overload/validation become real 429/400 status codes, not a broken 200 stream.
- **Design:** `on_token` runs on the engine thread and uses `loop.call_soon_threadsafe`; if the loop is closed it raises RuntimeError, which would have been caught by the engine as a failed step and killed *every* in-flight request. Caught locally instead and the request is cancelled.
- **Streaming detail:** text that could be the start of a stop string is held back. Seen live: "\n" waited 1 token (1127 -> 1280 ms) until "Assistant" proved it was not "\nUser:". Emoji test: " 🍕" is 2+ tokens and the first decodes to U+FFFD, so nothing is sent until the character completes.
- **Open question:** live streaming runs ~80 ms/token vs ~41 ms/token in the week-1 KV benchmark. Candidates: SlotView gather copy per layer, max_len=512 slot, engine/asyncio hop. Measure in week 4 rather than guess.

## 2026-10-12 — Step 3.6 /metrics (Week 3 done)
- **Result:** Prometheus text at `/metrics`: gauges (queue depth, waiting, batch size, free slots) read from the engine at scrape time; histograms TTFT / TPOT / e2e latency / decode batch size; counters for tokens, finished requests by reason, rejections by reason. Private registry per app so several apps in one process (tests) don't collide.
- **Bug found by the test:** the `n > 1` check returned 400 *before* the code that counted rejections, so that 400 was invisible in metrics. Fixed by routing every refusal through one `reject()` function.
- **Bug found by reading the output:** latency sums came out as 14.267000000000735, suspiciously round. On Windows `time.monotonic()` is GetTickCount64 with **15.6 ms resolution** (measured: 15.000 ms tick) vs `perf_counter` 0.2 us. Per-token times are 40-120 ms, so monotonic would have quantised TPOT by 15-40% and corrupted every week-4 benchmark. Switched engine timestamps and deadlines to `perf_counter`. Good interview story: a measurement bug caught before it produced numbers.
- **8 concurrent requests, 4 slots (one run, CPU):** mean TTFT ~1.8 s (half the requests queued for a slot), mean TPOT ~0.13 s/token at batch 4 vs ~0.08 single — per-step cost grows with batch size on CPU, but 4 tokens per step still beats 1.

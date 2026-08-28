# Experiment Analysis: TDBT2-02A

日期：2026-08-26

## 1. Run Integrity

- Run ID: `TDBT2-02A`
- Remote: `root@xj-member.bitahub.com:42181`
- Screen: `tdbt2_02a`
- Model: `facebook/opt-350m`
- Layers: `0,7,15,23`
- Operators: `qk,vo`
- Data source: `wikitext-2-raw-v1-arrow-cache-after-ImportError`
- Elapsed: `242.42 s`
- Result path: `results/remote-runs/TDBT2-02A/result.json`
- Console: `results/remote-runs/TDBT2-02A/console.log`

Clean-room invariants:

- `uses_qat_checkpoint = false`
- `uses_qat_logits = false`
- `uses_qat_latent_weights = false`
- `uses_qat_state_prior = false`
- `uses_quantized_point_operator_gradient = true`

NLL sanity:

- FP val NLL: `3.8957`
- direct PTQ val NLL: `8.8423`

The run is valid as a strict PTQ proxy experiment. It does not use QAT teacher information.

## 2. Raw Comparison Table

G-candidate variants:

| Layer | Op | Base val | QG val / impr | Endpoint-G val / impr | TDBT2-G val / impr | TDBT2-G vs endpoint | TDBT2-G untouched vs endpoint | Swaps G |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | qk | 0.102207 | 0.100557 / +1.61% | 0.100557 / +1.61% | 0.100557 / +1.61% | +0.00% | +0.00% | 14 |
| 0 | vo | 0.221083 | 0.200148 / +9.47% | 0.199867 / +9.60% | 0.199867 / +9.60% | +0.00% | +0.00% | 12 |
| 7 | qk | 0.162584 | 0.139255 / +14.35% | 0.136023 / +16.34% | 0.136023 / +16.34% | +0.00% | +0.00% | 13 |
| 7 | vo | 0.297459 | 0.290726 / +2.26% | 0.288783 / +2.92% | 0.288783 / +2.92% | +0.00% | +0.00% | 12 |
| 15 | qk | 0.180517 | 0.140666 / +22.08% | 0.140180 / +22.35% | 0.140180 / +22.35% | +0.00% | +0.00% | 14 |
| 15 | vo | 0.416151 | 0.411009 / +1.24% | 0.409029 / +1.71% | 0.409029 / +1.71% | +0.00% | +0.00% | 14 |
| 23 | qk | 0.172746 | 0.141671 / +17.99% | 0.136044 / +21.25% | 0.136044 / +21.25% | +0.00% | +0.00% | 12 |
| 23 | vo | 0.358471 | 0.355696 / +0.77% | 0.354425 / +1.13% | 0.354425 / +1.13% | +0.00% | +0.00% | 13 |

F-candidate variants:

| Layer | Op | Base val | Endpoint-F val / impr | TDBT2-F val / impr | TDBT2-F vs endpoint | Swaps F |
|---:|---|---:|---:|---:|---:|---:|
| 0 | qk | 0.102207 | 0.102150 / +0.05% | 0.102150 / +0.05% | +0.00% | 1 |
| 0 | vo | 0.221083 | 0.219898 / +0.54% | 0.219898 / +0.54% | +0.00% | 6 |
| 7 | qk | 0.162584 | 0.161273 / +0.81% | 0.161273 / +0.81% | +0.00% | 5 |
| 7 | vo | 0.297459 | 0.295242 / +0.75% | 0.295242 / +0.75% | +0.00% | 6 |
| 15 | qk | 0.180517 | 0.178618 / +1.05% | 0.178618 / +1.05% | +0.00% | 5 |
| 15 | vo | 0.416151 | 0.415820 / +0.08% | 0.415820 / +0.08% | +0.00% | 8 |
| 23 | qk | 0.172746 | 0.170616 / +1.23% | 0.170616 / +1.23% | +0.00% | 2 |
| 23 | vo | 0.358471 | 0.358015 / +0.13% | 0.358015 / +0.13% | +0.00% | 9 |

## 3. Gate Decision

Pre-registered B1 diagnostic gate:

> `TDBT2-G` or `TDBT2-F` must improve held-out operator distortion over `endpoint-beam` by at least 5% in at least 3/4 layer-operator pairs, with untouched-W direction not reversing.

Observed:

- `TDBT2-F` vs endpoint-F: `0/8` pairs pass.
- `TDBT2-G` vs endpoint-G: `0/8` pairs pass.
- TDBT2 and endpoint-greedy are exactly identical on all tested pairs.

Decision:

```text
B1 path/barrier gate: FAIL
B2 ternary mechanism ablation: DO NOT RUN
```

## 4. Findings

1. Support-swap candidate search has a real operator signal.

   `QG-one-shot` and endpoint-G improve direct PTQ on all eight layer-operator pairs. The gains are strongest for Q/K: `+1.61%`, `+14.35%`, `+22.08%`, and `+17.99%` for one-shot; endpoint-G reaches up to `+22.35%`.

2. The current barrier/trust path constraint adds no value.

   `TDBT2-G` equals `endpoint-greedy-G` on every pair, and `TDBT2-F` equals `endpoint-greedy-F` on every pair. The local trust constraint did not reject any endpoint-improving move that mattered.

3. Gradient-ranked candidates dominate forward-only salience.

   F variants improve only `0.05%` to `1.23%` on Q/K and less on V/O. G variants improve much more, especially for Q/K. This supports studying quantized-point gradient support projection, but not the path-barrier framing.

4. The experiment remains strict PTQ.

   No QAT artifacts were used. The result cannot be dismissed as QAT-teacher leakage.

## 5. Interpretation

The clean-room experiment argues against the current TDBT path thesis:

> The useful object in this harness is not a low-barrier discrete path. It is a gradient-ranked support projection endpoint.

This does not kill the broader ternary PTQ direction. It closes a narrower claim: with the current local trust definition, barrier-aware transport is not necessary beyond endpoint-greedy support selection.

## 6. Next Step Recommendation

Do not run `TDBT2-02B` as originally planned, because B2 was conditional on B1 passing.

The next idea should be reframed as:

```text
Ternary Quantized-Gradient Support Projection
```

Minimal next validation, if continued later:

- compare support-swap G against sign-flip-only G and random support-swap, but as a new clean-room plan, not as B2;
- add binary/no-zero control to test ternary specificity;
- evaluate whether Q/K gains transfer to end-to-end NLL after patching a small number of layers.

This should be a new `TDBT3-*` or renamed direction, because `TDBT2` path-barrier claim failed.


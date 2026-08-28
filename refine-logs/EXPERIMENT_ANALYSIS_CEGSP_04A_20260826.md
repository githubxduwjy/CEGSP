# Experiment Analysis: CEGSP-04A Edit-Budget Sensitivity

## Summary

- Runs: `CEGSP-04A-E16`, `E32`, `E64`, `E128`
- Model: `facebook/opt-350m`
- Data: O0 WikiText fit/val/untouched plus report-only C4 validation
- Sweep: `max-edits ∈ {16, 32, 64, 128}`
- k: `{4, 6}`
- Status: all complete
- Nonfinite: none detected
- Runtime: `49.47s`, `50.93s`, `52.38s`, `54.16s`

CEGSP-04A passes the budget-sensitivity gate: the result is not a single lucky `max-edits=64` setting. Every top-k family improves val, WikiText untouched, and C4 untouched across all four edit budgets.

## Raw budget table

Negative is better. Deltas are vs direct ternary.

| max edits | patch set | val Δ | WikiText untouched Δ | C4 untouched Δ |
|---:|---|---:|---:|---:|
| 16 | support top4 | -0.096361 | -0.091359 | -0.156148 |
| 16 | signflip top4 | -0.128018 | -0.116133 | -0.169034 |
| 16 | joint top4 | -0.096361 | -0.091359 | -0.156148 |
| 16 | support top6 | -0.136962 | -0.122514 | -0.187879 |
| 16 | signflip top6 | -0.124839 | -0.112246 | -0.197740 |
| 16 | joint top6 | -0.136962 | -0.122514 | -0.187879 |
| 32 | support top4 | -0.192585 | -0.158748 | -0.126881 |
| 32 | signflip top4 | -0.165274 | -0.139222 | -0.192757 |
| 32 | joint top4 | -0.192890 | -0.154677 | -0.123612 |
| 32 | support top6 | -0.210403 | -0.183081 | -0.160688 |
| 32 | signflip top6 | -0.138152 | -0.115488 | -0.214821 |
| 32 | joint top6 | -0.212436 | -0.179794 | -0.157730 |
| 64 | support top4 | -0.265143 | -0.226320 | -0.149488 |
| 64 | signflip top4 | -0.255191 | -0.218416 | -0.145943 |
| 64 | joint top4 | -0.271754 | -0.225032 | -0.145396 |
| 64 | support top6 | -0.297465 | -0.272000 | -0.192771 |
| 64 | signflip top6 | -0.302090 | -0.270071 | -0.198171 |
| 64 | joint top6 | -0.322584 | -0.280708 | -0.174876 |
| 128 | support top4 | -0.324017 | -0.276042 | -0.193622 |
| 128 | signflip top4 | -0.282360 | -0.275030 | -0.246487 |
| 128 | joint top4 | -0.288312 | -0.274868 | -0.171015 |
| 128 | support top6 | -0.375416 | -0.324008 | -0.184744 |
| 128 | signflip top6 | -0.323945 | -0.314968 | -0.241507 |
| 128 | joint top6 | -0.335574 | -0.314799 | -0.228521 |

## All-layer controls

| max edits | support all val/W/C4 Δ | signflip all val/W/C4 Δ |
|---:|---|---|
| 16 | -0.060502 / -0.048080 / -0.187561 | -0.141122 / -0.128825 / -0.182880 |
| 32 | +0.023043 / +0.029354 / -0.108057 | -0.056560 / -0.043829 / -0.097333 |
| 64 | +0.121583 / +0.133234 / +0.016003 | +0.038583 / +0.052664 / +0.037108 |
| 128 | +0.176448 / +0.233912 / +0.127409 | +0.073200 / +0.121033 / +0.115448 |

## Gate judgement

Primary gate: pass.

- There is a broad stable region, not a single lucky edit count.
- Top-k variants improve all three reported splits at all four edit budgets.
- Runtime increases only mildly from `49.47s` to `54.16s`.

## Interpretation

1. **Budget sensitivity is favorable.**  
   The mechanism works from 16 to 128 edits per layer. This reduces concern that CEGSP's positive results are caused by a lucky hand-picked `64`.

2. **More edits help top-k WikiText, but C4 is not monotonic for every family.**  
   WikiText untouched generally improves as edit budget grows. C4 has a more nuanced pattern: signflip top4/top6 and joint top6 remain strong, while support top6 peaks earlier. This argues against blindly maximizing edits.

3. **All-layer editing is dangerous when edit budget increases.**  
   With 16 edits, all-layer controls can improve, but at 64/128 they degrade both WikiText and C4. This confirms the method should constrain both:
   - number of edited layers;
   - number of edits per layer.

4. **Current frozen default remains reasonable.**  
   `max-edits=64`, `k=4/6` is not uniquely lucky, but it sits in a good middle region. For a paper method, a conservative default could be `k=4 or 6` with an adaptive edit budget selected only on validation.

## Updated claim strength

After CEGSP-03B and CEGSP-04A, the clean-room CEGSP branch supports a stronger diagnostic claim:

> At deployed ternary PTQ weights, a small number of CE-gradient-guided ternary edits can reliably improve held-out NLL on OPT-350M, and the effect transfers from WikiText selection to C4 validation under multiple offsets and multiple edit budgets.

It still does not yet support a full paper claim about large LLM SOTA or strong PTQ-baseline integration.

## Next minimal experiment

`CEGSP-04B`: second cached model sanity check, preferably `facebook/opt-125m` if already cached. Use proportional layer budgets instead of copying 24-layer `k=6` literally.

Purpose: determine whether the signal is specific to OPT-350M or is at least model-size robust within the OPT family.

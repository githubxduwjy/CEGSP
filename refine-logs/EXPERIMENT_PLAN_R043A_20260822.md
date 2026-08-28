# R043a Plan: GPTQ Transfer Screen

R042c established that validation-gated `T` refinement improves untouched block outputs. R043a tests whether this local effect survives official GPTQ residual propagation and autoregressive perplexity evaluation before paying the roughly 32-layer full-search cost.

## Configuration

- Official no-SSR GPTQ pipeline, LLaMA-2-7B, group/block size 128.
- WikiText2 calibration, seed 0, 8 sequences of length 2048.
- Apply gated `T` refinement only in layers 0, 10, 20, and 31, across all seven projections; all other layers use official ATQ.
- Within each refined block: first 6 samples propose `T`, last 2 accept/reject, then freeze `T` and refit `alpha,mu` on all 8 samples.
- Search budget remains four one-coordinate proposals per output row.
- Evaluation: full WikiText2 and C4 PPL at sequence length 2048.
- Matched control: official GPTQ no-SSR R037, W2/C4 25.8104/66.7370, same model, seed, sample count, block size, and code revision.

## Transfer gate

Launch all-layer gated GPTQ only if R043a is finite, remains within 24 GB, and improves both WikiText2 and C4 PPL over the matched official no-SSR control. This run is a transfer screen, not a final headline comparison; a pass must still be followed by all-layer no-SSR and then an SSR-compatible comparison.


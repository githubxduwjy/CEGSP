# Experiment Plan: CEGSP-01B k-Sweep

日期：2026-08-26

## Purpose

`CEGSP-01A` showed strong top-k improvements but all-layer edits degraded NLL. `CEGSP-01B` tests whether this is a stable layer-budget phenomenon and whether the method should be support-only, signflip-only, or joint support/sign editing.

## Setup

- Same as `CEGSP-01A`.
- k sweep: `1,2,4,6,8,12,16,24`
- Patch families:
  - support-only top-k
  - signflip-only top-k
  - joint per-layer best top-k

## Gate

- A broad k range should improve untouched NLL versus direct ternary.
- If joint top-k beats both support-only and signflip-only, the next method should be joint ternary edit selection.
- If only support or only signflip dominates, keep the method narrower.


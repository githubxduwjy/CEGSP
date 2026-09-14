# Result Schema

TernRefine experiment JSON files should expose enough metadata to audit the
paper claims without reading logs.

Recommended top-level fields:

| Field | Meaning |
| --- | --- |
| `status` | `complete` for finished runs |
| `protocol` | model, initializer, scope, schedule, and data role information |
| `state` | frozen ternary state hash, sidecar/parity information |
| `data` | fit and selection split indices or hashes |
| `apg_curve` | validation losses for evaluated prefixes |
| `selected_patch` or patch path | selected K, relocations, changed coordinates |
| `metrics` | W2/C4/downstream metrics after selection |
| `gate` | boolean audit fields used for claim checking |

Recommended `gate` fields:

- `state_parity_pass`
- `same_frozen_pt2_state`
- `fit_disjoint_from_selection`
- `one_backward_each`
- `no_rerank`
- `all_selected_patches_legal`
- `all_selected_patches_finite`
- `all_exact_relocations`
- `all_exact_changed_coordinates`

Qwen PT2 runs that bypass the strict parity gate must explicitly record this as
a conditional endpoint.


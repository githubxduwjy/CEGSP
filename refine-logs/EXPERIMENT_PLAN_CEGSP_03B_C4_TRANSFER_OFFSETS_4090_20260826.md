# CEGSP-03B：C4 transfer 的 offset 复现

## Motivation

CEGSP-03A 在 offset 0 上通过了 WikiText-to-C4 transfer gate，但单次 cross-data 成功仍可能是 split luck。CEGSP-03B 复用完全相同的方法、超参和 gate，只在 CEGSP-02A 已预先使用过的另外两个 offset 上复现。

## Fixed settings

- 模型：`facebook/opt-350m`
- 设备：RTX 4090 24GB
- 方法：CE gradient at deployed ternary weights；support/signflip/joint top-k edits
- k：`{4, 6}`
- 编辑预算：`max-edits=64`
- 梯度 batch：`1`
- QAT teacher/checkpoint/logits/latent weights：全部禁止
- C4：只作 untouched transfer test，不参与选择

## Runs

| run | WikiText fit offset | WikiText val offset | C4 token offset |
|---|---:|---:|---:|
| `CEGSP-03B-O1` | 4096 | 4096 | 4096 |
| `CEGSP-03B-O2` | 8192 | 8192 | 8192 |

## Gate

Primary:

- 在两个 offset 上，至少一个预注册 joint candidate（`joint top4` 或 `joint top6`）同时满足 val、WikiText untouched、C4 untouched delta `<= 0`。

Secondary:

- 如果 joint 不稳定，但 support-only 或 signflip-only 在两个 offset 上稳定通过，则下一步冻结更保守 family，而不是继续扩大搜索。
- 如果 WikiText 稳定改善但 C4 反复退化，则判定当前 selection 存在跨分布风险，下一步只允许做 selection regularization / multi-distribution validation，不允许直接扩模型。

## Stop rule

若 O1/O2 均出现 C4 退化，不否定 CE-gradient ternary editing 主线，但停止“单分布 WikiText selection 可直接泛化到 C4”的子主张。

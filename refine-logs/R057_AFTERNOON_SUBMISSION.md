# R057A 下午云端提交指南

## 需要同步的文件

- `remote-tools/r048_distribution_holdout_gate.py`
- `remote-tools/r057a_analyze_hparams.py`
- `remote-tools/run_r057a_hparam_screen.sh`

不要提交旧的 `run_r057_checkpoint_gate.sh`；它已被新的超参数方案替代。

## 本地同步

将 `<PORT>` 替换为下午实例的 SSH 端口：

```bash
rsync -av -e "ssh -p <PORT>" \
  remote-tools/r048_distribution_holdout_gate.py \
  remote-tools/r057a_analyze_hparams.py \
  remote-tools/run_r057a_hparam_screen.sh \
  root@xj-member.bitahub.com:/root/PT2-LLM-official/remote-tools/
```

## 远端预检

```bash
cd /root/PT2-LLM-official
/root/PT2-LLM/venv/bin/python -m py_compile \
  remote-tools/r048_distribution_holdout_gate.py \
  remote-tools/r057a_analyze_hparams.py
bash -n remote-tools/run_r057a_hparam_screen.sh
test -d /root/models/Llama-2-7b-hf
nvidia-smi
```

## 启动 R057A

```bash
cd /root/PT2-LLM-official
chmod +x remote-tools/run_r057a_hparam_screen.sh
mkdir -p aris-runs/r057a_hparam_gate_20260824
screen -dmS r057a_hparam_gate bash -lc \
  'cd /root/PT2-LLM-official && remote-tools/run_r057a_hparam_screen.sh > aris-runs/r057a_hparam_gate_20260824/launch.log 2>&1'
```

## 健康检查

```bash
screen -ls
tail -n 50 /root/PT2-LLM-official/aris-runs/r057a_hparam_gate_20260824/launch.log
find /root/PT2-LLM-official/aris-runs/r057a_hparam_gate_20260824 \
  -maxdepth 2 -name metrics.json -printf '%p %s\n' | sort
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
```

完成标志为：

```text
/root/PT2-LLM-official/aris-runs/r057a_hparam_gate_20260824/summary.json
```

## R057A 后的硬规则

- `decision=support_a`：冻结 `summary.json.selected.hyperparameters`，再生成并启动 R057B。
- `decision=fail_a_generalization`：记录负结果，不用 test 重新选 H1--H4。
- `decision=inconclusive_overconservative`：停止 R057B；说明 gate 过严，不放宽 epsilon。
- 所有 H0--H4 目录必须保留，无论结果正负。

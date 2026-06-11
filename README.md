# DiagPRM

**DiagPRM** is a Knowledge-Graph-Guided Process Reward framework for training LLMs to perform structured multi-turn medical diagnostic dialogues.

Built on top of [verl](https://github.com/volcengine/verl), DiagPRM trains a Doctor Agent that conducts evidence-driven clinical interviews via Turn-level GRPO — without any neural critic or value network.

---

## Overview

```
Patient Chief Complaint
        │
        ▼
┌─────────────────────────────────────────┐
│           Doctor Agent (LLM)            │
│  <think> → <hypothesis_state> →         │
│  <action: continue/switch/diagnose>     │
│  <question> or <diagnosis>              │
└─────────────┬───────────────────────────┘
              │ question
              ▼
┌─────────────────────────────────────────┐
│        Patient Simulator                │
│  (atomic_facts + LLM / rule-based)      │
└─────────────┬───────────────────────────┘
              │ answer
              ▼
┌─────────────────────────────────────────┐
│         Turn-level Reward               │
│  r(k) = Δkg + r_hyp + r_switch + r_diag│
│  (KG coverage delta + hypothesis acc.)  │
└─────────────────────────────────────────┘
```

The key insight: each question the doctor asks is scored **immediately** using a static Knowledge Graph query — no waiting until episode end, no neural reward model required.

---

## Reward Design

Each turn is rewarded by:

| Component | Formula | Description |
|-----------|---------|-------------|
| `Δ_kg` | `β × (cov_after − cov_before)` | Dense KG coverage delta |
| `r_hyp` | `±γ₁` | Primary hypothesis correctness |
| `r_switch` | `±λ` | Bonus/penalty for hypothesis switching |
| `r_diag` | `r_max` or `−1` | Outcome reward at final turn |
| `r_fmt` | `format_score` | Structural format compliance |

> **Total:** `r(k) = r_fmt + Δ_kg + r_hyp + r_switch + r_diag`

Default coefficients: `β=1.0, γ₁=0.3, λ=0.5, r_max=2.0, τ=0.5`

---

## Quick Start

### 1. Prepare Data

```bash
python recipe/diagprm/data_utils/prepare_diagprm_data.py \
    --input /path/to/merged_train_dataset.jsonl \
    --output_dir ./data \
    --val_ratio 0.05 \
    --kg_path /path/to/master_kg.json
```

This produces `data/diagprm_train.parquet` and `data/diagprm_val.parquet`.

**Input format** (`merged_train_dataset.jsonl`):
```json
{
  "prompt": [{"role": "user", "content": "A patient presents with ..."}],
  "ground_truth": {"answer": "A", "answer_info": "Pneumonia"},
  "extra_info": {"atomic_facts": ["Has fever", "Has cough", ...]}
}
```

### 2. Train

```bash
# Set required paths
export ACTOR_LOAD=/path/to/Qwen3-8B-Instruct
export KG_PATH=/path/to/master_kg.json

bash recipe/diagprm/run_diagprm.sh
```

Supported base models: `Qwen3-1.7B / 4B / 8B` (Instruct variants recommended).

### 3. Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_turns` | 10 | Max dialogue turns per episode |
| `n_resp_per_prompt` | 4 | GRPO rollouts per prompt (G) |
| `actor_lr` | 1e-6 | Learning rate |
| `kl_loss_coef` | 0.01 | KL divergence penalty |
| `train_batch_size` | 64 | Global training batch size |

---

## Doctor Agent Output Format

Every turn the Doctor must produce:

```xml
<think>
[Chain-of-thought reasoning about current evidence and hypothesis ranking]
</think>
<hypothesis_state>
  <hypothesis name="Pneumonia">
    <confirmed>[fever, cough, shortness of breath]</confirmed>
    <pending>[chest x-ray findings, sputum color]</pending>
  </hypothesis>
  <hypothesis name="Bronchitis">
    <confirmed>[cough]</confirmed>
    <pending>[fever, wheeze]</pending>
  </hypothesis>
</hypothesis_state>
<action>continue</action>
<question>Do you have any yellow or green sputum when you cough?</question>
```

When ready to diagnose:
```xml
<action>diagnose</action>
<diagnosis>Community-acquired Pneumonia</diagnosis>
```

---

## Project Structure

```
codes/
├── recipe/diagprm/
│   ├── diagprm_agent_loop.py      # Multi-turn Doctor-Patient agent loop
│   ├── diagprm_reward_fn.py       # Turn-level reward computation
│   ├── diagprm_reward_manager.py  # verl reward manager integration
│   ├── diagprm_trainer.py         # Custom GRPO trainer
│   ├── diagprm_main.py            # Entry point
│   ├── kg_utils.py                # KG loading & coverage calculation
│   ├── run_diagprm.sh             # Training launch script
│   ├── algo/
│   │   └── diagprm_algos.py       # Turn-level GRPO algorithm
│   ├── prompts/
│   │   ├── doctor.py              # Doctor system/initial prompts
│   │   ├── patient.py             # Patient simulator prompt
│   │   └── verifier.py            # Question quality verifier prompt
│   ├── data_utils/
│   │   └── prepare_diagprm_data.py  # Data preprocessing
│   └── config/
│       └── diagprm_trainer.yaml   # Full training config
```

---

## Knowledge Graph Format

`master_kg.json` maps each disease to its weighted symptom profile:

```json
{
  "pneumonia": {
    "fever": 0.9,
    "cough": 0.85,
    "shortness of breath": 0.8,
    "chest pain": 0.6
  },
  "bronchitis": {
    "cough": 0.95,
    "wheeze": 0.7,
    "low-grade fever": 0.5
  }
}
```

Legacy list format is also supported (all weights default to 1.0).

---

## Dependencies

```bash
# Core
pip install verl torch transformers

# For async rollout
pip install sglang vllm

# For data processing
pip install pandas tqdm
```

> DiagPRM requires `verl >= 0.5` and is tested with `Qwen3` series models.

---

## Citation

If you find this work useful, please consider citing our paper (ICLR 2027, under review).

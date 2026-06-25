# DiagPRM

**DiagPRM** is a framework for **KG-supervised turn-level credit assignment for diagnostic dialogue**. It trains a Doctor Agent to conduct evidence-driven multi-turn clinical interviews via Turn-level GRPO — without any neural critic or value network.

Built on top of [verl](https://github.com/volcengine/verl), DiagPRM uses a hidden oracle Knowledge Graph (KG) as the external process supervisor: at each turn, the KG coverage delta $\Delta_\text{KG}$ quantifies exactly how much new diagnostic evidence the doctor's question uncovered, providing dense turn-level credit assignment without any model self-evaluation.

---

## Overview

```
Patient Chief Complaint
        │
        ▼
┌─────────────────────────────────────────┐
│           Doctor Agent (LLM)            │
│  {"action": "ask",  "question": "..."}  │
│  {"action": "diagnose", "diagnosis":"…"}│
└─────────────┬───────────────────────────┘
              │ question
              ▼
┌─────────────────────────────────────────┐
│        Patient Simulator                │
│  (hidden symptom facts + LLM)           │
│  returns: natural-language answer       │
│           + hidden fact_id (to Reward)  │
└─────────────┬───────────────────────────┘
              │ answer (text only to Doctor)
              ▼
┌─────────────────────────────────────────┐
│   KG-Supervised Turn-level Reward       │
│  r_turn(k) = Δ_KG(k)   [dense, /turn]  │
│  r_diag    = outcome    [sparse, /ep.]  │
│  advantage  = A_turn(k) + α·A_diag      │
└─────────────────────────────────────────┘
```

The key insight: the **KG coverage delta** $\Delta_\text{KG}(k) = \Phi_\text{KG}(S_k) - \Phi_\text{KG}(S_{k-1})$ provides exact turn-level credit assignment — rewarding only the first confirmation of a new ground-truth symptom, with no neural critic and no model self-evaluation.

---

## Reward Design

DiagPRM implements **KG-supervised turn-level credit assignment** with two components:

| Component | Symbol | Formula / Trigger | Description |
|-----------|--------|-------------------|-------------|
| KG coverage delta | `Δ_KG(k)` | `Φ_KG(S_k) − Φ_KG(S_{k−1})` | **Dense turn-level reward**: positive only when the doctor's question confirms a new ground-truth symptom in the oracle KG |
| Diagnosis outcome | `r_diag` | Episode end: correct + coverage ≥ τ + new_facts ≥ m | **Sparse outcome reward**: guards against premature diagnosis |

**Advantage mixing** (Turn-level GRPO):
```
Â(k, i) = Â_turn(k, i)  +  α · Â_diag(i)
```
- `Â_turn`: within-group normalized `Δ_KG(k)` across G rollouts at the same turn position
- `Â_diag`: trajectory-group normalized `r_diag` across G rollouts

> **No neural critic. No model self-evaluation. All reward signals come from the hidden oracle KG or ground-truth labels.**

Default coefficients: `β=1.0, R_max=2.0, τ=0.5, m=2, α=0.3`

> **Ablation note**: `r_hyp` (hypothesis tracking reward) is NOT part of the main method. It appears only in the Hypothesis tracking ablation condition.

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

In the **main method**, the Doctor outputs clean medical dialogue actions only — no CoT, no hypothesis state field. This keeps RL focused on "does the question elicit new diagnostic evidence?" and "is the final diagnosis correct?":

```json
{"action": "ask", "question": "Do you have any yellow or green sputum when you cough?"}
```

When ready to diagnose:
```json
{"action": "diagnose", "diagnosis": "community-acquired pneumonia"}
```

> **Hypothesis tracking** (`<think>` + `<hypothesis_state>` + `r_hyp`) is an **ablation condition**, not the main method. See the experiment matrix below.

---

## Experiment Matrix

The table below covers all reported conditions. Results are **TBD** (to be filled after experiments).

| Condition | `Δ_KG` reward | `r_diag` | `α` (diag signal) | KG tool (Doctor-visible) | Hypothesis tracking + `r_hyp` | Acc ↑ | #Turn ↓ | KG Cov ↑ |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Outcome-only GRPO** | ✗ | ✓ | — | ✗ | ✗ | TBD | TBD | TBD |
| **DiagPRM (main)** | ✓ | ✓ | 0.3 | ✗ | ✗ | TBD | TBD | TBD |
| **w/o Δ_KG** | ✗ | ✓ | 0.3 | ✗ | ✗ | TBD | TBD | TBD |
| **α = 0** | ✓ | ✓ | 0 | ✗ | ✗ | TBD | TBD | TBD |
| **KG tool only** | ✗ | ✓ | — | ✓ | ✗ | TBD | TBD | TBD |
| **+ Hypothesis tracking** | ✓ | ✓ | 0.3 | ✗ | ✓ | TBD | TBD | TBD |

**Notes:**
- *Outcome-only GRPO*: standard trajectory-level GRPO, no turn-level KG signal — the primary baseline showing the value of KG-supervised credit assignment.
- *w/o Δ_KG*: removes the KG coverage delta; retains `r_diag` and advantage mixing — isolates the contribution of dense KG-supervised turns.
- *α = 0*: turn-level KG signal only, no trajectory-level diagnosis signal — tests whether global outcome signal is necessary.
- *KG tool only*: Doctor can query KG at inference time (observable), but no turn-level KG process reward — separates oracle supervision from observation.
- *+ Hypothesis tracking*: adds explicit `<think>` + `<hypothesis_state>` output format and `r_hyp` reward — tests whether explicit hypothesis management adds value beyond KG-supervised credit assignment.

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

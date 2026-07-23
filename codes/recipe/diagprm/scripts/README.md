# DiagPRM Scripts

## Recommended Atomic-Fact Flow

Use these scripts for our method in the ATPO medical multi-turn setting. The
doctor-side data and prompt are aligned to ATPO; the method difference is the
hidden atomic-fact environment plus our turn-level reward / advantage.

- `build_atomic_fact_rl_dataset.py`: convert ATPO-format JSONL datasets into
  parquet for RL/eval, preserving the released dataset prompt and storing hidden
  `atomic_facts` for reward computation.
- `run_atomic_fact_sft_qwen3.sh`: SFT on the released ATPO
  `sft_training_data.jsonl` prompt/response strings as-is. It does not rewrite
  prompts, strip `<think>`, inject `/no_think`, or add a turn-limit sentence.
- `run_atomic_fact_rl.sh`: RL with ATPO-aligned prompt/data and our
  `AtomicFactAgentLoop` + `AtomicFactRewardManager`.
- `run_eval_atomic_fact_checkpoint.sh`: evaluate an SFT/RL checkpoint on
  ATPO-format MedQA / MedMCQA / MedicalExam splits. Default settings use
  dataset prompts, 10 max turns, Qwen thinking enabled, and an LLM patient.
- `run_test_atomic_fact_sft_checkpoint.sh` and
  `test_atomic_fact_sft_checkpoint.py`: small checkpoint smoke tests.
- `patient_vllm_utils.sh` and `start_patient_vllm.sh`: helper scripts for
  launching a local OpenAI-compatible Qwen3 patient/verifier vLLM.

## KG / DiagPRM Flow

These scripts belong to the earlier KG-supervised diagnostic dialogue line.

- `generate_diagprm_sft_from_kg.py`: generate DiagPRM SFT trajectories from KG
  cases.
- `run_generate_diagprm_sft_teacher.sh`: single-process wrapper for SFT
  trajectory generation.
- `run_vllm_sft_nanqi4.sh`: launch 4 vLLM servers and sharded SFT trajectory
  generation on nanqi4.
- `run_vllm_sft_nanqi4_continue_missing.sh`: continue generation while excluding
  previously attempted case ids.
- `build_sft_exclude_case_ids.py`: build exclude lists from partial generation
  runs.
- `merge_diagprm_sft_runs.py`: merge, deduplicate, and filter SFT generation
  outputs.
- `run_diagprm_sft_qwen3.sh`: SFT with merged DiagPRM KG-dialogue data.
- `run_test_diagprm_sft_checkpoint.sh` and `test_diagprm_sft_checkpoint.py`:
  merge/test a saved DiagPRM SFT checkpoint.

## Common Disease KG Augment

- `build_common_disease_augments_from_med_sources.py`: mine common-disease fact
  candidates from local MEDIQ/MedQA/MedicalExam/MedMCQA/PrimeKG sources.
- `curate_common_disease_augments_from_qa.py`: normalize QA-mined case facts
  into disease-level patient-observable augment facts.
- `build_common_disease_augmented_kg.py`: build an augmented DiagPRM dataset
  directory from `clean_v2` and an augment JSON.
- `build_common_disease_rl_dataset.py`: build a common-disease-only RL dataset
  with multiple chief-complaint scenarios per disease.

## KG RL Runs

- `run_diagprm_common_rl.sh`: run DiagPRM RL on the common-disease-only dataset.
- `run_diagprm_fullkg_rl.sh`: run DiagPRM RL on the full `clean_v2` KG dataset.
- `run_diagprm_common_rl_outcome_only.sh`: common-disease outcome-only ablation.
- `run_eval_diagprm_common_rl_checkpoint.sh`: evaluate a common-disease RL
  checkpoint on validation/test.

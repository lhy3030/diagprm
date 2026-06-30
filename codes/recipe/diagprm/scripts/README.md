# DiagPRM Scripts

DiagPRM-specific data, SFT, vLLM collection, checkpoint test, and common-disease
augmentation utilities live here. Generic verl utilities remain in `codes/scripts`.

## Common Disease KG Augment

- `build_common_disease_augments_from_med_sources.py`: mine common-disease fact candidates from local MEDIQ/MedQA/MedicalExam/MedMCQA/PrimeKG sources.
- `curate_common_disease_augments_from_qa.py`: normalize QA-mined case facts into disease-level patient-observable augment facts.
- `build_common_disease_augmented_kg.py`: build a new augmented DiagPRM dataset directory from `clean_v2` and an augment JSON.

## SFT Data

- `generate_diagprm_sft_from_kg.py`: generate DiagPRM SFT trajectories from KG cases.
- `run_generate_diagprm_sft_teacher.sh`: single-process wrapper for SFT trajectory generation.
- `run_vllm_sft_nanqi4.sh`: launch 4 vLLM servers and sharded SFT trajectory generation on nanqi4.
- `run_vllm_sft_nanqi4_continue_missing.sh`: continue generation while excluding previously attempted case ids.
- `build_sft_exclude_case_ids.py`: build exclude lists from partial generation runs.
- `merge_diagprm_sft_runs.py`: merge, deduplicate, and filter SFT generation outputs.

## SFT Training And Checkpoint Test

- `run_diagprm_sft_qwen3.sh`: train Qwen3 SFT with the merged DiagPRM SFT dataset.
- `run_test_diagprm_sft_checkpoint.sh`: merge/test a saved SFT checkpoint.
- `test_diagprm_sft_checkpoint.py`: generation smoke test for a merged checkpoint.
